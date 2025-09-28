package recognizer

import (
	"errors"
	"fmt"
	"image"
	"time"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/mempool"
	onnx "github.com/MeKo-Tech/pogo/internal/onnx"
	onnxrt "github.com/yalue/onnxruntime_go"
)

// Result represents the recognition output for a region.
type Result struct {
	Text            string
	Confidence      float64
	CharConfidences []float64
	Indices         []int
	Rotated         bool
	Width           int
	Height          int
	TimingNs        struct {
		Preprocess int64
		Model      int64
		Decode     int64
		Total      int64
	}
}

// RecognizeRegion performs end-to-end preprocessing + inference + decoding for a single region.
func (r *Recognizer) RecognizeRegion(img image.Image, region detector.DetectedRegion) (*Result, error) {
	if img == nil {
		return nil, errors.New("input image is nil")
	}

	totalStart := time.Now()

	// 1) Crop and optionally rotate (using per-line orientation if configured)
	t0 := time.Now()
	var patch image.Image
	var rotated bool
	var err error
	if r.textLineOrienter != nil {
		patch, rotated, err = CropRegionImageWithOrienter(img, region, r.textLineOrienter, true)
	} else {
		patch, rotated, err = CropRegionImage(img, region, true)
	}
	if err != nil {
		return nil, fmt.Errorf("crop region: %w", err)
	}

	// 2) Resize with fixed height and padding
	targetH := r.config.ImageHeight
	if targetH <= 0 {
		targetH = 32
	}
	resized, outW, outH, err := ResizeForRecognition(patch, targetH, r.config.MaxWidth, r.config.PadWidthMultiple)
	if err != nil {
		return nil, fmt.Errorf("resize: %w", err)
	}

	// 3) Normalize to tensor (pooled)
	tensor, buf, err := NormalizeForRecognitionWithPool(resized)
	if err != nil {
		return nil, fmt.Errorf("normalize: %w", err)
	}
	preprocessNs := time.Since(t0).Nanoseconds()

	// 4) Run model
	r.mu.RLock()
	session := r.session
	r.mu.RUnlock()
	if session == nil {
		return nil, errors.New("recognizer session is nil")
	}
	m0 := time.Now()
	inputTensor, err := onnxrt.NewTensor(onnxrt.NewShape(tensor.Shape...), tensor.Data)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer func() { _ = inputTensor.Destroy() }()
	outputs := []onnxrt.Value{nil}
	if err := session.Run([]onnxrt.Value{inputTensor}, outputs); err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			if o != nil {
				_ = o.Destroy()
			}
		}
	}()
	mempool.PutFloat32(buf)
	modelNs := time.Since(m0).Nanoseconds()

	// 5) Decode CTC greedy
	d0 := time.Now()
	outTensor := outputs[0]
	floatTensor, ok := outTensor.(*onnxrt.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("expected float32 tensor, got %T", outTensor)
	}
	data := floatTensor.GetData()
	shape := outTensor.GetShape()

	// Determine classes/time order
	// Heuristic: try both and choose one where classes == charset+1
	classesGuess := r.charset.Size() + 1
	classesFirst := determineClassesFirst(shape, classesGuess)
	blankIndex := 0 // PaddleOCR CTC typically uses blank=0
	decoded := DecodeCTCGreedy(data, shape, blankIndex, classesFirst)
	if len(decoded) == 0 {
		return nil, errors.New("empty decoded output")
	}
	seq := decoded[0]

	// Map to text
	runes := make([]rune, 0, len(seq.Collapsed))
	for _, idx := range seq.Collapsed {
		// map index to token; idx in [0..classes-1], 0 is blank
		ch := r.charset.LookupToken(idx - 1) // shift by -1 to skip blank
		if ch == "" {
			continue
		}
		runes = append(runes, []rune(ch)...)
	}
	text := string(runes)
	conf := SequenceConfidence(seq.CollapsedProb)
	decodeNs := time.Since(d0).Nanoseconds()
	totalNs := time.Since(totalStart).Nanoseconds()

	return &Result{
		Text:            text,
		Confidence:      conf,
		CharConfidences: seq.CollapsedProb,
		Indices:         seq.Collapsed,
		Rotated:         rotated,
		Width:           outW,
		Height:          outH,
		TimingNs:        struct{ Preprocess, Model, Decode, Total int64 }{Preprocess: preprocessNs, Model: modelNs, Decode: decodeNs, Total: totalNs},
	}, nil
}

// RecognizeBatch processes multiple regions on the same source image.
func (r *Recognizer) RecognizeBatch(img image.Image, regions []detector.DetectedRegion) ([]Result, error) {
	if img == nil {
		return nil, errors.New("input image is nil")
	}
	if len(regions) == 0 {
		return nil, errors.New("no regions provided")
	}

	// Preprocess patches
	targetH := r.config.ImageHeight
	if targetH <= 0 {
		targetH = 32
	}

	// Convert each to normalized tensor with potential varying widths; since ONNX Runtime
	// expects same input width across batch, we first compute per-patch resized widths, then
	// pad each to the maximum width in this batch (respecting PadWidthMultiple).
	type prep struct {
		img     image.Image
		rotated bool
		w, h    int
	}
	prepped := make([]prep, len(regions))
	maxW := 0
	for i, reg := range regions {
		var patch image.Image
		var rotated bool
		var err error
		if r.textLineOrienter != nil {
			patch, rotated, err = CropRegionImageWithOrienter(img, reg, r.textLineOrienter, true)
		} else {
			patch, rotated, err = CropRegionImage(img, reg, true)
		}
		if err != nil {
			return nil, fmt.Errorf("crop region %d: %w", i, err)
		}
		resized, outW, outH, err := ResizeForRecognition(patch, targetH, r.config.MaxWidth, r.config.PadWidthMultiple)
		if err != nil {
			return nil, fmt.Errorf("resize region %d: %w", i, err)
		}
		prepped[i] = prep{img: resized, rotated: rotated, w: outW, h: outH}
		if outW > maxW {
			maxW = outW
		}
	}

	// Pad to max width where needed
	batchTensors := make([][]float32, len(prepped))
	bufs := make([][]float32, len(prepped))
	for i, p := range prepped {
		canvas := p.img
		if p.w < maxW {
			padded := image.NewRGBA(image.Rect(0, 0, maxW, p.h))
			src := toRGBA(p.img)
			sb := src.Bounds()
			for y := range sb.Dy() {
				for x := range sb.Dx() {
					padded.Set(x, y, src.At(sb.Min.X+x, sb.Min.Y+y))
				}
			}
			canvas = padded
		}
		ten, buf, err := NormalizeForRecognitionWithPool(canvas)
		if err != nil {
			return nil, fmt.Errorf("normalize region %d: %w", i, err)
		}
		batchTensors[i] = ten.Data
		bufs[i] = buf
	}

	// Build batch tensor [N, C, H, W]
	if len(batchTensors) == 0 {
		return nil, errors.New("no tensors prepared")
	}
	tensor, err := onnx.NewBatchImageTensor(batchTensors, 3, prepped[0].h, maxW)
	if err != nil {
		return nil, fmt.Errorf("build batch tensor: %w", err)
	}

	// Run model
	r.mu.RLock()
	session := r.session
	r.mu.RUnlock()
	if session == nil {
		return nil, errors.New("recognizer session is nil")
	}
	inputTensor, err := onnxrt.NewTensor(onnxrt.NewShape(tensor.Shape...), tensor.Data)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer func() { _ = inputTensor.Destroy() }()
	outputs := []onnxrt.Value{nil}
	if err := session.Run([]onnxrt.Value{inputTensor}, outputs); err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			if o != nil {
				_ = o.Destroy()
			}
		}
	}()
	// Return per-image buffers now that batch tensor has been created and used
	for _, b := range bufs {
		mempool.PutFloat32(b)
	}

	outTensor := outputs[0]
	floatTensor, ok := outTensor.(*onnxrt.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("expected float32 tensor, got %T", outTensor)
	}
	data := floatTensor.GetData()
	shape := outTensor.GetShape()

	classesGuess := r.charset.Size() + 1
	classesFirst := determineClassesFirst(shape, classesGuess)
	blankIndex := 0
	decoded := DecodeCTCGreedy(data, shape, blankIndex, classesFirst)

	out := make([]Result, len(regions))
	for i := range out {
		if i >= len(decoded) {
			break
		}
		seq := decoded[i]
		runes := make([]rune, 0, len(seq.Collapsed))
		for _, idx := range seq.Collapsed {
			ch := r.charset.LookupToken(idx - 1)
			if ch == "" {
				continue
			}
			runes = append(runes, []rune(ch)...)
		}
		out[i] = Result{
			Text:            string(runes),
			Confidence:      SequenceConfidence(seq.CollapsedProb),
			CharConfidences: seq.CollapsedProb,
			Indices:         seq.Collapsed,
			Rotated:         prepped[i].rotated,
			Width:           prepped[i].w,
			Height:          prepped[i].h,
		}
	}
	return out, nil
}

// Helper to convert image.Image to *image.RGBA without external deps.
func toRGBA(img image.Image) *image.RGBA {
	b := img.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.Set(x-b.Min.X, y-b.Min.Y, img.At(x, y))
		}
	}
	return dst
}

// determineClassesFirst infers whether the classes dimension comes before the time dimension.
func determineClassesFirst(shape []int64, classesGuess int) bool {
	if len(shape) < 3 {
		return false
	}
	dims := make([]int64, len(shape))
	copy(dims, shape)
	for len(dims) > 3 && dims[len(dims)-1] == 1 {
		dims = dims[:len(dims)-1]
	}
	if len(dims) < 3 {
		return false
	}
	tDim := dims[1]
	cDim := dims[2]
	if int(cDim) == classesGuess {
		return false
	}
	if int(tDim) == classesGuess {
		return true
	}
	return false
}
