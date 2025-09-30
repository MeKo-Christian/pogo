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

	// Preprocess the region
	preprocessed, preprocessNs, err := r.preprocessRegion(img, region)
	if err != nil {
		return nil, err
	}

	// Run inference
	modelOutput, modelNs, err := r.runInference(preprocessed.tensor)
	if err != nil {
		return nil, err
	}
	defer func() {
		for _, o := range modelOutput.outputs {
			if o != nil {
				_ = o.Destroy()
			}
		}
	}()
	mempool.PutFloat32(preprocessed.buf)

	// Decode the output
	result, decodeNs, err := r.decodeOutput(modelOutput, preprocessed)
	if err != nil {
		return nil, err
	}

	totalNs := time.Since(totalStart).Nanoseconds()
	result.TimingNs = struct{ Preprocess, Model, Decode, Total int64 }{
		Preprocess: preprocessNs,
		Model:      modelNs,
		Decode:     decodeNs,
		Total:      totalNs,
	}

	return result, nil
}

type preprocessedRegion struct {
	tensor  onnx.Tensor
	buf     []float32
	rotated bool
	width   int
	height  int
}

type preprocessedBatchRegion struct {
	img     image.Image
	rotated bool
	w, h    int
}

func (r *Recognizer) preprocessRegion(
	img image.Image,
	region detector.DetectedRegion,
) (*preprocessedRegion, int64, error) {
	t0 := time.Now()

	// Crop and optionally rotate
	var patch image.Image
	var rotated bool
	var err error
	if r.textLineOrienter != nil {
		patch, rotated, err = CropRegionImageWithOrienter(img, region, r.textLineOrienter, true)
	} else {
		patch, rotated, err = CropRegionImage(img, region, true)
	}
	if err != nil {
		return nil, 0, fmt.Errorf("crop region: %w", err)
	}

	// Resize with fixed height and padding
	targetH := r.config.ImageHeight
	if targetH <= 0 {
		targetH = 32
	}
	resized, outW, outH, err := ResizeForRecognition(patch, targetH, r.config.MaxWidth, r.config.PadWidthMultiple)
	if err != nil {
		return nil, 0, fmt.Errorf("resize: %w", err)
	}

	// Normalize to tensor
	tensor, buf, err := NormalizeForRecognitionWithPool(resized)
	if err != nil {
		return nil, 0, fmt.Errorf("normalize: %w", err)
	}

	return &preprocessedRegion{
		tensor:  tensor,
		buf:     buf,
		rotated: rotated,
		width:   outW,
		height:  outH,
	}, time.Since(t0).Nanoseconds(), nil
}

type modelOutput struct {
	outputs []onnxrt.Value
	data    []float32
	shape   []int64
}

func (r *Recognizer) runInference(tensor onnx.Tensor) (*modelOutput, int64, error) {
	r.mu.RLock()
	session := r.session
	r.mu.RUnlock()
	if session == nil {
		return nil, 0, errors.New("recognizer session is nil")
	}

	m0 := time.Now()
	inputTensor, err := onnxrt.NewTensor(onnxrt.NewShape(tensor.Shape...), tensor.Data)
	if err != nil {
		return nil, 0, fmt.Errorf("create input tensor: %w", err)
	}
	defer func() { _ = inputTensor.Destroy() }()

	outputs := []onnxrt.Value{nil}
	if err := session.Run([]onnxrt.Value{inputTensor}, outputs); err != nil {
		return nil, 0, fmt.Errorf("inference failed: %w", err)
	}

	// Extract tensor data
	outTensor := outputs[0]
	floatTensor, ok := outTensor.(*onnxrt.Tensor[float32])
	if !ok {
		for _, o := range outputs {
			if o != nil {
				_ = o.Destroy()
			}
		}
		return nil, 0, fmt.Errorf("expected float32 tensor, got %T", outTensor)
	}

	return &modelOutput{
		outputs: outputs,
		data:    floatTensor.GetData(),
		shape:   outTensor.GetShape(),
	}, time.Since(m0).Nanoseconds(), nil
}

func (r *Recognizer) decodeOutput(output *modelOutput, preprocessed *preprocessedRegion) (*Result, int64, error) {
	d0 := time.Now()

	r.mu.RLock()
	decodingMethod := r.config.DecodingMethod
	beamWidth := r.config.BeamWidth
	r.mu.RUnlock()

	classesGuess := r.charset.Size() + 1
	classesFirst := determineClassesFirst(output.shape, classesGuess)
	blankIndex := 0 // PaddleOCR CTC typically uses blank=0

	var seq interface{}
	if decodingMethod == "beam_search" && beamWidth > 1 {
		// Use beam search decoding
		decoded := DecodeCTCBeamSearch(output.data, output.shape, blankIndex, beamWidth, classesFirst)
		if len(decoded) == 0 {
			return nil, 0, errors.New("empty beam search decoded output")
		}
		seq = decoded[0]
	} else {
		// Use greedy decoding (default)
		decoded := DecodeCTCGreedy(output.data, output.shape, blankIndex, classesFirst)
		if len(decoded) == 0 {
			return nil, 0, errors.New("empty decoded output")
		}
		seq = decoded[0]
	}

	// Map to text based on sequence type
	var collapsed []int
	var charProbs []float64
	var confidence float64

	switch s := seq.(type) {
	case DecodedSequence:
		collapsed = s.Collapsed
		charProbs = s.CollapsedProb
		confidence = SequenceConfidence(s.CollapsedProb)
	case BeamSearchResult:
		collapsed = s.Sequence
		charProbs = s.CharProbs
		confidence = SequenceConfidence(s.CharProbs)
	default:
		return nil, 0, errors.New("unknown sequence type")
	}

	runes := make([]rune, 0, len(collapsed))
	for _, idx := range collapsed {
		// map index to token; idx in [0..classes-1], 0 is blank
		ch := r.charset.LookupToken(idx - 1) // shift by -1 to skip blank
		if ch == "" {
			continue
		}
		runes = append(runes, []rune(ch)...)
	}
	text := string(runes)

	return &Result{
		Text:            text,
		Confidence:      confidence,
		CharConfidences: charProbs,
		Indices:         collapsed,
		Rotated:         preprocessed.rotated,
		Width:           preprocessed.width,
		Height:          preprocessed.height,
	}, time.Since(d0).Nanoseconds(), nil
}

// preprocessBatchRegions handles cropping and resizing of regions for batch processing.
func (r *Recognizer) preprocessBatchRegions(img image.Image, regions []detector.DetectedRegion) (
	[]preprocessedBatchRegion, int, error,
) {
	targetH := r.config.ImageHeight
	if targetH <= 0 {
		targetH = 32
	}

	prepped := make([]preprocessedBatchRegion, len(regions))
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
			return nil, 0, fmt.Errorf("crop region %d: %w", i, err)
		}
		resized, outW, outH, err := ResizeForRecognition(patch, targetH, r.config.MaxWidth, r.config.PadWidthMultiple)
		if err != nil {
			return nil, 0, fmt.Errorf("resize region %d: %w", i, err)
		}
		prepped[i] = preprocessedBatchRegion{img: resized, rotated: rotated, w: outW, h: outH}
		if outW > maxW {
			maxW = outW
		}
	}
	return prepped, maxW, nil
}

// padBatchRegions pads all regions to the maximum width for batch processing.
func (r *Recognizer) padBatchRegions(prepped []preprocessedBatchRegion, maxW int) []preprocessedBatchRegion {
	for i, p := range prepped {
		if p.w < maxW {
			padded := image.NewRGBA(image.Rect(0, 0, maxW, p.h))
			src := toRGBA(p.img)
			sb := src.Bounds()
			for y := range sb.Dy() {
				for x := range sb.Dx() {
					padded.Set(x, y, src.At(sb.Min.X+x, sb.Min.Y+y))
				}
			}
			prepped[i].img = padded
			prepped[i].w = maxW
		}
	}
	return prepped
}

// normalizeBatchRegions converts padded images to normalized tensors.
func (r *Recognizer) normalizeBatchRegions(prepped []preprocessedBatchRegion) ([][]float32, [][]float32, error) {
	batchTensors := make([][]float32, len(prepped))
	bufs := make([][]float32, len(prepped))
	for i, p := range prepped {
		ten, buf, err := NormalizeForRecognitionWithPool(p.img)
		if err != nil {
			return nil, nil, fmt.Errorf("normalize region %d: %w", i, err)
		}
		batchTensors[i] = ten.Data
		bufs[i] = buf
	}
	return batchTensors, bufs, nil
}

// buildBatchTensor creates a single batch tensor from individual region tensors.
func (r *Recognizer) buildBatchTensor(batchTensors [][]float32, prepped []preprocessedBatchRegion) (
	onnx.Tensor, error,
) {
	if len(batchTensors) == 0 {
		return onnx.Tensor{}, errors.New("no tensors prepared")
	}
	tensor, err := onnx.NewBatchImageTensor(batchTensors, 3, prepped[0].h, prepped[0].w)
	if err != nil {
		return onnx.Tensor{}, fmt.Errorf("build batch tensor: %w", err)
	}
	return tensor, nil
}

// runBatchInference executes the model on the batch tensor.
func (r *Recognizer) runBatchInference(tensor onnx.Tensor) (*modelOutput, error) {
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

	outTensor := outputs[0]
	floatTensor, ok := outTensor.(*onnxrt.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("expected float32 tensor, got %T", outTensor)
	}
	return &modelOutput{
		outputs: outputs,
		data:    floatTensor.GetData(),
		shape:   outTensor.GetShape(),
	}, nil
}

// extractSequenceData extracts collapsed indices, character probabilities, and confidence from a sequence.
func extractSequenceData(seq interface{}) ([]int, []float64, float64) {
	switch s := seq.(type) {
	case DecodedSequence:
		return s.Collapsed, s.CollapsedProb, SequenceConfidence(s.CollapsedProb)
	case BeamSearchResult:
		return s.Sequence, s.CharProbs, SequenceConfidence(s.CharProbs)
	default:
		return nil, nil, 0.0
	}
}

// convertIndicesToRunes converts token indices to runes using the charset.
func convertIndicesToRunes(indices []int, charset *Charset) []rune {
	runes := make([]rune, 0, len(indices))
	for _, idx := range indices {
		ch := charset.LookupToken(idx - 1)
		if ch == "" {
			continue
		}
		runes = append(runes, []rune(ch)...)
	}
	return runes
}

// buildBatchResults constructs Result structs from decoded sequences.
func (r *Recognizer) buildBatchResults(decoded interface{}, prepped []preprocessedBatchRegion) []Result {
	out := make([]Result, len(prepped))

	r.mu.RLock()
	charset := r.charset
	r.mu.RUnlock()

	for i := range out {
		// Always set width/height from prepped region
		out[i].Width = prepped[i].w
		out[i].Height = prepped[i].h
		out[i].Rotated = prepped[i].rotated

		var seq interface{}
		switch d := decoded.(type) {
		case []DecodedSequence:
			if i >= len(d) {
				continue
			}
			seq = d[i]
		case []BeamSearchResult:
			if i >= len(d) {
				continue
			}
			seq = d[i]
		default:
			continue
		}

		collapsed, charProbs, confidence := extractSequenceData(seq)
		if collapsed == nil {
			continue
		}

		runes := convertIndicesToRunes(collapsed, charset)
		out[i].Text = string(runes)
		out[i].Confidence = confidence
		out[i].CharConfidences = charProbs
		out[i].Indices = collapsed
	}
	return out
}

// RecognizeBatch processes multiple regions on the same source image.
func (r *Recognizer) RecognizeBatch(img image.Image, regions []detector.DetectedRegion) ([]Result, error) {
	if img == nil {
		return nil, errors.New("input image is nil")
	}
	if len(regions) == 0 {
		return nil, errors.New("no regions provided")
	}

	// Preprocess regions
	prepped, maxW, err := r.preprocessBatchRegions(img, regions)
	if err != nil {
		return nil, err
	}

	// Pad to max width
	prepped = r.padBatchRegions(prepped, maxW)

	// Normalize to tensors
	batchTensors, bufs, err := r.normalizeBatchRegions(prepped)
	if err != nil {
		return nil, err
	}

	// Build batch tensor
	tensor, err := r.buildBatchTensor(batchTensors, prepped)
	if err != nil {
		return nil, err
	}

	// Run inference
	output, err := r.runBatchInference(tensor)
	if err != nil {
		return nil, err
	}

	// Return per-image buffers now that batch tensor has been created and used
	for _, b := range bufs {
		mempool.PutFloat32(b)
	}

	// Decode output
	r.mu.RLock()
	decodingMethod := r.config.DecodingMethod
	beamWidth := r.config.BeamWidth
	r.mu.RUnlock()

	classesGuess := r.charset.Size() + 1
	classesFirst := determineClassesFirst(output.shape, classesGuess)
	blankIndex := 0

	var decoded interface{}
	if decodingMethod == "beam_search" && beamWidth > 1 {
		decoded = DecodeCTCBeamSearch(output.data, output.shape, blankIndex, beamWidth, classesFirst)
	} else {
		decoded = DecodeCTCGreedy(output.data, output.shape, blankIndex, classesFirst)
	}

	// Build results
	return r.buildBatchResults(decoded, prepped), nil
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
