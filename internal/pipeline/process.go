package pipeline

import (
	"context"
	"errors"
	"fmt"
	"image"
	"time"

	"github.com/MeKo-Tech/pogo/internal/pdf"
	"github.com/MeKo-Tech/pogo/internal/recognizer"
	"github.com/MeKo-Tech/pogo/internal/utils"
)

// OCRRegionResult combines detection geometry with recognition output.
type OCRRegionResult struct {
	// Geometry and detection
	Polygon       []struct{ X, Y float64 } `json:"polygon"`
	Box           struct{ X, Y, W, H int } `json:"box"`
	DetConfidence float64                  `json:"det_confidence"`

	// Recognition
	Text            string    `json:"text"`
	RecConfidence   float64   `json:"rec_confidence"`
	CharConfidences []float64 `json:"char_confidences,omitempty"`
	Rotated         bool      `json:"rotated"`
	Language        string    `json:"language,omitempty"`

	// Timing
	Timing struct {
		RecognizePreprocessNs int64 `json:"recognize_preprocess_ns"`
		RecognizeModelNs      int64 `json:"recognize_model_ns"`
		RecognizeDecodeNs     int64 `json:"recognize_decode_ns"`
		RecognizeTotalNs      int64 `json:"recognize_total_ns"`
	} `json:"timing"`
}

// OCRImageResult is the per-image aggregated OCR output.
type OCRImageResult struct {
	Width       int               `json:"width"`
	Height      int               `json:"height"`
	Regions     []OCRRegionResult `json:"regions"`
	AvgDetConf  float64           `json:"avg_det_confidence"`
	Orientation struct {
		Angle      int     `json:"angle"`
		Confidence float64 `json:"confidence"`
		Applied    bool    `json:"applied"`
	} `json:"orientation"`
	Processing struct {
		DetectionNs   int64 `json:"detection_ns"`
		RecognitionNs int64 `json:"recognition_ns"`
		TotalNs       int64 `json:"total_ns"`
	} `json:"processing"`
}

// OCRPDFResult represents the OCR result for a PDF document.
type OCRPDFResult struct {
	Filename   string             `json:"filename"`
	TotalPages int                `json:"total_pages"`
	Pages      []OCRPDFPageResult `json:"pages"`
	Processing struct {
		ExtractionNs int64 `json:"extraction_ns"`
		TotalNs      int64 `json:"total_ns"`
	} `json:"processing"`
}

// OCRPDFPageResult represents OCR results for a single PDF page.
type OCRPDFPageResult struct {
	PageNumber int                 `json:"page_number"`
	Width      int                 `json:"width"`
	Height     int                 `json:"height"`
	Images     []OCRPDFImageResult `json:"images"`
	Processing struct {
		TotalNs int64 `json:"total_ns"`
	} `json:"processing"`
}

// OCRPDFImageResult represents OCR results for a single image extracted from a PDF page.
type OCRPDFImageResult struct {
	ImageIndex int               `json:"image_index"`
	Width      int               `json:"width"`
	Height     int               `json:"height"`
	Regions    []OCRRegionResult `json:"regions"`
	Confidence float64           `json:"confidence"`
}

// ProcessImage runs detection then recognition on a single image.
func (p *Pipeline) ProcessImage(img image.Image) (*OCRImageResult, error) {
	return p.ProcessImageContext(context.Background(), img)
}

// ProcessImageContext is like ProcessImage but allows cancellation via context.
func (p *Pipeline) ProcessImageContext(ctx context.Context, img image.Image) (*OCRImageResult, error) {
	if p == nil || p.Detector == nil || p.Recognizer == nil {
		return nil, errors.New("pipeline not initialized")
	}
	if img == nil {
		return nil, errors.New("input image is nil")
	}

	totalStart := time.Now()

	// Optional document orientation detection + rotation
	working := img
	var appliedAngle int
	var appliedConf float64
	if p.Orienter != nil && (p.cfg.Orientation.Enabled || p.cfg.EnableOrientation) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if res, err := p.Orienter.Predict(img); err == nil {
			appliedConf = res.Confidence
			switch res.Angle {
			case 90:
				working = utils.Rotate90(working)
				appliedAngle = 90
			case 180:
				working = utils.Rotate180(working)
				appliedAngle = 180
			case 270:
				working = utils.Rotate270(working)
				appliedAngle = 270
			default:
				appliedAngle = 0
			}
		}
	}

	// Optional rectification (minimal CPU-only). Currently returns original image
	// while exercising model path, so it is safe to enable.
	if p.Rectifier != nil && p.cfg.Rectification.Enabled {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if rxImg, err := p.Rectifier.Apply(working); err == nil && rxImg != nil {
			working = rxImg
		}
	}
	// Detection
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	detStart := time.Now()
	regions, err := p.Detector.DetectRegions(working)
	if err != nil {
		return nil, fmt.Errorf("detection failed: %w", err)
	}
	detNs := time.Since(detStart).Nanoseconds()

	// Recognition (batch)
	recStart := time.Now()
	var recResults []recognizer.Result
	if len(regions) > 0 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		recResults, err = p.Recognizer.RecognizeBatch(working, regions)
		if err != nil {
			return nil, fmt.Errorf("recognition failed: %w", err)
		}
	}
	recNs := time.Since(recStart).Nanoseconds()

	// Aggregate results
	out := &OCRImageResult{}
	// Report original image dimensions; if rotated, transform regions back
	ob := img.Bounds()
	out.Width, out.Height = ob.Dx(), ob.Dy()
	if p.Orienter != nil && (p.cfg.Orientation.Enabled || p.cfg.EnableOrientation) {
		out.Orientation.Angle = appliedAngle
		out.Orientation.Confidence = appliedConf
		out.Orientation.Applied = appliedAngle != 0
	}
	out.Regions = make([]OCRRegionResult, 0, len(regions))
	var detSum float64
	cleanOpts := recognizer.DefaultCleanOptions()
	if p.cfg.Recognizer.Language != "" {
		cleanOpts.Language = p.cfg.Recognizer.Language
	}
	for i, r := range regions {
		var reg OCRRegionResult
		// geometry (transform back to original orientation if applied)
		toOriginal := func(x, y float64) (float64, float64) {
			switch appliedAngle {
			case 90:
				return float64(ob.Dx()-1) - y, x
			case 180:
				return float64(ob.Dx()-1) - x, float64(ob.Dy()-1) - y
			case 270:
				return y, float64(ob.Dy()-1) - x
			default:
				return x, y
			}
		}
		// Transform AABB by mapping its corners
		bx := float64(r.Box.MinX)
		by := float64(r.Box.MinY)
		bw := float64(r.Box.Width())
		bh := float64(r.Box.Height())
		x1, y1 := toOriginal(bx, by)
		x2, y2 := toOriginal(bx+bw, by)
		x3, y3 := toOriginal(bx+bw, by+bh)
		x4, y4 := toOriginal(bx, by+bh)
		minX := minf4(x1, x2, x3, x4)
		maxX := maxf4(x1, x2, x3, x4)
		minY := minf4(y1, y2, y3, y4)
		maxY := maxf4(y1, y2, y3, y4)
		reg.Box = struct{ X, Y, W, H int }{X: int(minX + 0.5), Y: int(minY + 0.5), W: int(maxX - minX + 0.5), H: int(maxY - minY + 0.5)}
		reg.Polygon = make([]struct{ X, Y float64 }, len(r.Polygon))
		for j, pt := range r.Polygon {
			ox, oy := toOriginal(pt.X, pt.Y)
			reg.Polygon[j] = struct{ X, Y float64 }{ox, oy}
		}
		reg.DetConfidence = r.Confidence
		detSum += r.Confidence
		// recognition mapping
		if i < len(recResults) {
			rr := recResults[i]
			text := recognizer.PostProcessText(rr.Text, cleanOpts)
			reg.Text = text
			reg.RecConfidence = rr.Confidence
			reg.CharConfidences = rr.CharConfidences
			reg.Rotated = rr.Rotated
			// language detection (heuristic, post-clean)
			reg.Language = recognizer.DetectLanguage(text)
			reg.Timing.RecognizePreprocessNs = rr.TimingNs.Preprocess
			reg.Timing.RecognizeModelNs = rr.TimingNs.Model
			reg.Timing.RecognizeDecodeNs = rr.TimingNs.Decode
			reg.Timing.RecognizeTotalNs = rr.TimingNs.Total
		}
		out.Regions = append(out.Regions, reg)
	}
	if len(regions) > 0 {
		out.AvgDetConf = detSum / float64(len(regions))
	}
	out.Processing.DetectionNs = detNs
	out.Processing.RecognitionNs = recNs
	out.Processing.TotalNs = time.Since(totalStart).Nanoseconds()
	return out, nil
}

// ProcessImages processes multiple images sequentially and returns results.
func (p *Pipeline) ProcessImages(images []image.Image) ([]*OCRImageResult, error) {
	return p.ProcessImagesContext(context.Background(), images)
}

// ProcessImagesContext processes images with context cancellation support.
func (p *Pipeline) ProcessImagesContext(ctx context.Context, images []image.Image) ([]*OCRImageResult, error) {
	if len(images) == 0 {
		return nil, errors.New("no images provided")
	}
	results := make([]*OCRImageResult, len(images))
	for i, img := range images {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		res, err := p.ProcessImageContext(ctx, img)
		if err != nil {
			return nil, fmt.Errorf("image %d: %w", i, err)
		}
		results[i] = res
	}
	return results, nil
}

// ProcessPDF processes a PDF file and returns OCR results for all pages.
func (p *Pipeline) ProcessPDF(filename string, pageRange string) (*OCRPDFResult, error) {
	return p.ProcessPDFContext(context.Background(), filename, pageRange)
}

// ProcessPDFContext processes a PDF file with context cancellation support.
func (p *Pipeline) ProcessPDFContext(ctx context.Context, filename string, pageRange string) (*OCRPDFResult, error) {
	if p == nil || p.Detector == nil || p.Recognizer == nil {
		return nil, errors.New("pipeline not initialized")
	}

	totalStart := time.Now()

	// Extract images from PDF
	extractStart := time.Now()
	pageImages, err := pdf.ExtractImages(filename, pageRange)
	if err != nil {
		return nil, fmt.Errorf("failed to extract images from PDF: %w", err)
	}
	extractNs := time.Since(extractStart).Nanoseconds()

	// Process each page
	var pages []OCRPDFPageResult
	for pageNum, images := range pageImages {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		pageResult, err := p.processPDFPage(ctx, pageNum, images)
		if err != nil {
			return nil, fmt.Errorf("failed to process page %d: %w", pageNum, err)
		}

		pages = append(pages, *pageResult)
	}

	totalNs := time.Since(totalStart).Nanoseconds()

	result := &OCRPDFResult{
		Filename:   filename,
		TotalPages: len(pages),
		Pages:      pages,
		Processing: struct {
			ExtractionNs int64 `json:"extraction_ns"`
			TotalNs      int64 `json:"total_ns"`
		}{
			ExtractionNs: extractNs,
			TotalNs:      totalNs,
		},
	}

	return result, nil
}

// processPDFPage processes all images from a single PDF page.
func (p *Pipeline) processPDFPage(ctx context.Context, pageNum int, images []image.Image) (*OCRPDFPageResult, error) {
	pageStart := time.Now()

	var imageResults []OCRPDFImageResult
	var pageWidth, pageHeight int

	for i, img := range images {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		// Update page dimensions (use largest image dimensions)
		bounds := img.Bounds()
		imgWidth := bounds.Dx()
		imgHeight := bounds.Dy()
		if imgWidth > pageWidth {
			pageWidth = imgWidth
		}
		if imgHeight > pageHeight {
			pageHeight = imgHeight
		}

		// Process image with full OCR pipeline
		ocrResult, err := p.ProcessImageContext(ctx, img)
		if err != nil {
			return nil, fmt.Errorf("OCR processing failed for image %d: %w", i, err)
		}

		imageResult := OCRPDFImageResult{
			ImageIndex: i,
			Width:      imgWidth,
			Height:     imgHeight,
			Regions:    ocrResult.Regions,
			Confidence: ocrResult.AvgDetConf,
		}

		imageResults = append(imageResults, imageResult)
	}

	pageNs := time.Since(pageStart).Nanoseconds()

	pageResult := &OCRPDFPageResult{
		PageNumber: pageNum,
		Width:      pageWidth,
		Height:     pageHeight,
		Images:     imageResults,
		Processing: struct {
			TotalNs int64 `json:"total_ns"`
		}{
			TotalNs: pageNs,
		},
	}

	return pageResult, nil
}

func minf4(a, b, c, d float64) float64 {
	m := a
	if b < m {
		m = b
	}
	if c < m {
		m = c
	}
	if d < m {
		m = d
	}
	return m
}

func maxf4(a, b, c, d float64) float64 {
	m := a
	if b > m {
		m = b
	}
	if c > m {
		m = c
	}
	if d > m {
		m = d
	}
	return m
}
