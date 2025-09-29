package pipeline

import (
	"context"
	"errors"
	"fmt"
	"image"
	"log/slog"
	"time"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/orientation"
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

// applyOrientationDetection applies orientation detection and rotation if enabled.
func (p *Pipeline) applyOrientationDetection(ctx context.Context, img image.Image) (image.Image, int, float64, error) {
	working := img
	var appliedAngle int
	var appliedConf float64

	if p.Orienter != nil && (p.cfg.Orientation.Enabled || p.cfg.EnableOrientation) {
		if err := ctx.Err(); err != nil {
			return nil, 0, 0, err
		}
		slog.Debug("Running orientation detection")
		if res, err := p.Orienter.Predict(img); err == nil {
			appliedConf = res.Confidence
			switch res.Angle {
			case 90:
				working = utils.Rotate90(working)
				appliedAngle = 90
				slog.Debug("Applied 90° rotation", "confidence", appliedConf)
			case 180:
				working = utils.Rotate180(working)
				appliedAngle = 180
				slog.Debug("Applied 180° rotation", "confidence", appliedConf)
			case 270:
				working = utils.Rotate270(working)
				appliedAngle = 270
				slog.Debug("Applied 270° rotation", "confidence", appliedConf)
			default:
				appliedAngle = 0
				slog.Debug("No rotation applied", "confidence", appliedConf)
			}
		} else {
			slog.Debug("Orientation detection failed", "error", err)
		}
	}

	return working, appliedAngle, appliedConf, nil
}

// applyRectification applies document rectification if enabled.
func (p *Pipeline) applyRectification(ctx context.Context, img image.Image) (image.Image, error) {
	if p.Rectifier != nil && p.cfg.Rectification.Enabled {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		slog.Debug("Running document rectification")
		if rxImg, err := p.Rectifier.Apply(img); err == nil && rxImg != nil {
			slog.Debug("Document rectification applied")
			return rxImg, nil
		} else if err != nil {
			slog.Debug("Document rectification failed", "error", err)
		}
	}
	return img, nil
}

// performDetection runs text detection on the image.
func (p *Pipeline) performDetection(ctx context.Context, img image.Image) ([]detector.DetectedRegion, int64, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	slog.Debug("Starting text detection")
	detStart := time.Now()
	regions, err := p.Detector.DetectRegions(img)
	if err != nil {
		return nil, 0, fmt.Errorf("detection failed: %w", err)
	}
	detNs := time.Since(detStart).Nanoseconds()
	slog.Debug("Text detection completed", "regions_found", len(regions), "duration_ms", detNs/1000000)
	return regions, detNs, nil
}

// performRecognition runs text recognition on detected regions.
func (p *Pipeline) performRecognition(ctx context.Context, img image.Image,
	regions []detector.DetectedRegion,
) ([]recognizer.Result, int64, error) {
	recStart := time.Now()
	var recResults []recognizer.Result
	if len(regions) > 0 {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		slog.Debug("Starting text recognition", "regions_count", len(regions))
		var err error
		recResults, err = p.Recognizer.RecognizeBatch(img, regions)
		if err != nil {
			return nil, 0, fmt.Errorf("recognition failed: %w", err)
		}
		slog.Debug("Text recognition completed", "duration_ms", time.Since(recStart).Nanoseconds()/1000000)
	} else {
		slog.Debug("No text regions detected, skipping recognition")
	}
	recNs := time.Since(recStart).Nanoseconds()
	return recResults, recNs, nil
}

// buildImageResult builds the final OCRImageResult from detection and recognition results.
func (p *Pipeline) buildImageResult(
	img image.Image,
	regions []detector.DetectedRegion,
	recResults []recognizer.Result,
	appliedAngle int,
	appliedConf float64,
	detNs, recNs, totalNs int64,
) *OCRImageResult {
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
		reg := p.buildRegionResult(r, recResults, i, appliedAngle, ob, cleanOpts)
		detSum += r.Confidence
		out.Regions = append(out.Regions, reg)
	}

	if len(regions) > 0 {
		out.AvgDetConf = detSum / float64(len(regions))
	}
	out.Processing.DetectionNs = detNs
	out.Processing.RecognitionNs = recNs
	out.Processing.TotalNs = totalNs

	return out
}

// buildRegionResult creates a single OCRRegionResult from detection and recognition data.
func (p *Pipeline) buildRegionResult(
	r detector.DetectedRegion,
	recResults []recognizer.Result,
	index int,
	appliedAngle int,
	originalBounds image.Rectangle,
	cleanOpts recognizer.CleanOptions,
) OCRRegionResult {
	var reg OCRRegionResult

	// Transform coordinates back to original orientation if applied
	toOriginal := func(x, y float64) (float64, float64) {
		switch appliedAngle {
		case 90:
			return float64(originalBounds.Dx()-1) - y, x
		case 180:
			return float64(originalBounds.Dx()-1) - x, float64(originalBounds.Dy()-1) - y
		case 270:
			return y, float64(originalBounds.Dy()-1) - x
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
	reg.Box = struct{ X, Y, W, H int }{
		X: int(minX + 0.5),
		Y: int(minY + 0.5),
		W: int(maxX - minX + 0.5),
		H: int(maxY - minY + 0.5),
	}

	reg.Polygon = make([]struct{ X, Y float64 }, len(r.Polygon))
	for j, pt := range r.Polygon {
		ox, oy := toOriginal(pt.X, pt.Y)
		reg.Polygon[j] = struct{ X, Y float64 }{ox, oy}
	}

	reg.DetConfidence = r.Confidence

	// Add recognition results if available
	if index < len(recResults) {
		rr := recResults[index]
		text := recognizer.PostProcessText(rr.Text, cleanOpts)
		reg.Text = text
		reg.RecConfidence = rr.Confidence
		reg.CharConfidences = rr.CharConfidences
		reg.Rotated = rr.Rotated
		reg.Language = recognizer.DetectLanguage(text)
		reg.Timing.RecognizePreprocessNs = rr.TimingNs.Preprocess
		reg.Timing.RecognizeModelNs = rr.TimingNs.Model
		reg.Timing.RecognizeDecodeNs = rr.TimingNs.Decode
		reg.Timing.RecognizeTotalNs = rr.TimingNs.Total
	}

	return reg
}

// ProcessImageContext is like ProcessImage but allows cancellation via context.
func (p *Pipeline) ProcessImageContext(ctx context.Context, img image.Image) (*OCRImageResult, error) {
	if p == nil || p.Detector == nil || p.Recognizer == nil {
		return nil, errors.New("pipeline not initialized")
	}
	if img == nil {
		return nil, errors.New("input image is nil")
	}

	bounds := img.Bounds()
	slog.Debug("Starting image processing", "width", bounds.Dx(), "height", bounds.Dy())

	totalStart := time.Now()

	// Apply orientation detection and rotation
	working, appliedAngle, appliedConf, err := p.applyOrientationDetection(ctx, img)
	if err != nil {
		return nil, err
	}

	// Apply rectification
	working, err = p.applyRectification(ctx, working)
	if err != nil {
		return nil, err
	}

	// Perform detection
	regions, detNs, err := p.performDetection(ctx, working)
	if err != nil {
		return nil, err
	}

	// Perform recognition
	recResults, recNs, err := p.performRecognition(ctx, working, regions)
	if err != nil {
		return nil, err
	}

	// Build final result
	totalNs := time.Since(totalStart).Nanoseconds()
	result := p.buildImageResult(img, regions, recResults, appliedAngle, appliedConf, detNs, recNs, totalNs)

	slog.Debug("Image processing completed",
		"total_duration_ms", result.Processing.TotalNs/1000000,
		"detection_duration_ms", detNs/1000000,
		"recognition_duration_ms", recNs/1000000,
		"regions_processed", len(result.Regions))

	return result, nil
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

	// Pre-process orientation detection for all images if enabled
	var orientationResults []orientation.Result
	var workingImages []image.Image
	if p.Orienter != nil && (p.cfg.Orientation.Enabled || p.cfg.EnableOrientation) {
		// Use batch orientation detection for better performance
		results, err := p.Orienter.BatchPredict(images)
		if err != nil {
			slog.Debug("Batch orientation detection failed, falling back to individual processing", "error", err)
			// Fall back to individual processing
			return p.processImagesIndividual(ctx, images)
		}
		orientationResults = results

		// Apply rotations to create working images
		workingImages = make([]image.Image, len(images))
		for i, img := range images {
			result := results[i]
			switch result.Angle {
			case 90:
				workingImages[i] = utils.Rotate90(img)
			case 180:
				workingImages[i] = utils.Rotate180(img)
			case 270:
				workingImages[i] = utils.Rotate270(img)
			default:
				workingImages[i] = img
			}
		}
	} else {
		// No orientation processing needed
		workingImages = images
		orientationResults = make([]orientation.Result, len(images)) // All zero values
	}

	// Process each image with its pre-rotated working image
	results := make([]*OCRImageResult, len(images))
	for i, workingImg := range workingImages {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		// Apply rectification to working image
		rectifiedImg, err := p.applyRectification(ctx, workingImg)
		if err != nil {
			return nil, err
		}

		// Perform detection
		regions, detNs, err := p.performDetection(ctx, rectifiedImg)
		if err != nil {
			return nil, fmt.Errorf("image %d detection: %w", i, err)
		}

		// Perform recognition
		recResults, recNs, err := p.performRecognition(ctx, rectifiedImg, regions)
		if err != nil {
			return nil, fmt.Errorf("image %d recognition: %w", i, err)
		}

		// Build result with orientation info
		totalNs := detNs + recNs // Simplified total for batch processing
		appliedAngle := orientationResults[i].Angle
		appliedConf := orientationResults[i].Confidence
		results[i] = p.buildImageResult(images[i], regions, recResults, appliedAngle, appliedConf, detNs, recNs, totalNs)
	}

	return results, nil
}

// processImagesIndividual processes images one by one (fallback method).
func (p *Pipeline) processImagesIndividual(ctx context.Context, images []image.Image) ([]*OCRImageResult, error) {
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
	pages := make([]OCRPDFPageResult, 0, len(pageImages))
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

	imageResults := make([]OCRPDFImageResult, 0, len(images))
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
