package pipeline

import (
	"context"
	"errors"
	"image"
	"log/slog"
	"time"
)

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
