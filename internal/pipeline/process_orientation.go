package pipeline

import (
	"context"
	"fmt"
	"image"
	"log/slog"

	"github.com/MeKo-Tech/pogo/internal/orientation"
	"github.com/MeKo-Tech/pogo/internal/utils"
)

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

// prepareOrientation handles orientation detection and image rotation preparation.
func (p *Pipeline) prepareOrientation(ctx context.Context,
	images []image.Image) ([]orientation.Result, []image.Image, error) {
	if p.Orienter == nil || (!p.cfg.Orientation.Enabled && !p.cfg.EnableOrientation) {
		// No orientation processing needed
		orientationResults := make([]orientation.Result, len(images)) // All zero values
		return orientationResults, images, nil
	}

	// Check for cancellation before expensive operation
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}

	// Use batch orientation detection for better performance
	results, err := p.Orienter.BatchPredict(images)
	if err != nil {
		slog.Debug("Batch orientation detection failed, falling back to individual processing", "error", err)
		// Fall back to individual processing
		return nil, nil, fmt.Errorf("orientation detection failed: %w", err)
	}

	// Apply rotations to create working images
	workingImages := make([]image.Image, len(images))
	for i, img := range images {
		workingImages[i] = p.applyOrientationRotation(img, results[i].Angle)
	}

	return results, workingImages, nil
}

// applyOrientationRotation applies the appropriate rotation based on detected angle.
func (p *Pipeline) applyOrientationRotation(img image.Image, angle int) image.Image {
	switch angle {
	case 90:
		return utils.Rotate90(img)
	case 180:
		return utils.Rotate180(img)
	case 270:
		return utils.Rotate270(img)
	default:
		return img
	}
}

// processImagesWithOrientation processes images that have been prepared with orientation.
func (p *Pipeline) processImagesWithOrientation(ctx context.Context, originalImages []image.Image,
	orientationResults []orientation.Result, workingImages []image.Image) ([]*OCRImageResult, error) {
	results := make([]*OCRImageResult, len(originalImages))
	for i, workingImg := range workingImages {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		result, err := p.processSingleImage(ctx, originalImages[i], workingImg, orientationResults[i])
		if err != nil {
			return nil, fmt.Errorf("image %d: %w", i, err)
		}

		results[i] = result
	}

	return results, nil
}

// processSingleImage processes a single image through the full OCR pipeline.
func (p *Pipeline) processSingleImage(ctx context.Context, originalImg, workingImg image.Image,
	orientationResult orientation.Result) (*OCRImageResult, error) {
	// Apply rectification to working image
	rectifiedImg, err := p.applyRectification(ctx, workingImg)
	if err != nil {
		return nil, err
	}

	// Perform detection
	regions, detNs, err := p.performDetection(ctx, rectifiedImg)
	if err != nil {
		return nil, err
	}

	// Perform recognition
	recResults, recNs, err := p.performRecognition(ctx, rectifiedImg, regions)
	if err != nil {
		return nil, err
	}

	// Build result with orientation info
	totalNs := detNs + recNs // Simplified total for batch processing
	appliedAngle := orientationResult.Angle
	appliedConf := orientationResult.Confidence
	return p.buildImageResult(originalImg, regions, recResults, appliedAngle, appliedConf, detNs, recNs, totalNs), nil
}
