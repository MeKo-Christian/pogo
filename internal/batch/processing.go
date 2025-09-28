package batch

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/MeKo-Tech/pogo/internal/utils"
)

// loadAndValidateImage loads an image and validates it meets constraints.
func loadAndValidateImage(path string) (image.Image, utils.ImageMetadata, error) {
	if !utils.IsSupportedImage(path) {
		return nil, utils.ImageMetadata{}, fmt.Errorf("unsupported image format: %s", path)
	}

	img, meta, err := utils.LoadImage(path)
	if err != nil {
		return nil, utils.ImageMetadata{}, fmt.Errorf("failed to load %s: %w", path, err)
	}

	cons := utils.DefaultImageConstraints()
	if err := utils.ValidateImageConstraints(img, cons); err != nil {
		slog.Warn("image does not meet constraints, skipping", "file", path, "error", err)
	}

	return img, meta, nil
}

// applyConfidenceFilters applies detection and recognition confidence filters to OCR results.
func applyConfidenceFilters(res *pipeline.OCRImageResult, confFlag, minRecConf float64) {
	// Apply detection confidence filter
	if confFlag > 0 {
		filtered := make([]pipeline.OCRRegionResult, 0, len(res.Regions))
		var sum float64
		for _, r := range res.Regions {
			if r.DetConfidence >= confFlag {
				filtered = append(filtered, r)
				sum += r.DetConfidence
			}
		}
		res.Regions = filtered
		if len(filtered) > 0 {
			res.AvgDetConf = sum / float64(len(filtered))
		} else {
			res.AvgDetConf = 0
		}
	}

	// Apply recognition confidence filter
	if minRecConf > 0 {
		filtered := make([]pipeline.OCRRegionResult, 0, len(res.Regions))
		for _, r := range res.Regions {
			if r.RecConfidence >= minRecConf {
				filtered = append(filtered, r)
			}
		}
		res.Regions = filtered
	}
}

// generateAndSaveOverlay creates an overlay image and saves it to disk.
func generateAndSaveOverlay(img image.Image, res *pipeline.OCRImageResult,
	meta utils.ImageMetadata, overlayDir string) {
	ov := pipeline.RenderOverlay(img, res, color.RGBA{255, 0, 0, 255}, color.RGBA{0, 255, 0, 255})
	if ov == nil {
		return
	}

	if err := os.MkdirAll(overlayDir, 0o750); err != nil {
		return
	}

	base := filepath.Base(meta.Path)
	outPath := filepath.Join(overlayDir, strings.TrimSuffix(base, filepath.Ext(base))+"_overlay.png")
	if f, err := os.Create(outPath); err == nil { //nolint:gosec
		// G304: outPath constructed from CLI overlay-dir flag, expected user input
		_ = png.Encode(f, ov)
		_ = f.Close()
	}
}

// processSingleImage processes a single image through the OCR pipeline.
func processSingleImage(pl *pipeline.Pipeline, path string, confFlag, minRecConf float64,
	overlayDir string) (*pipeline.OCRImageResult, error) {
	// Load and validate image
	img, meta, err := loadAndValidateImage(path)
	if err != nil {
		return nil, err
	}

	// Process with OCR pipeline
	res, err := pl.ProcessImage(img)
	if err != nil {
		return nil, fmt.Errorf("OCR failed for %s: %w", path, err)
	}

	// Apply confidence filters
	applyConfidenceFilters(res, confFlag, minRecConf)

	// Generate overlay if requested
	if overlayDir != "" {
		generateAndSaveOverlay(img, res, meta, overlayDir)
	}

	return res, nil
}

// processImagesParallel loads and processes images in parallel.
func processImagesParallel(pl *pipeline.Pipeline, imagePaths []string,
	confFlag, minRecConf float64, overlayDir string) ([]*pipeline.OCRImageResult, error) {
	imageResults := make([]*pipeline.OCRImageResult, len(imagePaths))

	for i, path := range imagePaths {
		res, err := processSingleImage(pl, path, confFlag, minRecConf, overlayDir)
		if err != nil {
			return nil, err
		}
		imageResults[i] = res
	}

	return imageResults, nil
}
