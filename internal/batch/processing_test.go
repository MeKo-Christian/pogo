package batch

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadAndValidateImage_ValidImage(t *testing.T) {
	// Create a simple test image
	tempDir := testutil.CreateTempDir(t)
	imagePath := filepath.Join(tempDir, "test.png")

	// Create a minimal PNG image
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			img.Set(x, y, color.White)
		}
	}

	file, err := os.Create(imagePath)
	require.NoError(t, err)
	defer file.Close()

	err = png.Encode(file, img)
	require.NoError(t, err)

	loadedImg, meta, err := loadAndValidateImage(imagePath)
	require.NoError(t, err)
	require.NotNil(t, loadedImg)
	assert.Equal(t, imagePath, meta.Path)
	assert.Equal(t, 100, loadedImg.Bounds().Dx())
	assert.Equal(t, 100, loadedImg.Bounds().Dy())
}

func TestLoadAndValidateImage_UnsupportedFormat(t *testing.T) {
	tempDir := testutil.CreateTempDir(t)
	imagePath := filepath.Join(tempDir, "test.txt")

	err := os.WriteFile(imagePath, []byte("not an image"), 0o600)
	require.NoError(t, err)

	loadedImg, meta, err := loadAndValidateImage(imagePath)
	assert.Error(t, err)
	assert.Nil(t, loadedImg)
	assert.Equal(t, utils.ImageMetadata{}, meta)
	assert.Contains(t, err.Error(), "unsupported image format")
}

func TestLoadAndValidateImage_NonExistentFile(t *testing.T) {
	loadedImg, meta, err := loadAndValidateImage("/nonexistent/file.png")
	assert.Error(t, err)
	assert.Nil(t, loadedImg)
	assert.Equal(t, utils.ImageMetadata{}, meta)
}

func TestApplyConfidenceFilters_NoFilters(t *testing.T) {
	result := &pipeline.OCRImageResult{
		Regions: []pipeline.OCRRegionResult{
			{
				Text:          "Test",
				RecConfidence: 0.9,
				DetConfidence: 0.8,
				Box:           struct{ X, Y, W, H int }{X: 0, Y: 0, W: 10, H: 10},
			},
		},
		AvgDetConf: 0.8,
	}

	applyConfidenceFilters(result, 0.0, 0.0)

	assert.Len(t, result.Regions, 1)
	assert.Equal(t, 0.8, result.AvgDetConf)
}

func TestApplyConfidenceFilters_DetectionFilter(t *testing.T) {
	result := &pipeline.OCRImageResult{
		Regions: []pipeline.OCRRegionResult{
			{
				Text:          "High confidence",
				RecConfidence: 0.9,
				DetConfidence: 0.9,
				Box:           struct{ X, Y, W, H int }{X: 0, Y: 0, W: 10, H: 10},
			},
			{
				Text:          "Low confidence",
				RecConfidence: 0.8,
				DetConfidence: 0.3,
				Box:           struct{ X, Y, W, H int }{X: 10, Y: 10, W: 10, H: 10},
			},
		},
		AvgDetConf: 0.6,
	}

	applyConfidenceFilters(result, 0.5, 0.0) // Filter out det confidence < 0.5

	assert.Len(t, result.Regions, 1)
	assert.Equal(t, "High confidence", result.Regions[0].Text)
	assert.Equal(t, 0.9, result.AvgDetConf)
}

func TestApplyConfidenceFilters_RecognitionFilter(t *testing.T) {
	result := &pipeline.OCRImageResult{
		Regions: []pipeline.OCRRegionResult{
			{
				Text:          "High rec confidence",
				RecConfidence: 0.9,
				DetConfidence: 0.8,
				Box:           struct{ X, Y, W, H int }{X: 0, Y: 0, W: 10, H: 10},
			},
			{
				Text:          "Low rec confidence",
				RecConfidence: 0.3,
				DetConfidence: 0.8,
				Box:           struct{ X, Y, W, H int }{X: 10, Y: 10, W: 10, H: 10},
			},
		},
		AvgDetConf: 0.8,
	}

	applyConfidenceFilters(result, 0.0, 0.5) // Filter out rec confidence < 0.5

	assert.Len(t, result.Regions, 1)
	assert.Equal(t, "High rec confidence", result.Regions[0].Text)
}

func TestApplyConfidenceFilters_BothFilters(t *testing.T) {
	result := &pipeline.OCRImageResult{
		Regions: []pipeline.OCRRegionResult{
			{
				Text:          "Good both",
				RecConfidence: 0.9,
				DetConfidence: 0.9,
				Box:           struct{ X, Y, W, H int }{X: 0, Y: 0, W: 10, H: 10},
			},
			{
				Text:          "Bad det",
				RecConfidence: 0.9,
				DetConfidence: 0.3,
				Box:           struct{ X, Y, W, H int }{X: 10, Y: 10, W: 10, H: 10},
			},
			{
				Text:          "Bad rec",
				RecConfidence: 0.3,
				DetConfidence: 0.9,
				Box:           struct{ X, Y, W, H int }{X: 20, Y: 20, W: 10, H: 10},
			},
		},
		AvgDetConf: 0.7,
	}

	applyConfidenceFilters(result, 0.5, 0.5) // Filter both

	assert.Len(t, result.Regions, 1)
	assert.Equal(t, "Good both", result.Regions[0].Text)
	assert.Equal(t, 0.9, result.AvgDetConf)
}

func TestApplyConfidenceFilters_AllFilteredOut(t *testing.T) {
	result := &pipeline.OCRImageResult{
		Regions: []pipeline.OCRRegionResult{
			{
				Text:          "Low confidence",
				RecConfidence: 0.3,
				DetConfidence: 0.3,
				Box:           struct{ X, Y, W, H int }{X: 0, Y: 0, W: 10, H: 10},
			},
		},
		AvgDetConf: 0.3,
	}

	applyConfidenceFilters(result, 0.5, 0.5)

	assert.Len(t, result.Regions, 0)
	assert.Equal(t, 0.0, result.AvgDetConf)
}

func TestGenerateAndSaveOverlay_ValidInputs(t *testing.T) {
	tempDir := testutil.CreateTempDir(t)
	overlayDir := filepath.Join(tempDir, "overlays")

	// Create a simple test image
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			img.Set(x, y, color.White)
		}
	}

	result := &pipeline.OCRImageResult{
		Regions: []pipeline.OCRRegionResult{
			{
				Text:          "Test",
				RecConfidence: 0.9,
				DetConfidence: 0.8,
				Box:           struct{ X, Y, W, H int }{X: 10, Y: 10, W: 20, H: 10},
			},
		},
		AvgDetConf: 0.8,
	}

	meta := utils.ImageMetadata{
		Path: "/test/image.png",
	}

	generateAndSaveOverlay(img, result, meta, overlayDir)

	// Check if overlay directory was created
	assert.True(t, testutil.DirExists(overlayDir))

	// Check if overlay file was created
	overlayFiles, err := filepath.Glob(filepath.Join(overlayDir, "*_overlay.png"))
	require.NoError(t, err)
	assert.NotEmpty(t, overlayFiles)
}

func TestGenerateAndSaveOverlay_NilOverlay(t *testing.T) {
	tempDir := testutil.CreateTempDir(t)
	overlayDir := filepath.Join(tempDir, "overlays")

	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	result := &pipeline.OCRImageResult{} // Empty result
	meta := utils.ImageMetadata{Path: "/test/image.png"}

	// This should create an overlay file even for empty results
	generateAndSaveOverlay(img, result, meta, overlayDir)

	overlayFiles, err := filepath.Glob(filepath.Join(overlayDir, "*"))
	require.NoError(t, err)
	assert.NotEmpty(t, overlayFiles, "Overlay file should be created")
	assert.Contains(t, overlayFiles[0], "_overlay.png")
}

func TestProcessSingleImage_ValidImage(t *testing.T) {
	// Skip if models directory doesn't exist (ONNX runtime not set up)
	modelsDir := testutil.GetTestDataDir(t)
	if !testutil.DirExists(filepath.Join(modelsDir, "models")) {
		t.Skip("Models directory not found, skipping integration test")
	}

	// Create a simple test image
	tempDir := testutil.CreateTempDir(t)
	imagePath := filepath.Join(tempDir, "test.png")

	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			img.Set(x, y, color.White)
		}
	}

	file, err := os.Create(imagePath)
	require.NoError(t, err)
	defer file.Close()

	err = png.Encode(file, img)
	require.NoError(t, err)

	config := &Config{
		ModelsDir:  modelsDir,
		Workers:    1,
		Confidence: 0.3,
		MinRecConf: 0.0,
	}

	pl, err := buildPipeline(config, nil)
	require.NoError(t, err)

	result, err := processSingleImage(pl, imagePath, 0.3, 0.0, "")
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, 100, result.Width)
	assert.Equal(t, 100, result.Height)
}

func TestProcessSingleImage_UnsupportedImage(t *testing.T) {
	// Skip if models directory doesn't exist (ONNX runtime not set up)
	modelsDir := testutil.GetTestDataDir(t)
	if !testutil.DirExists(filepath.Join(modelsDir, "models")) {
		t.Skip("Models directory not found, skipping integration test")
	}

	tempDir := testutil.CreateTempDir(t)
	imagePath := filepath.Join(tempDir, "test.txt")

	err := os.WriteFile(imagePath, []byte("not an image"), 0o600)
	require.NoError(t, err)

	config := &Config{
		ModelsDir:  modelsDir,
		Workers:    1,
		Confidence: 0.3,
		MinRecConf: 0.0,
	}

	pl, err := buildPipeline(config, nil)
	require.NoError(t, err)

	result, err := processSingleImage(pl, imagePath, 0.3, 0.0, "")
	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "unsupported image format")
}

func TestProcessImagesParallel_ValidImages(t *testing.T) {
	// Skip if models directory doesn't exist (ONNX runtime not set up)
	modelsDir := testutil.GetTestDataDir(t)
	if !testutil.DirExists(filepath.Join(modelsDir, "models")) {
		t.Skip("Models directory not found, skipping integration test")
	}

	// Create test images
	tempDir := testutil.CreateTempDir(t)
	imagePaths := make([]string, 2)

	for i := 0; i < 2; i++ {
		imagePath := filepath.Join(tempDir, fmt.Sprintf("test%d.png", i))
		imagePaths[i] = imagePath

		img := image.NewRGBA(image.Rect(0, 0, 50, 50))
		for y := 0; y < 50; y++ {
			for x := 0; x < 50; x++ {
				img.Set(x, y, color.White)
			}
		}

		file, err := os.Create(imagePath)
		require.NoError(t, err)

		err = png.Encode(file, img)
		require.NoError(t, err)
		file.Close()
	}

	config := &Config{
		ModelsDir:  modelsDir,
		Workers:    1,
		Confidence: 0.3,
		MinRecConf: 0.0,
	}

	pl, err := buildPipeline(config, nil)
	require.NoError(t, err)

	results, err := processImagesParallel(pl, imagePaths, 0.3, 0.0, "")
	require.NoError(t, err)
	assert.Len(t, results, 2)

	for _, result := range results {
		require.NotNil(t, result)
		assert.Equal(t, 50, result.Width)
		assert.Equal(t, 50, result.Height)
	}
}

func TestProcessImagesParallel_WithConfidenceFilters(t *testing.T) {
	// Skip if models directory doesn't exist (ONNX runtime not set up)
	modelsDir := testutil.GetTestDataDir(t)
	if !testutil.DirExists(filepath.Join(modelsDir, "models")) {
		t.Skip("Models directory not found, skipping integration test")
	}

	// Create a test image
	tempDir := testutil.CreateTempDir(t)
	imagePath := filepath.Join(tempDir, "test.png")

	img := image.NewRGBA(image.Rect(0, 0, 50, 50))
	for y := 0; y < 50; y++ {
		for x := 0; x < 50; x++ {
			img.Set(x, y, color.White)
		}
	}

	file, err := os.Create(imagePath)
	require.NoError(t, err)
	defer file.Close()

	err = png.Encode(file, img)
	require.NoError(t, err)

	config := &Config{
		ModelsDir:  modelsDir,
		Workers:    1,
		Confidence: 0.3,
		MinRecConf: 0.0,
	}

	pl, err := buildPipeline(config, nil)
	require.NoError(t, err)

	// Test with high confidence filters (should still work even if no regions pass)
	results, err := processImagesParallel(pl, []string{imagePath}, 0.9, 0.9, "")
	require.NoError(t, err)
	assert.Len(t, results, 1)
	assert.NotNil(t, results[0])
}
