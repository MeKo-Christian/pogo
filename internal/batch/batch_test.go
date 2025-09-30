package batch

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProcessBatch_NoImageFiles(t *testing.T) {
	config := &Config{
		ModelsDir: testutil.GetTestDataDir(t),
		Workers:   1,
	}

	// Test with empty file list
	result, err := ProcessBatch([]string{}, config)
	require.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "no image files found")
}

func TestProcessBatch_InvalidImagePath(t *testing.T) {
	config := &Config{
		ModelsDir: testutil.GetTestDataDir(t),
		Workers:   1,
	}

	// Test with non-existent file
	result, err := ProcessBatch([]string{"/nonexistent/file.png"}, config)
	require.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "cannot access")
}

func TestProcessBatch_ValidImage(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Skip if models directory doesn't exist (ONNX runtime not set up)
	modelsDir := testutil.GetTestDataDir(t)
	if !testutil.DirExists(filepath.Join(modelsDir, "models")) {
		t.Skip("Models directory not found, skipping integration test")
	}

	// Create a temporary image file
	imagePath := "/tmp/test_batch_valid.png"

	// Create a minimal valid PNG file (1x1 white pixel)
	pngData := "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde" +
		"\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8" +
		"\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82"
	err := os.WriteFile(imagePath, []byte(pngData), 0o600)
	require.NoError(t, err)

	// Verify the file exists and has the right extension
	require.True(t, testutil.FileExists(imagePath))
	require.True(t, strings.HasSuffix(imagePath, ".png"))

	t.Logf("Created test image at: %s", imagePath)

	config := &Config{
		ModelsDir:        modelsDir,
		Workers:          1,
		Confidence:       0.3,
		MinRecConf:       0.0,
		ShowProgress:     false,
		Quiet:            true,
		ProgressInterval: time.Millisecond * 100,
	}

	result, err := ProcessBatch([]string{imagePath}, config)
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Len(t, result.Results, 1)
	assert.Len(t, result.ImagePaths, 1)
	assert.Equal(t, imagePath, result.ImagePaths[0])
	assert.Greater(t, result.Duration, time.Duration(0))
	assert.Equal(t, 1, result.WorkerCount)
}

func TestProcessBatch_MultipleImages(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Use test images
	imagePaths := []string{
		testutil.GetTestImagePath(t, "simple_text.png"),
		testutil.GetTestImagePath(t, "english_text.png"),
	}

	// Filter to only existing files
	var existingPaths []string
	for _, path := range imagePaths {
		if testutil.FileExists(path) {
			existingPaths = append(existingPaths, path)
		}
	}

	if len(existingPaths) == 0 {
		t.Skip("No test images found")
	}

	config := &Config{
		ModelsDir:        testutil.GetTestDataDir(t),
		Workers:          2,
		Confidence:       0.3,
		MinRecConf:       0.0,
		ShowProgress:     false,
		Quiet:            true,
		ProgressInterval: time.Millisecond * 100,
	}

	result, err := ProcessBatch(existingPaths, config)
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Len(t, result.Results, len(existingPaths))
	assert.Len(t, result.ImagePaths, len(existingPaths))
	assert.Greater(t, result.Duration, time.Duration(0))
	assert.Equal(t, 2, result.WorkerCount)
}

func TestProcessBatch_WithOverlay(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Use a test image
	imagePath := testutil.GetTestImagePath(t, "simple_text.png")
	if !testutil.FileExists(imagePath) {
		t.Skip("Test image not found")
	}

	// Create temporary directory for overlays
	overlayDir := testutil.CreateTempDir(t)

	config := &Config{
		ModelsDir:        testutil.GetTestDataDir(t),
		Workers:          1,
		Confidence:       0.3,
		MinRecConf:       0.0,
		OverlayDir:       overlayDir,
		ShowProgress:     false,
		Quiet:            true,
		ProgressInterval: time.Millisecond * 100,
	}

	result, err := ProcessBatch([]string{imagePath}, config)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Check if overlay was created
	overlayFiles, err := filepath.Glob(filepath.Join(overlayDir, "*_overlay.png"))
	require.NoError(t, err)
	assert.NotEmpty(t, overlayFiles, "Overlay file should have been created")
}

func TestProcessBatch_WithConfidenceFilters(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Use a test image
	imagePath := testutil.GetTestImagePath(t, "simple_text.png")
	if !testutil.FileExists(imagePath) {
		t.Skip("Test image not found")
	}

	config := &Config{
		ModelsDir:        testutil.GetTestDataDir(t),
		Workers:          1,
		Confidence:       0.8, // High confidence threshold
		MinRecConf:       0.8, // High recognition confidence threshold
		ShowProgress:     false,
		Quiet:            true,
		ProgressInterval: time.Millisecond * 100,
	}

	result, err := ProcessBatch([]string{imagePath}, config)
	require.NoError(t, err)
	require.NotNil(t, result)

	// With high confidence thresholds, we might get fewer or no results
	// but the processing should still succeed
	assert.Len(t, result.Results, 1)
	assert.Len(t, result.ImagePaths, 1)
}

func TestProcessBatch_PipelineBuildFailure(t *testing.T) {
	config := &Config{
		ModelsDir:        "/nonexistent/models",
		Workers:          1,
		Confidence:       0.3,
		MinRecConf:       0.0,
		ShowProgress:     false,
		Quiet:            true,
		ProgressInterval: time.Millisecond * 100,
	}

	// Create a valid image file for testing
	tempDir := testutil.CreateTempDir(t)
	imagePath := filepath.Join(tempDir, "test.png")

	// Create a minimal valid PNG file
	pngData := "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde" +
		"\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8" +
		"\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82"
	err := os.WriteFile(imagePath, []byte(pngData), 0o600)
	require.NoError(t, err)

	result, err := ProcessBatch([]string{imagePath}, config)
	require.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "failed to build OCR pipeline")
}
