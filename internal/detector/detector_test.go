package detector

import (
	"image"
	"image/color"
	"os"
	"path/filepath"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig()

	assert.Equal(t, models.GetDetectionModelPath("", false), config.ModelPath)
	assert.InDelta(t, float32(0.3), config.DbThresh, 0.0001)
	assert.InDelta(t, float32(0.5), config.DbBoxThresh, 0.0001)
	assert.Equal(t, 960, config.MaxImageSize)
	assert.False(t, config.UseServerModel)
	assert.Equal(t, 0, config.NumThreads)
}

func TestNewDetector_InvalidModelPath(t *testing.T) {
	config := Config{
		ModelPath:    "nonexistent/model.onnx",
		DbThresh:     0.3,
		DbBoxThresh:  0.5,
		MaxImageSize: 960,
	}

	detector, err := NewDetector(config)
	require.Error(t, err)
	assert.Nil(t, detector)
	assert.Contains(t, err.Error(), "model file not found")
}

func TestNewDetector_EmptyModelPath(t *testing.T) {
	config := Config{
		ModelPath: "",
	}

	detector, err := NewDetector(config)
	require.Error(t, err)
	assert.Nil(t, detector)
	assert.Contains(t, err.Error(), "model path cannot be empty")
}

func TestNewDetector_ValidModel(t *testing.T) {
	// Check if model file exists, skip test if not available
	modelPath := models.GetDetectionModelPath("", false)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Detection model not available, skipping test")
	}

	config := DefaultConfig()
	config.ModelPath = modelPath

	detector, err := NewDetector(config)
	require.NoError(t, err)
	require.NotNil(t, detector)
	defer func() {
		require.NoError(t, detector.Close())
	}()

	// Verify detector properties
	assert.Equal(t, config, detector.GetConfig())
	assert.NotEmpty(t, detector.GetInputShape())
	assert.NotEmpty(t, detector.GetOutputShape())

	// Verify input shape is 4D [N, C, H, W]
	inputShape := detector.GetInputShape()
	assert.Len(t, inputShape, 4)
	// Note: inputShape[0] might be -1 for dynamic batch size
	assert.True(t, inputShape[0] == 1 || inputShape[0] == -1, "Batch dimension should be 1 or -1 (dynamic)")
	assert.Equal(t, int64(3), inputShape[1]) // RGB channels

	// Get model info
	info := detector.GetModelInfo()
	assert.Equal(t, modelPath, info["model_path"])
	assert.NotEmpty(t, info["input_name"])
	assert.NotEmpty(t, info["output_name"])
}

func TestDetector_PreprocessImage(t *testing.T) {
	// Create a test image
	img := testutil.CreateTestImage(640, 480, color.RGBA{255, 255, 255, 255})

	// Skip if model not available
	modelPath := models.GetDetectionModelPath("", false)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Detection model not available, skipping test")
	}

	config := DefaultConfig()
	config.ModelPath = modelPath

	detector, err := NewDetector(config)
	require.NoError(t, err)
	defer func() {
		require.NoError(t, detector.Close())
	}()

	// Test preprocessing
	tensor, err := detector.preprocessImage(img)
	require.NoError(t, err)

	// Verify tensor properties
	assert.Len(t, tensor.Shape, 4)
	assert.Equal(t, int64(1), tensor.Shape[0]) // Batch size
	assert.Equal(t, int64(3), tensor.Shape[1]) // RGB channels

	// Verify tensor data length matches shape
	expectedLen := tensor.Shape[0] * tensor.Shape[1] * tensor.Shape[2] * tensor.Shape[3]
	assert.Len(t, tensor.Data, int(expectedLen))

	// Verify tensor values are normalized (0-1 range)
	for _, val := range tensor.Data {
		assert.GreaterOrEqual(t, val, float32(0.0))
		assert.LessOrEqual(t, val, float32(1.0))
	}
}

func TestDetector_PreprocessImage_NilInput(t *testing.T) {
	// Skip if model not available
	modelPath := models.GetDetectionModelPath("", false)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Detection model not available, skipping test")
	}

	config := DefaultConfig()
	config.ModelPath = modelPath

	detector, err := NewDetector(config)
	require.NoError(t, err)
	defer func() {
		require.NoError(t, detector.Close())
	}()

	// Test with nil image
	_, err = detector.preprocessImage(nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "input image is nil")
}

func TestDetector_RunInference(t *testing.T) {
	// Skip if model not available
	modelPath := models.GetDetectionModelPath("", false)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Detection model not available, skipping test")
	}

	config := DefaultConfig()
	config.ModelPath = modelPath

	detector, err := NewDetector(config)
	require.NoError(t, err)
	defer func() { _ = detector.Close() }()

	// Create a test image with some text-like patterns
	img := testutil.CreateTestImageWithText("Hello World", 640, 480)

	// Run inference
	result, err := detector.RunInference(img)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify result properties
	assert.NotEmpty(t, result.ProbabilityMap)
	assert.Positive(t, result.Width)
	assert.Positive(t, result.Height)
	assert.Equal(t, 640, result.OriginalWidth)
	assert.Equal(t, 480, result.OriginalHeight)
	assert.Positive(t, result.ProcessingTime)

	// Verify probability map has expected size
	expectedSize := result.Width * result.Height
	assert.Len(t, result.ProbabilityMap, expectedSize)

	// Verify probability values are in valid range [0, 1]
	for _, prob := range result.ProbabilityMap {
		assert.GreaterOrEqual(t, prob, float32(0.0))
		assert.LessOrEqual(t, prob, float32(1.0))
	}
}

func TestDetector_RunInference_NilImage(t *testing.T) {
	// Skip if model not available
	modelPath := models.GetDetectionModelPath("", false)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Detection model not available, skipping test")
	}

	config := DefaultConfig()
	config.ModelPath = modelPath

	detector, err := NewDetector(config)
	require.NoError(t, err)
	defer func() { _ = detector.Close() }()

	// Test with nil image
	result, err := detector.RunInference(nil)
	require.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "input image is nil")
}

func TestDetector_RunInferenceWithMetrics(t *testing.T) {
	// Skip if model not available
	modelPath := models.GetDetectionModelPath("", false)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Detection model not available, skipping test")
	}

	config := DefaultConfig()
	config.ModelPath = modelPath

	detector, err := NewDetector(config)
	require.NoError(t, err)
	defer func() { _ = detector.Close() }()

	// Create a test image
	img := testutil.CreateTestImageWithText("Test", 320, 240)

	// Run inference with metrics
	result, metrics, err := detector.RunInferenceWithMetrics(img)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, metrics)

	// Verify metrics
	assert.Positive(t, metrics.PreprocessingTime)
	assert.Positive(t, metrics.ModelExecutionTime)
	assert.Positive(t, metrics.PostprocessingTime)
	assert.Positive(t, metrics.TotalTime)
	assert.Greater(t, metrics.ThroughputIPS, 0.0)
	assert.GreaterOrEqual(t, metrics.MemoryAllocMB, 0.0)
	assert.Greater(t, metrics.TensorSizeMB, 0.0)

	// Verify time components add up approximately to total
	componentSum := metrics.PreprocessingTime + metrics.ModelExecutionTime + metrics.PostprocessingTime
	// Allow some tolerance for measurement overhead
	assert.InDelta(t, float64(metrics.TotalTime), float64(componentSum), float64(metrics.TotalTime)*0.1)
}

func TestDetector_RunBatchInference(t *testing.T) {
	// Skip if model not available
	modelPath := models.GetDetectionModelPath("", false)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Detection model not available, skipping test")
	}

	config := DefaultConfig()
	config.ModelPath = modelPath

	detector, err := NewDetector(config)
	require.NoError(t, err)
	defer func() { _ = detector.Close() }()

	// Create multiple test images
	images := []image.Image{
		testutil.CreateTestImageWithText("Image 1", 320, 240),
		testutil.CreateTestImageWithText("Image 2", 320, 240),
		testutil.CreateTestImageWithText("Image 3", 320, 240),
	}

	// Run batch inference
	batchResult, err := detector.RunBatchInference(images)
	require.NoError(t, err)
	require.NotNil(t, batchResult)

	// Verify batch result
	assert.Len(t, batchResult.Results, len(images))
	assert.Positive(t, batchResult.TotalTime)
	assert.Greater(t, batchResult.ThroughputIPS, 0.0)
	assert.GreaterOrEqual(t, batchResult.MemoryUsageMB, 0.0)

	// Verify individual results
	for i, result := range batchResult.Results {
		assert.NotEmpty(t, result.ProbabilityMap)
		assert.Positive(t, result.Width)
		assert.Positive(t, result.Height)
		assert.Equal(t, 320, result.OriginalWidth)
		assert.Equal(t, 240, result.OriginalHeight)
		assert.Positive(t, result.ProcessingTime)

		// All results should have same dimensions after preprocessing
		if i > 0 {
			assert.Equal(t, batchResult.Results[0].Width, result.Width)
			assert.Equal(t, batchResult.Results[0].Height, result.Height)
		}
	}
}

func TestDetector_RunBatchInference_EmptyInput(t *testing.T) {
	// Skip if model not available
	modelPath := models.GetDetectionModelPath("", false)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Detection model not available, skipping test")
	}

	config := DefaultConfig()
	config.ModelPath = modelPath

	detector, err := NewDetector(config)
	require.NoError(t, err)
	defer func() { _ = detector.Close() }()

	// Test with empty slice
	result, err := detector.RunBatchInference([]image.Image{})
	require.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "no images provided")
}

func TestDetector_RunBatchInference_NilImageInBatch(t *testing.T) {
	// Skip if model not available
	modelPath := models.GetDetectionModelPath("", false)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Detection model not available, skipping test")
	}

	config := DefaultConfig()
	config.ModelPath = modelPath

	detector, err := NewDetector(config)
	require.NoError(t, err)
	defer func() { _ = detector.Close() }()

	// Test with nil image in batch
	images := []image.Image{
		testutil.CreateTestImageWithText("Valid", 320, 240),
		nil,
		testutil.CreateTestImageWithText("Also Valid", 320, 240),
	}

	result, err := detector.RunBatchInference(images)
	require.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "image at index 1 is nil")
}

func TestDetector_BenchmarkDetection(t *testing.T) {
	// Skip if model not available
	modelPath := models.GetDetectionModelPath("", false)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Detection model not available, skipping test")
	}

	config := DefaultConfig()
	config.ModelPath = modelPath

	detector, err := NewDetector(config)
	require.NoError(t, err)
	defer func() { _ = detector.Close() }()

	// Create a test image
	img := testutil.CreateTestImageWithText("Benchmark", 320, 240)

	// Run benchmark with small number of iterations for testing
	benchResult, err := detector.BenchmarkDetection(img, 3)
	require.NoError(t, err)
	require.NotNil(t, benchResult)

	// Verify benchmark result
	assert.Equal(t, "detection_inference", benchResult.Name)
	assert.Equal(t, 3, benchResult.Iterations)
	assert.Positive(t, benchResult.Duration)
	assert.NoError(t, benchResult.Error)
}

func TestDetector_Close(t *testing.T) {
	// Skip if model not available
	modelPath := models.GetDetectionModelPath("", false)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Detection model not available, skipping test")
	}

	config := DefaultConfig()
	config.ModelPath = modelPath

	detector, err := NewDetector(config)
	require.NoError(t, err)

	// Close detector
	err = detector.Close()
	require.NoError(t, err)

	// Verify that subsequent operations fail gracefully
	img := testutil.CreateTestImage(100, 100, color.RGBA{255, 255, 255, 255})
	result, err := detector.RunInference(img)
	require.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "detector session is nil")
}

func TestDetector_ConcurrentAccess(t *testing.T) {
	// Skip if model not available
	modelPath := models.GetDetectionModelPath("", false)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Detection model not available, skipping test")
	}

	config := DefaultConfig()
	config.ModelPath = modelPath

	detector, err := NewDetector(config)
	require.NoError(t, err)
	defer func() { _ = detector.Close() }()

	// Test concurrent access to detector methods
	img := testutil.CreateTestImageWithText("Concurrent", 320, 240)

	// Run multiple goroutines accessing detector concurrently
	const numGoroutines = 5
	results := make(chan error, numGoroutines)

	for range numGoroutines {
		go func() {
			_, err := detector.RunInference(img)
			results <- err
		}()
	}

	// Wait for all goroutines to complete
	for range numGoroutines {
		err := <-results
		assert.NoError(t, err)
	}
}

// Integration test with actual model files.
func TestDetector_Integration(t *testing.T) {
	// This test requires actual model files and ONNX Runtime
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Check for both mobile and server models
	mobileModel := models.GetDetectionModelPath("", false)
	serverModel := models.GetDetectionModelPath("", true)

	if _, err := os.Stat(mobileModel); os.IsNotExist(err) {
		t.Skip("Mobile detection model not available, skipping integration test")
	}

	// Test mobile model
	t.Run("MobileModel", func(t *testing.T) {
		config := DefaultConfig()
		config.ModelPath = mobileModel

		detector, err := NewDetector(config)
		require.NoError(t, err)
		defer func() { _ = detector.Close() }()

		// Load test fixture if available
		testImagesDir := "../../testdata/images"
		if _, err := os.Stat(testImagesDir); err != nil {
			return
		}
		entries, err := os.ReadDir(testImagesDir)
		if err != nil || len(entries) == 0 {
			return
		}
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			ext := filepath.Ext(entry.Name())
			if ext != ".png" && ext != ".jpg" {
				continue
			}
			imagePath := filepath.Join(testImagesDir, entry.Name())
			img, err := testutil.LoadImageFile(imagePath)
			if err != nil {
				continue
			}
			result, err := detector.RunInference(img)
			require.NoError(t, err)
			require.NotNil(t, result)

			// Basic sanity checks on real image
			assert.NotEmpty(t, result.ProbabilityMap)
			assert.Positive(t, result.Width)
			assert.Positive(t, result.Height)
			break
		}
	})

	// Test server model if available
	if _, err := os.Stat(serverModel); err == nil {
		t.Run("ServerModel", func(t *testing.T) {
			config := DefaultConfig()
			config.ModelPath = serverModel
			config.UseServerModel = true

			detector, err := NewDetector(config)
			require.NoError(t, err)
			defer func() { _ = detector.Close() }()

			// Create test image
			img := testutil.CreateTestImageWithText("Server Model Test", 640, 480)

			result, err := detector.RunInference(img)
			require.NoError(t, err)
			require.NotNil(t, result)

			assert.NotEmpty(t, result.ProbabilityMap)
			assert.Positive(t, result.Width)
			assert.Positive(t, result.Height)
		})
	}
}
