package detector

import (
	"fmt"
	"image"
	"image/color"
	"os"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/onnx"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPreprocessBatchImages(t *testing.T) {
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

	// Test with valid images of same size
	images := []image.Image{
		testutil.CreateTestImage(320, 240, color.RGBA{255, 255, 255, 255}),
		testutil.CreateTestImage(320, 240, color.RGBA{255, 255, 255, 255}),
		testutil.CreateTestImage(320, 240, color.RGBA{255, 255, 255, 255}),
	}

	tensors, results, height, width, err := detector.preprocessBatchImages(images)
	require.NoError(t, err)

	assert.Len(t, tensors, len(images))
	assert.Len(t, results, len(images))
	assert.Positive(t, height)
	assert.Positive(t, width)

	// Verify all tensors have same dimensions
	for _, tensor := range tensors {
		assert.Len(t, tensor, height*width*3) // 3 channels
	}

	// Verify results have correct original dimensions
	for _, result := range results {
		assert.Equal(t, 320, result.OriginalWidth)
		assert.Equal(t, 240, result.OriginalHeight)
	}
}

func TestPreprocessBatchImages_NilImage(t *testing.T) {
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
		testutil.CreateTestImage(320, 240, color.RGBA{255, 255, 255, 255}),
		nil,
		testutil.CreateTestImage(320, 240, color.RGBA{255, 255, 255, 255}),
	}

	tensors, results, height, width, err := detector.preprocessBatchImages(images)
	require.Error(t, err)
	assert.Nil(t, tensors)
	assert.Nil(t, results)
	assert.Zero(t, height)
	assert.Zero(t, width)
	assert.Contains(t, err.Error(), "image at index 1 is nil")
}

func TestPreprocessBatchImages_DifferentSizes(t *testing.T) {
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

	// Test with images of different sizes (should fail after preprocessing)
	images := []image.Image{
		testutil.CreateTestImage(320, 240, color.RGBA{255, 255, 255, 255}),
		testutil.CreateTestImage(640, 480, color.RGBA{255, 255, 255, 255}), // Different size
	}

	tensors, results, height, width, err := detector.preprocessBatchImages(images)
	require.Error(t, err)
	assert.Nil(t, tensors)
	assert.Nil(t, results)
	assert.Zero(t, height)
	assert.Zero(t, width)
	assert.Contains(t, err.Error(), "different dimensions after preprocessing")
}

func TestRunBatchInferenceCore(t *testing.T) {
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

	// Create a single image and preprocess it to get a valid tensor
	img := testutil.CreateTestImage(320, 240, color.RGBA{255, 255, 255, 255})
	tensor, err := detector.preprocessImage(img)
	require.NoError(t, err)

	// Run batch inference core with the preprocessed tensor
	outputData, batchSize, channels, outputHeight, outputWidth, err := detector.runBatchInferenceCore(tensor)
	require.NoError(t, err)

	assert.Equal(t, 1, batchSize) // Single image in batch
	assert.Equal(t, 1, channels)  // Output channels for detection
	assert.Positive(t, outputHeight)
	assert.Positive(t, outputWidth)
	assert.NotEmpty(t, outputData)

	// Verify output data size matches expected dimensions
	expectedSize := batchSize * channels * outputHeight * outputWidth
	assert.Len(t, outputData, expectedSize)
}

func TestRunBatchInferenceCore_NilSession(t *testing.T) {
	// Skip if model not available
	modelPath := models.GetDetectionModelPath("", false)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Detection model not available, skipping test")
	}

	config := DefaultConfig()
	config.ModelPath = modelPath

	detector, err := NewDetector(config)
	require.NoError(t, err)

	// Close detector to nil the session
	_ = detector.Close()

	// Create a test batch tensor
	tensors := [][]float32{
		make([]float32, 3*240*320),
	}
	batchTensor, err := onnx.NewBatchImageTensor(tensors, 3, 240, 320)
	require.NoError(t, err)

	// Run batch inference core with nil session
	outputData, batchSize, channels, outputHeight, outputWidth, err := detector.runBatchInferenceCore(batchTensor)
	require.Error(t, err)
	assert.Nil(t, outputData)
	assert.Zero(t, batchSize)
	assert.Zero(t, channels)
	assert.Zero(t, outputHeight)
	assert.Zero(t, outputWidth)
	assert.Contains(t, err.Error(), "detector session is nil")
}

func TestSplitBatchOutput(t *testing.T) {
	// Create test results
	results := []*DetectionResult{
		{OriginalWidth: 320, OriginalHeight: 240},
		{OriginalWidth: 640, OriginalHeight: 480},
	}

	// Create mock output data for 2 images, 1 channel, 60x80 output each
	batchSize := 2
	channels := 1
	outputHeight := 60
	outputWidth := 80
	elementsPerImage := channels * outputHeight * outputWidth

	outputData := make([]float32, batchSize*elementsPerImage)
	// Fill with some test data
	for i := range outputData {
		outputData[i] = float32(i%256) / 255.0 // Mock probability values
	}

	// Split the batch output
	splitBatchOutput(outputData, results, batchSize, channels, outputHeight, outputWidth)

	// Verify results
	assert.Len(t, results, 2)

	// First result
	assert.Equal(t, 320, results[0].OriginalWidth)
	assert.Equal(t, 240, results[0].OriginalHeight)
	assert.Equal(t, outputWidth, results[0].Width)
	assert.Equal(t, outputHeight, results[0].Height)
	assert.Len(t, results[0].ProbabilityMap, elementsPerImage)
	assert.Equal(t, outputData[0:elementsPerImage], results[0].ProbabilityMap)

	// Second result
	assert.Equal(t, 640, results[1].OriginalWidth)
	assert.Equal(t, 480, results[1].OriginalHeight)
	assert.Equal(t, outputWidth, results[1].Width)
	assert.Equal(t, outputHeight, results[1].Height)
	assert.Len(t, results[1].ProbabilityMap, elementsPerImage)
	assert.Equal(t, outputData[elementsPerImage:2*elementsPerImage], results[1].ProbabilityMap)
}

func TestBatchDetectionResult_Structure(t *testing.T) {
	results := []*DetectionResult{
		{
			ProbabilityMap: []float32{0.1, 0.2, 0.3},
			Width:          10,
			Height:         10,
			OriginalWidth:  320,
			OriginalHeight: 240,
			ProcessingTime: 1000000, // 1ms in nanoseconds
		},
	}

	batchResult := &BatchDetectionResult{
		Results:       results,
		TotalTime:     2000000, // 2ms in nanoseconds
		ThroughputIPS: 2.5,
		MemoryUsageMB: 15.7,
	}

	assert.Len(t, batchResult.Results, 1)
	assert.Equal(t, int64(2000000), batchResult.TotalTime)
	assert.InDelta(t, 2.5, batchResult.ThroughputIPS, 0.01)
	assert.InDelta(t, 15.7, batchResult.MemoryUsageMB, 0.01)
}

func TestRunBatchInference_SingleImage(t *testing.T) {
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

	// Test with single image
	images := []image.Image{
		testutil.CreateTestImageWithText("Single Batch", 320, 240),
	}

	batchResult, err := detector.RunBatchInference(images)
	require.NoError(t, err)
	require.NotNil(t, batchResult)

	assert.Len(t, batchResult.Results, 1)
	assert.Positive(t, batchResult.TotalTime)
	assert.Greater(t, batchResult.ThroughputIPS, 0.0)

	result := batchResult.Results[0]
	assert.NotEmpty(t, result.ProbabilityMap)
	assert.Positive(t, result.Width)
	assert.Positive(t, result.Height)
	assert.Equal(t, 320, result.OriginalWidth)
	assert.Equal(t, 240, result.OriginalHeight)
	assert.Positive(t, result.ProcessingTime)
}

func TestRunBatchInference_LargeBatch(t *testing.T) {
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

	// Test with larger batch (5 images)
	const batchSize = 5
	images := make([]image.Image, batchSize)
	for i := range images {
		images[i] = testutil.CreateTestImageWithText(fmt.Sprintf("Image %d", i), 320, 240)
	}

	batchResult, err := detector.RunBatchInference(images)
	require.NoError(t, err)
	require.NotNil(t, batchResult)

	assert.Len(t, batchResult.Results, batchSize)
	assert.Positive(t, batchResult.TotalTime)
	assert.Greater(t, batchResult.ThroughputIPS, 0.0)

	// Verify all results
	for _, result := range batchResult.Results {
		assert.NotEmpty(t, result.ProbabilityMap)
		assert.Positive(t, result.Width)
		assert.Positive(t, result.Height)
		assert.Equal(t, 320, result.OriginalWidth)
		assert.Equal(t, 240, result.OriginalHeight)
		assert.Positive(t, result.ProcessingTime)

		// Verify probability values are in valid range
		for _, prob := range result.ProbabilityMap {
			assert.GreaterOrEqual(t, prob, float32(0.0))
			assert.LessOrEqual(t, prob, float32(1.0))
		}
	}
}
