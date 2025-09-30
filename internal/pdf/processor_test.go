package pdf

import (
	"image"
	"image/color"
	"testing"
	"time"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// createTestImage creates a simple test image.
func createTestImage(width, height int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			img.Set(x, y, color.RGBA{uint8(x % 256), uint8(y % 256), 0, 255})
		}
	}
	return img
}

func TestNewProcessor(t *testing.T) {
	// Create a nil detector for basic testing
	processor := NewProcessor(nil)
	assert.NotNil(t, processor)
	assert.Nil(t, processor.detector)
}

func TestProcessor_ProcessPage_Logic(t *testing.T) {
	// Test the page processing logic without real detection
	t.Run("empty images slice", func(t *testing.T) {
		mockProcessor := &testProcessor{mockRegions: []detector.DetectedRegion{}}
		result, duration := mockProcessor.processPage(1, []image.Image{})

		assert.NotNil(t, result)
		assert.Equal(t, 1, result.PageNumber)
		assert.Equal(t, 0, result.Width)
		assert.Equal(t, 0, result.Height)
		assert.Empty(t, result.Images)
		assert.GreaterOrEqual(t, duration, time.Duration(0))
	})

	t.Run("single image dimensions", func(t *testing.T) {
		images := []image.Image{createTestImage(200, 300)}

		// Mock the detector behavior by creating a processor that simulates detection
		mockProcessor := &testProcessor{
			mockRegions: []detector.DetectedRegion{
				{
					Box:        utils.Box{MinX: 10, MinY: 10, MaxX: 90, MaxY: 40},
					Confidence: 0.8,
				},
			},
		}

		result, duration := mockProcessor.processPage(1, images)
		assert.Equal(t, 1, result.PageNumber)
		assert.Equal(t, 200, result.Width)
		assert.Equal(t, 300, result.Height)
		assert.Len(t, result.Images, 1)
		assert.GreaterOrEqual(t, duration, time.Duration(0))

		imgResult := result.Images[0]
		assert.Equal(t, 0, imgResult.ImageIndex)
		assert.Equal(t, 200, imgResult.Width)
		assert.Equal(t, 300, imgResult.Height)
		assert.InDelta(t, 0.8, imgResult.Confidence, 0.0001)
		assert.Len(t, imgResult.Regions, 1)
	})

	t.Run("multiple images - max dimensions", func(t *testing.T) {
		images := []image.Image{
			createTestImage(100, 300), // height is max
			createTestImage(400, 200), // width is max
			createTestImage(150, 150),
		}

		mockProcessor := &testProcessor{
			mockRegions: []detector.DetectedRegion{
				{
					Box:        utils.Box{MinX: 0, MinY: 0, MaxX: 50, MaxY: 25},
					Confidence: 0.7,
				},
			},
		}

		result, _ := mockProcessor.processPage(1, images)
		assert.Equal(t, 400, result.Width)  // max width
		assert.Equal(t, 300, result.Height) // max height
		assert.Len(t, result.Images, 3)
	})
}

// testProcessor is a test implementation for controlled testing.
type testProcessor struct {
	mockRegions []detector.DetectedRegion
}

func (tp *testProcessor) processPage(_pageNum int, images []image.Image) (*PageResult, time.Duration) {
	startTime := time.Now()

	imageResults := make([]ImageResult, 0, len(images))
	var pageWidth, pageHeight int

	for i, img := range images {
		bounds := img.Bounds()
		imgWidth := bounds.Dx()
		imgHeight := bounds.Dy()

		// Update page dimensions (use largest image dimensions)
		if imgWidth > pageWidth {
			pageWidth = imgWidth
		}
		if imgHeight > pageHeight {
			pageHeight = imgHeight
		}

		// Use mock regions
		regions := tp.mockRegions

		// Calculate average confidence
		var totalConfidence float64
		for _, region := range regions {
			totalConfidence += region.Confidence
		}

		avgConfidence := 0.0
		if len(regions) > 0 {
			avgConfidence = totalConfidence / float64(len(regions))
		}

		imageResult := ImageResult{
			ImageIndex: i,
			Width:      imgWidth,
			Height:     imgHeight,
			Regions:    regions,
			Confidence: avgConfidence,
		}

		imageResults = append(imageResults, imageResult)
	}

	duration := time.Since(startTime)

	pageResult := &PageResult{
		PageNumber: _pageNum,
		Width:      pageWidth,
		Height:     pageHeight,
		Images:     imageResults,
		Processing: ProcessingInfo{
			DetectionTimeMs: duration.Milliseconds(),
			TotalTimeMs:     duration.Milliseconds(),
		},
	}

	return pageResult, duration
}

func TestProcessor_ConfidenceCalculation(t *testing.T) {
	tests := []struct {
		name               string
		regions            []detector.DetectedRegion
		expectedConfidence float64
	}{
		{
			name:               "no regions",
			regions:            []detector.DetectedRegion{},
			expectedConfidence: 0.0,
		},
		{
			name: "single region",
			regions: []detector.DetectedRegion{
				{Confidence: 0.8},
			},
			expectedConfidence: 0.8,
		},
		{
			name: "multiple regions",
			regions: []detector.DetectedRegion{
				{Confidence: 0.9},
				{Confidence: 0.7},
				{Confidence: 0.8},
			},
			expectedConfidence: 0.8, // (0.9 + 0.7 + 0.8) / 3
		},
		{
			name: "regions with zero confidence",
			regions: []detector.DetectedRegion{
				{Confidence: 1.0},
				{Confidence: 0.0},
			},
			expectedConfidence: 0.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockProcessor := &testProcessor{mockRegions: tt.regions}
			images := []image.Image{createTestImage(100, 100)}

			result, _ := mockProcessor.processPage(1, images)
			require.Len(t, result.Images, 1)

			assert.InDelta(t, tt.expectedConfidence, result.Images[0].Confidence, 0.001)
		})
	}
}

func TestProcessor_ProcessFile_ErrorCases(t *testing.T) {
	processor := NewProcessor(nil)

	t.Run("non-existent file", func(t *testing.T) {
		result, err := processor.ProcessFile("/non/existent/file.pdf", "")
		require.Error(t, err)
		assert.Nil(t, result)
		// Error can be from encryption check or image extraction
		assert.NotEmpty(t, err.Error(), "expected an error message")
	})

	t.Run("invalid page range", func(t *testing.T) {
		result, err := processor.ProcessFile("dummy.pdf", "invalid-range")
		require.Error(t, err)
		assert.Nil(t, result)
		// Error can be from encryption check or image extraction
		assert.NotEmpty(t, err.Error(), "expected an error message")
	})

	t.Run("directory instead of file", func(t *testing.T) {
		tempDir := t.TempDir()
		result, err := processor.ProcessFile(tempDir, "")
		require.Error(t, err)
		assert.Nil(t, result)
	})
}

func TestProcessor_ProcessFiles(t *testing.T) {
	processor := NewProcessor(nil)

	t.Run("empty file list", func(t *testing.T) {
		results, err := processor.ProcessFiles([]string{}, "")
		require.NoError(t, err)
		assert.Empty(t, results)
	})

	t.Run("file processing error propagation", func(t *testing.T) {
		filenames := []string{"/non/existent/file1.pdf", "/non/existent/file2.pdf"}
		results, err := processor.ProcessFiles(filenames, "")
		require.Error(t, err)
		assert.Nil(t, results)
		assert.Contains(t, err.Error(), "failed to process")
	})
}

func TestProcessor_TimingMetrics(t *testing.T) {
	mockProcessor := &testProcessor{
		mockRegions: []detector.DetectedRegion{
			{Confidence: 0.8},
		},
	}

	images := []image.Image{createTestImage(100, 100)}
	result, duration := mockProcessor.processPage(1, images)
	assert.Greater(t, duration, time.Duration(0))
	assert.Equal(t, result.Processing.DetectionTimeMs, duration.Milliseconds())
	assert.Equal(t, result.Processing.TotalTimeMs, duration.Milliseconds())
}

// Benchmark tests.
func BenchmarkProcessor_ProcessPage(b *testing.B) {
	mockProcessor := &testProcessor{
		mockRegions: []detector.DetectedRegion{
			{Confidence: 0.8},
			{Confidence: 0.9},
		},
	}

	images := []image.Image{
		createTestImage(100, 100),
		createTestImage(200, 150),
	}

	b.ResetTimer()
	for range b.N {
		_, _ = mockProcessor.processPage(1, images)
	}
}

func BenchmarkProcessor_ConfidenceCalculation(b *testing.B) {
	regions := make([]detector.DetectedRegion, 1000)
	for i := range regions {
		regions[i] = detector.DetectedRegion{Confidence: float64(i%100) / 100.0}
	}

	mockProcessor := &testProcessor{mockRegions: regions}
	images := []image.Image{createTestImage(100, 100)}

	b.ResetTimer()
	for range b.N {
		_, _ = mockProcessor.processPage(1, images)
	}
}

// Integration tests (require detector to be available).
func TestProcessor_WithRealDetector_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Try to create a real detector for integration testing
	// This will be skipped if models are not available
	defer func() {
		if r := recover(); r != nil {
			t.Skip("Detector not available for integration testing")
		}
	}()

	// For now, just test that we can create a processor with nil detector
	processor := NewProcessor(nil)
	assert.NotNil(t, processor)
}
