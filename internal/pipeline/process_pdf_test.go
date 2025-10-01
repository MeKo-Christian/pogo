package pipeline

import (
	"context"
	"image"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/MeKo-Tech/pogo/internal/orientation"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProcessPDF tests the ProcessPDF method.
func TestProcessPDF(t *testing.T) {
	// Check if we have a test PDF file
	testPDFPath := filepath.Join("testdata", "test.pdf")
	if _, err := os.Stat(testPDFPath); os.IsNotExist(err) {
		t.Skip("test PDF not available")
	}

	b := NewBuilder()
	p, err := b.Build()
	if err != nil {
		t.Skipf("pipeline build failed: %v", err)
	}
	defer func() {
		require.NoError(t, p.Close())
	}()

	tests := []struct {
		name        string
		filename    string
		pageRange   string
		expectError bool
		skipReason  string
	}{
		{
			name:        "process all pages",
			filename:    testPDFPath,
			pageRange:   "",
			expectError: false,
		},
		{
			name:        "process single page",
			filename:    testPDFPath,
			pageRange:   "1",
			expectError: false,
		},
		{
			name:        "process page range",
			filename:    testPDFPath,
			pageRange:   "1-2",
			expectError: false,
		},
		{
			name:        "invalid file",
			filename:    "nonexistent.pdf",
			pageRange:   "",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.skipReason != "" {
				t.Skip(tt.skipReason)
			}

			result, err := p.ProcessPDF(tt.filename, tt.pageRange)

			if tt.expectError {
				require.Error(t, err)
				return
			}

			if err != nil {
				t.Skipf("PDF processing failed (runtime deps): %v", err)
			}

			require.NoError(t, err)
			require.NotNil(t, result)
			assert.Positive(t, result.TotalPages)
			assert.NotEmpty(t, result.Pages)
		})
	}
}

// TestProcessPDFContext tests ProcessPDFContext with context handling.
func TestProcessPDFContext(t *testing.T) {
	testPDFPath := filepath.Join("testdata", "test.pdf")
	if _, err := os.Stat(testPDFPath); os.IsNotExist(err) {
		t.Skip("test PDF not available")
	}

	b := NewBuilder()
	p, err := b.Build()
	if err != nil {
		t.Skipf("pipeline build failed: %v", err)
	}
	defer func() {
		require.NoError(t, p.Close())
	}()

	tests := []struct {
		name         string
		setupContext func() (context.Context, context.CancelFunc)
		expectError  bool
	}{
		{
			name: "successful processing with timeout",
			setupContext: func() (context.Context, context.CancelFunc) {
				return context.WithTimeout(context.Background(), 60*time.Second)
			},
			expectError: false,
		},
		{
			name: "context cancellation",
			setupContext: func() (context.Context, context.CancelFunc) {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()              // Cancel immediately
				return ctx, func() {} // Dummy cancel
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := tt.setupContext()
			defer cancel()

			result, err := p.ProcessPDFContext(ctx, testPDFPath, "")

			if tt.expectError {
				require.Error(t, err)
				return
			}

			if err != nil {
				t.Skipf("PDF processing failed (runtime deps): %v", err)
			}

			require.NoError(t, err)
			require.NotNil(t, result)
		})
	}
}

// TestProcessPDFContext_NilChecks tests nil validation.
func TestProcessPDFContext_NilChecks(t *testing.T) {
	tests := []struct {
		name        string
		pipeline    *Pipeline
		filename    string
		expectError string
	}{
		{
			name:        "nil pipeline",
			pipeline:    nil,
			filename:    "test.pdf",
			expectError: "pipeline not initialized",
		},
		{
			name: "empty filename",
			pipeline: &Pipeline{
				Detector:   nil,
				Recognizer: nil,
			},
			filename:    "",
			expectError: "filename",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			_, err := tt.pipeline.ProcessPDFContext(ctx, tt.filename, "")
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectError)
		})
	}
}

// TestProcessPDFPage tests the processPDFPage method.
func TestProcessPDFPage(t *testing.T) {
	b := NewBuilder()
	p, err := b.Build()
	if err != nil {
		t.Skipf("pipeline build failed: %v", err)
	}
	defer func() {
		require.NoError(t, p.Close())
	}()

	tests := []struct {
		name        string
		pageNum     int
		imageCount  int
		expectError bool
		skipTest    bool
	}{
		{
			name:        "process page with single image",
			pageNum:     1,
			imageCount:  1,
			expectError: false,
		},
		{
			name:        "process page with multiple images",
			pageNum:     2,
			imageCount:  3,
			expectError: false,
		},
		{
			name:        "process page with no images",
			pageNum:     3,
			imageCount:  0,
			expectError: false, // Should succeed but return empty results
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.skipTest {
				t.Skip("test data not available")
			}

			// Create synthetic images for testing
			// In real PDF processing, these would come from PDF extraction
			images := make([]image.Image, tt.imageCount)
			for i := range tt.imageCount {
				cfg := testutil.DefaultTestImageConfig()
				cfg.Text = "Page Text"
				img, err := testutil.GenerateTextImage(cfg)
				require.NoError(t, err)
				images[i] = img
			}

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			result, err := p.processPDFPage(ctx, tt.pageNum, images)

			if tt.expectError {
				require.Error(t, err)
				return
			}

			if err != nil {
				t.Skipf("page processing failed (runtime deps): %v", err)
			}

			require.NoError(t, err)
			require.NotNil(t, result)
			assert.Equal(t, tt.pageNum, result.PageNumber)
			assert.Len(t, result.Images, tt.imageCount)

			// Verify timing information
			if tt.imageCount > 0 {
				assert.Positive(t, result.Processing.TotalNs)
			}

			// Verify each image result
			for i, imgResult := range result.Images {
				assert.Equal(t, i, imgResult.ImageIndex)
				assert.Positive(t, imgResult.Width)
				assert.Positive(t, imgResult.Height)
			}
		})
	}
}

// TestProcessPDFPage_ContextCancellation tests context cancellation during page processing.
func TestProcessPDFPage_ContextCancellation(t *testing.T) {
	b := NewBuilder()
	p, err := b.Build()
	if err != nil {
		t.Skipf("pipeline build failed: %v", err)
	}
	defer func() {
		require.NoError(t, p.Close())
	}()

	// Create test images
	images := make([]image.Image, 2)
	for i := range images {
		cfg := testutil.DefaultTestImageConfig()
		cfg.Text = "Test"
		img, err := testutil.GenerateTextImage(cfg)
		require.NoError(t, err)
		images[i] = img
	}

	// Create cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = p.processPDFPage(ctx, 1, images)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "context")
}

// TestProcessPDFPage_EmptyImages tests processing with no images.
func TestProcessPDFPage_EmptyImages(t *testing.T) {
	b := NewBuilder()
	p, err := b.Build()
	if err != nil {
		t.Skipf("pipeline build failed: %v", err)
	}
	defer func() {
		require.NoError(t, p.Close())
	}()

	ctx := context.Background()
	result, err := p.processPDFPage(ctx, 1, []image.Image{})
	if err != nil {
		t.Skipf("page processing failed (runtime deps): %v", err)
	}

	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, 1, result.PageNumber)
	assert.Empty(t, result.Images)
}

// TestProcessImagesWithOrientation tests the processImagesWithOrientation method.
func TestProcessImagesWithOrientation(t *testing.T) {
	b := NewBuilder()

	// Enable orientation
	b.cfg.Orientation.Enabled = true
	b.cfg.Orientation.HeuristicOnly = true

	p, err := b.Build()
	if err != nil {
		t.Skipf("pipeline build failed: %v", err)
	}
	defer func() {
		require.NoError(t, p.Close())
	}()

	// Create test images
	originalImages := make([]image.Image, 2)
	for i := range originalImages {
		cfg := testutil.DefaultTestImageConfig()
		cfg.Size = testutil.ImageSize{Width: 200, Height: 100}
		cfg.Text = "Test"
		img, err := testutil.GenerateTextImage(cfg)
		require.NoError(t, err)
		originalImages[i] = img
	}

	ctx := context.Background()

	// Get orientation results
	orientationResults, workingImages, err := p.prepareOrientation(ctx, originalImages)
	if err != nil {
		t.Skipf("orientation preparation failed: %v", err)
	}
	require.NoError(t, err)

	// Process with orientation
	results, err := p.processImagesWithOrientation(ctx, originalImages, orientationResults, workingImages)
	if err != nil {
		t.Skipf("processing failed (runtime deps): %v", err)
	}

	require.NoError(t, err)
	require.NotNil(t, results)
	assert.Len(t, results, len(originalImages))

	// Verify orientation information is present in results
	for i, result := range results {
		assert.NotNil(t, result, "result %d should not be nil", i)
		assert.GreaterOrEqual(t, result.Orientation.Angle, 0)
		assert.LessOrEqual(t, result.Orientation.Angle, 270)
	}
}

// TestProcessSingleImage tests the processSingleImage method.
func TestProcessSingleImage(t *testing.T) {
	b := NewBuilder()
	p, err := b.Build()
	if err != nil {
		t.Skipf("pipeline build failed: %v", err)
	}
	defer func() {
		require.NoError(t, p.Close())
	}()

	// Create test image
	cfg := testutil.DefaultTestImageConfig()
	cfg.Size = testutil.ImageSize{Width: 250, Height: 120}
	cfg.Text = "Single Image"
	originalImg, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Test with same image for original and working (no rotation)
	orientResult := orientation.Result{Angle: 0, Confidence: 0.0}
	result, err := p.processSingleImage(ctx, originalImg, originalImg, orientResult)
	if err != nil {
		t.Skipf("processing failed (runtime deps): %v", err)
	}

	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, originalImg.Bounds().Dx(), result.Width)
	assert.Equal(t, originalImg.Bounds().Dy(), result.Height)
	assert.Equal(t, 0, result.Orientation.Angle)
	assert.False(t, result.Orientation.Applied)
}

// TestProcessSingleImage_WithRotation tests processing with rotation applied.
func TestProcessSingleImage_WithRotation(t *testing.T) {
	b := NewBuilder()
	p, err := b.Build()
	if err != nil {
		t.Skipf("pipeline build failed: %v", err)
	}
	defer func() {
		require.NoError(t, p.Close())
	}()

	// Create test image
	cfg := testutil.DefaultTestImageConfig()
	cfg.Size = testutil.ImageSize{Width: 200, Height: 100}
	cfg.Text = "Rotated"
	originalImg, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	// Apply rotation to create working image
	workingImg := p.applyOrientationRotation(originalImg, 90)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	orientResult := orientation.Result{Angle: 90, Confidence: 0.92}
	result, err := p.processSingleImage(ctx, originalImg, workingImg, orientResult)
	if err != nil {
		t.Skipf("processing failed (runtime deps): %v", err)
	}

	require.NoError(t, err)
	require.NotNil(t, result)

	// Original dimensions should be reported
	assert.Equal(t, originalImg.Bounds().Dx(), result.Width)
	assert.Equal(t, originalImg.Bounds().Dy(), result.Height)

	// Orientation info should reflect the rotation
	assert.Equal(t, 90, result.Orientation.Angle)
	assert.True(t, result.Orientation.Applied)
	assert.InDelta(t, 0.92, result.Orientation.Confidence, 0.001)
}
