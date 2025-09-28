package pipeline

import (
	"context"
	"errors"
	"image"
	"image/color"
	"runtime"
	"testing"
	"time"

	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test helper functions for core processing that we know exist

func TestApplyConfigDefaults(t *testing.T) {
	p := &Pipeline{}

	tests := []struct {
		name     string
		input    ParallelConfig
		expected int
	}{
		{
			name:     "Zero workers",
			input:    ParallelConfig{MaxWorkers: 0},
			expected: runtime.NumCPU(),
		},
		{
			name:     "Negative workers",
			input:    ParallelConfig{MaxWorkers: -5},
			expected: runtime.NumCPU(),
		},
		{
			name:     "Valid workers",
			input:    ParallelConfig{MaxWorkers: 8},
			expected: 8,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := p.applyConfigDefaults(tt.input)
			assert.Equal(t, tt.expected, result.MaxWorkers)
		})
	}
}

func TestValidateParallelProcessing(t *testing.T) {
	tests := []struct {
		name      string
		pipeline  *Pipeline
		images    []image.Image
		wantError bool
	}{
		{
			name:      "Empty images",
			pipeline:  &Pipeline{},
			images:    []image.Image{},
			wantError: true,
		},
		{
			name:      "Nil pipeline",
			pipeline:  nil,
			images:    []image.Image{testutil.CreateTestImage(100, 100, color.White)},
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.pipeline.validateParallelProcessing(tt.images)
			if tt.wantError {
				require.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCreateChannels(t *testing.T) {
	p := &Pipeline{}
	imageCount := 5

	jobs, results := p.createChannels(imageCount)

	assert.NotNil(t, jobs)
	assert.NotNil(t, results)
	assert.Equal(t, imageCount, cap(jobs))
	assert.Equal(t, imageCount, cap(results))
}

func TestSendJobs(t *testing.T) {
	p := &Pipeline{}
	images := []image.Image{
		testutil.CreateTestImage(100, 100, color.White),
		testutil.CreateTestImage(100, 100, color.Black),
	}
	jobs := make(chan imageJob, 2)

	ctx := context.Background()
	p.sendJobs(ctx, jobs, images)

	// Read all jobs
	receivedJobs := make([]imageJob, 0, len(images))
	for job := range jobs {
		receivedJobs = append(receivedJobs, job)
	}

	assert.Len(t, receivedJobs, 2)
	for i, job := range receivedJobs {
		assert.Equal(t, i, job.index)
		assert.NotNil(t, job.image)
	}
}

func TestSendJobsContextCancellation(t *testing.T) {
	p := &Pipeline{}
	images := []image.Image{
		testutil.CreateTestImage(100, 100, color.White),
		testutil.CreateTestImage(100, 100, color.Black),
	}
	jobs := make(chan imageJob, 1) // Small buffer to force blocking

	ctx, cancel := context.WithCancel(context.Background())

	// Start sending jobs in goroutine
	go p.sendJobs(ctx, jobs, images)

	// Read one job, then cancel
	<-jobs
	cancel()

	// Give time for cancellation to take effect
	time.Sleep(10 * time.Millisecond)

	// Channel should be closed
	select {
	case _, ok := <-jobs:
		assert.False(t, ok, "Channel should be closed")
	case <-time.After(100 * time.Millisecond):
		t.Error("Channel should have been closed")
	}
}

func TestProcessImagesContextEmptyInput(t *testing.T) {
	p := &Pipeline{}

	// Test empty input
	ctx := context.Background()
	results, err := p.ProcessImagesContext(ctx, []image.Image{})

	require.Error(t, err)
	assert.Contains(t, err.Error(), "no images provided")
	assert.Nil(t, results)
}

func TestProcessImagesContextCancellation(t *testing.T) {
	p := &Pipeline{}
	images := []image.Image{
		testutil.CreateTestImage(100, 100, color.White),
	}

	// Create cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	results, err := p.ProcessImagesContext(ctx, images)

	require.Error(t, err)
	require.ErrorIs(t, err, context.Canceled)
	assert.Nil(t, results)
}

func TestApplyOrientationDetectionNilOrienter(t *testing.T) {
	p := &Pipeline{
		cfg:      DefaultConfig(),
		Orienter: nil,
	}
	// Enable orientation to trigger the check
	p.cfg.Orientation.Enabled = true

	img := testutil.CreateTestImage(100, 100, color.White)
	ctx := context.Background()

	resultImg, angle, conf, err := p.applyOrientationDetection(ctx, img)

	require.NoError(t, err)
	assert.Equal(t, img, resultImg)
	assert.Equal(t, 0, angle)
	assert.InEpsilon(t, 0.0, conf, 1e-6)
}

func TestApplyOrientationDetectionContextCancelled(t *testing.T) {
	p := &Pipeline{
		cfg:      DefaultConfig(),
		Orienter: nil,
	}
	p.cfg.Orientation.Enabled = true

	img := testutil.CreateTestImage(100, 100, color.White)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	resultImg, angle, conf, err := p.applyOrientationDetection(ctx, img)

	require.Error(t, err)
	require.ErrorIs(t, err, context.Canceled)
	assert.Nil(t, resultImg)
	assert.Equal(t, 0, angle)
	assert.InEpsilon(t, 0.0, conf, 1e-6)
}

func TestApplyRectificationNilRectifier(t *testing.T) {
	p := &Pipeline{
		cfg:       DefaultConfig(),
		Rectifier: nil,
	}
	p.cfg.Rectification.Enabled = true

	img := testutil.CreateTestImage(100, 100, color.White)
	ctx := context.Background()

	resultImg, err := p.applyRectification(ctx, img)

	require.NoError(t, err)
	assert.Equal(t, img, resultImg)
}

func TestApplyRectificationContextCancelled(t *testing.T) {
	p := &Pipeline{
		cfg:       DefaultConfig(),
		Rectifier: nil,
	}
	p.cfg.Rectification.Enabled = true

	img := testutil.CreateTestImage(100, 100, color.White)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	resultImg, err := p.applyRectification(ctx, img)

	require.Error(t, err)
	require.ErrorIs(t, err, context.Canceled)
	assert.Nil(t, resultImg)
}

func TestBuildOrderedResults(t *testing.T) {
	p := &Pipeline{}
	resultMap := map[int]*OCRImageResult{
		0: {Processing: struct {
			DetectionNs   int64 `json:"detection_ns"`
			RecognitionNs int64 `json:"recognition_ns"`
			TotalNs       int64 `json:"total_ns"`
		}{TotalNs: 100}},
		2: {Processing: struct {
			DetectionNs   int64 `json:"detection_ns"`
			RecognitionNs int64 `json:"recognition_ns"`
			TotalNs       int64 `json:"total_ns"`
		}{TotalNs: 200}},
		// Missing index 1
	}
	errorMap := map[int]error{
		1: errors.New("test error"),
	}
	images := []image.Image{
		testutil.CreateTestImage(100, 100, color.White),
		testutil.CreateTestImage(100, 100, color.Black),
		testutil.CreateTestImage(100, 100, color.White),
	}
	config := DefaultParallelConfig()

	ordered, err := p.buildOrderedResults(resultMap, errorMap, images, config)

	require.Error(t, err) // Should have error from missing result
	require.Len(t, ordered, 3)
	assert.NotNil(t, ordered[0])
	assert.Nil(t, ordered[1]) // Missing result
	assert.NotNil(t, ordered[2])
	assert.Equal(t, int64(100), ordered[0].Processing.TotalNs)
	assert.Equal(t, int64(200), ordered[2].Processing.TotalNs)
}

func TestCollectAndOrderResults(t *testing.T) {
	p := &Pipeline{}
	images := []image.Image{
		testutil.CreateTestImage(100, 100, color.White),
		testutil.CreateTestImage(100, 100, color.Black),
	}
	config := DefaultParallelConfig()

	// Create results channel with unordered results
	results := make(chan imageResult, 2)
	results <- imageResult{
		index: 1,
		result: &OCRImageResult{Processing: struct {
			DetectionNs   int64 `json:"detection_ns"`
			RecognitionNs int64 `json:"recognition_ns"`
			TotalNs       int64 `json:"total_ns"`
		}{TotalNs: 200}},
	}
	results <- imageResult{
		index: 0,
		result: &OCRImageResult{Processing: struct {
			DetectionNs   int64 `json:"detection_ns"`
			RecognitionNs int64 `json:"recognition_ns"`
			TotalNs       int64 `json:"total_ns"`
		}{TotalNs: 100}},
	}
	close(results)

	ctx := context.Background()
	orderedResults, err := p.collectAndOrderResults(ctx, results, images, config)

	require.NoError(t, err)
	assert.Len(t, orderedResults, 2)

	// Verify correct ordering
	assert.Equal(t, int64(100), orderedResults[0].Processing.TotalNs)
	assert.Equal(t, int64(200), orderedResults[1].Processing.TotalNs)
}
