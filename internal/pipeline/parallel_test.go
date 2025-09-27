package pipeline

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"sync"
	"testing"
	"time"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/recognizer"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultParallelConfig(t *testing.T) {
	config := DefaultParallelConfig()

	assert.Positive(t, config.MaxWorkers, "MaxWorkers should be > 0")
	assert.Equal(t, 0, config.BatchSize, "BatchSize should default to 0")
	assert.Equal(t, uint64(0), config.MemoryLimitBytes, "MemoryLimitBytes should default to 0")
	assert.Nil(t, config.ProgressCallback, "ProgressCallback should default to nil")
	assert.Nil(t, config.ErrorHandler, "ErrorHandler should default to nil")
}

func TestProcessImagesParallel_EmptyInput(t *testing.T) {
	p := &Pipeline{}
	config := DefaultParallelConfig()

	results, err := p.ProcessImagesParallel(nil, config)
	assert.Error(t, err)
	assert.Nil(t, results)
	assert.Contains(t, err.Error(), "no images provided")
}

func TestProcessImagesParallel_NilPipeline(t *testing.T) {
	var p *Pipeline
	config := DefaultParallelConfig()
	images := []image.Image{testutil.CreateTestImage(100, 100, color.White)}

	results, err := p.ProcessImagesParallel(images, config)
	assert.Error(t, err)
	assert.Nil(t, results)
	assert.Contains(t, err.Error(), "pipeline not initialized")
}

func TestProcessImagesParallel_SingleWorker(t *testing.T) {
	// Create mock pipeline
	p := createMockPipeline(t)
	defer func() { _ = p.Close() }()

	config := DefaultParallelConfig()
	config.MaxWorkers = 1

	images := []image.Image{
		testutil.CreateTestImage(100, 100, color.White),
		testutil.CreateTestImage(100, 100, color.White),
	}

	results, err := p.ProcessImagesParallel(images, config)
	require.NoError(t, err)
	assert.Len(t, results, len(images))

	for _, result := range results {
		assert.NotNil(t, result)
		assert.Equal(t, 100, result.Width)
		assert.Equal(t, 100, result.Height)
	}
}

func TestProcessImagesParallel_MultipleWorkers(t *testing.T) {
	// Create mock pipeline
	p := createMockPipeline(t)
	defer func() { _ = p.Close() }()

	config := DefaultParallelConfig()
	config.MaxWorkers = 4

	images := []image.Image{
		testutil.CreateTestImage(100, 100, color.White),
		testutil.CreateTestImage(200, 200, color.White),
		testutil.CreateTestImage(150, 150, color.White),
		testutil.CreateTestImage(120, 120, color.White),
	}

	results, err := p.ProcessImagesParallel(images, config)
	require.NoError(t, err)
	assert.Len(t, results, len(images))

	// Results should maintain order
	expectedWidths := []int{100, 200, 150, 120}
	for i, result := range results {
		assert.NotNil(t, result)
		assert.Equal(t, expectedWidths[i], result.Width)
		assert.Equal(t, expectedWidths[i], result.Height)
	}
}

func TestProcessImagesParallel_WithProgressCallback(t *testing.T) {
	// Create mock pipeline
	p := createMockPipeline(t)
	defer func() { _ = p.Close() }()

	// Mock progress callback
	progressCalls := make([]struct{ current, total int }, 0)
	var mu sync.Mutex

	config := DefaultParallelConfig()
	config.MaxWorkers = 2
	config.ProgressCallback = &mockProgressCallback{
		onStart: func(total int) {
			mu.Lock()
			defer mu.Unlock()
			assert.Equal(t, 3, total)
		},
		onProgress: func(current, total int) {
			mu.Lock()
			defer mu.Unlock()
			progressCalls = append(progressCalls, struct{ current, total int }{current, total})
		},
		onComplete: func() {
			mu.Lock()
			defer mu.Unlock()
			// Verify we got progress updates
			assert.NotEmpty(t, progressCalls)
		},
	}

	images := []image.Image{
		testutil.CreateTestImage(100, 100, color.White),
		testutil.CreateTestImage(100, 100, color.White),
		testutil.CreateTestImage(100, 100, color.White),
	}

	results, err := p.ProcessImagesParallel(images, config)
	require.NoError(t, err)
	assert.Len(t, results, len(images))

	mu.Lock()
	defer mu.Unlock()
	// Should have received at least one progress update
	assert.NotEmpty(t, progressCalls)
	// Final progress should be complete
	if len(progressCalls) > 0 {
		final := progressCalls[len(progressCalls)-1]
		assert.Equal(t, 3, final.current)
		assert.Equal(t, 3, final.total)
	}
}

func TestProcessImagesParallel_WithContextCancellation(t *testing.T) {
	// Create mock pipeline with delay
	p := createMockPipelineWithDelay(t, 100*time.Millisecond)
	defer p.Close()

	config := DefaultParallelConfig()
	config.MaxWorkers = 2

	images := []image.Image{
		testutil.CreateTestImage(100, 100, color.White),
		testutil.CreateTestImage(100, 100, color.White),
		testutil.CreateTestImage(100, 100, color.White),
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	results, err := p.ProcessImagesParallelContext(ctx, images, config)
	assert.Error(t, err)
	assert.Nil(t, results)
	assert.Contains(t, err.Error(), "context deadline exceeded")
}

func TestProcessImagesParallel_WithErrorHandler(t *testing.T) {
	// Create a special mock that can fail based on image dimensions rather than call count
	p := &TestPipeline{
		cfg:        Config{},
		detector:   &mockDetector{},
		recognizer: &mockRecognizerWithImageError{},
	}
	defer p.Close()

	errorCalls := make([]struct {
		index int
		img   image.Image
		err   error
	}, 0)
	var mu sync.Mutex

	config := DefaultParallelConfig()
	config.MaxWorkers = 2
	config.ErrorHandler = func(index int, img image.Image, err error) {
		mu.Lock()
		defer mu.Unlock()
		errorCalls = append(errorCalls, struct {
			index int
			img   image.Image
			err   error
		}{index, img, err})
	}

	images := []image.Image{
		testutil.CreateTestImage(100, 100, color.White),
		testutil.CreateTestImage(200, 200, color.White), // This will fail (200x200 triggers error)
		testutil.CreateTestImage(150, 150, color.White),
	}

	results, err := p.ProcessImagesParallel(images, config)
	assert.Error(t, err) // Should return first error
	assert.NotNil(t, results)

	mu.Lock()
	defer mu.Unlock()
	// Should have called error handler for the 200x200 image
	assert.Len(t, errorCalls, 1)
	assert.Equal(t, 1, errorCalls[0].index) // Image at index 1 (200x200)
	assert.Contains(t, errorCalls[0].err.Error(), "image dimensions are 200x200")
}

func TestProcessImagesParallelBatched(t *testing.T) {
	// Create mock pipeline
	p := createMockPipeline(t)
	defer p.Close()

	config := DefaultParallelConfig()
	config.MaxWorkers = 2
	config.BatchSize = 2

	images := []image.Image{
		testutil.CreateTestImage(100, 100, color.White),
		testutil.CreateTestImage(200, 200, color.White),
		testutil.CreateTestImage(150, 150, color.White),
		testutil.CreateTestImage(120, 120, color.White),
		testutil.CreateTestImage(180, 180, color.White),
	}

	results, err := p.ProcessImagesParallelBatched(images, config)
	require.NoError(t, err)
	assert.Len(t, results, len(images))

	// Results should maintain order
	expectedWidths := []int{100, 200, 150, 120, 180}
	for i, result := range results {
		assert.NotNil(t, result)
		assert.Equal(t, expectedWidths[i], result.Width)
		assert.Equal(t, expectedWidths[i], result.Height)
	}
}

func TestCalculateParallelStats(t *testing.T) {
	images := []image.Image{
		testutil.CreateTestImage(100, 100, color.White),
		testutil.CreateTestImage(100, 100, color.White),
		testutil.CreateTestImage(100, 100, color.White),
	}

	results := []*OCRImageResult{
		{Width: 100, Height: 100},
		nil, // Failed
		{Width: 100, Height: 100},
	}

	duration := 2 * time.Second
	workerCount := 4

	stats := CalculateParallelStats(images, results, duration, workerCount)

	assert.Equal(t, 3, stats.TotalImages)
	assert.Equal(t, 2, stats.ProcessedImages)
	assert.Equal(t, 1, stats.FailedImages)
	assert.Equal(t, 4, stats.WorkerCount)
	assert.Equal(t, duration, stats.TotalDuration)
	assert.Equal(t, time.Second, stats.AveragePerImage)  // 2s / 2 processed
	assert.InDelta(t, 1.0, stats.ThroughputPerSec, 0.01) // 2 images / 2s
}

// Mock progress callback for testing.
type mockProgressCallback struct {
	onStart    func(total int)
	onProgress func(current, total int)
	onComplete func()
	onError    func(current int, err error)
}

func (m *mockProgressCallback) OnStart(total int) {
	if m.onStart != nil {
		m.onStart(total)
	}
}

func (m *mockProgressCallback) OnProgress(current, total int) {
	if m.onProgress != nil {
		m.onProgress(current, total)
	}
}

func (m *mockProgressCallback) OnComplete() {
	if m.onComplete != nil {
		m.onComplete()
	}
}

func (m *mockProgressCallback) OnError(current int, err error) {
	if m.onError != nil {
		m.onError(current, err)
	}
}

// TestPipeline is a test-friendly version of Pipeline with injectable dependencies.
type TestPipeline struct {
	cfg             Config
	detector        DetectorInterface
	recognizer      RecognizerInterface
	ResourceManager *ResourceManager
}

// DetectorInterface defines the interface for detectors.
type DetectorInterface interface {
	DetectRegions(img image.Image) ([]detector.DetectedRegion, error)
	Close() error
	GetModelInfo() map[string]interface{}
	Warmup(iterations int) error
}

// RecognizerInterface defines the interface for recognizers.
type RecognizerInterface interface {
	RecognizeBatch(img image.Image, regions []detector.DetectedRegion) ([]recognizer.Result, error)
	Close() error
	GetModelInfo() map[string]interface{}
	Warmup(iterations int) error
	SetTextLineOrienter(orienter interface{})
}

// TestPipeline methods that match Pipeline interface.
func (p *TestPipeline) ProcessImageContext(ctx context.Context, img image.Image) (*OCRImageResult, error) {
	if p == nil || p.detector == nil || p.recognizer == nil {
		return nil, errors.New("pipeline not initialized")
	}
	if img == nil {
		return nil, errors.New("input image is nil")
	}

	// Simplified processing for tests
	regions, err := p.detector.DetectRegions(img)
	if err != nil {
		return nil, err
	}

	results, err := p.recognizer.RecognizeBatch(img, regions)
	if err != nil {
		return nil, err
	}

	// Create result
	b := img.Bounds()
	return &OCRImageResult{
		Width:   b.Dx(),
		Height:  b.Dy(),
		Regions: make([]OCRRegionResult, len(results)),
	}, nil
}

func (p *TestPipeline) ProcessImagesContext(ctx context.Context, images []image.Image) ([]*OCRImageResult, error) {
	if len(images) == 0 {
		return nil, errors.New("no images provided")
	}
	results := make([]*OCRImageResult, len(images))
	for i, img := range images {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		res, err := p.ProcessImageContext(ctx, img)
		if err != nil {
			return nil, err
		}
		results[i] = res
	}
	return results, nil
}

func (p *TestPipeline) ProcessImagesParallel(images []image.Image, config ParallelConfig) ([]*OCRImageResult, error) {
	return p.ProcessImagesParallelContext(context.Background(), images, config)
}

func (p *TestPipeline) ProcessImagesParallelContext(ctx context.Context, images []image.Image, config ParallelConfig) ([]*OCRImageResult, error) {
	if len(images) == 0 {
		return nil, errors.New("no images provided")
	}
	if p == nil || p.detector == nil || p.recognizer == nil {
		return nil, errors.New("pipeline not initialized")
	}

	// Apply defaults
	if config.MaxWorkers <= 0 {
		config.MaxWorkers = 4 // Default for tests
	}

	// For single image or single worker, fall back to sequential processing
	if len(images) == 1 || config.MaxWorkers == 1 {
		return p.ProcessImagesContext(ctx, images)
	}

	// Initialize progress tracking
	if config.ProgressCallback != nil {
		config.ProgressCallback.OnStart(len(images))
		defer config.ProgressCallback.OnComplete()
	}

	// Create worker pool
	jobs := make(chan imageJob, len(images))
	results := make(chan imageResult, len(images))

	// Start workers
	var wg sync.WaitGroup
	for range config.MaxWorkers {
		wg.Add(1)
		go p.testWorker(ctx, jobs, results, &wg, config)
	}

	// Send jobs
	go func() {
		defer close(jobs)
		for i, img := range images {
			select {
			case jobs <- imageJob{index: i, image: img}:
			case <-ctx.Done():
				return
			}
		}
	}()

	// Collect results
	go func() {
		wg.Wait()
		close(results)
	}()

	// Aggregate results in order
	resultMap := make(map[int]*OCRImageResult)
	errorMap := make(map[int]error)
	processedCount := 0

	for result := range results {
		resultMap[result.index] = result.result
		errorMap[result.index] = result.err
		processedCount++

		// Report progress
		if config.ProgressCallback != nil {
			config.ProgressCallback.OnProgress(processedCount, len(images))
		}
	}

	// Check for context cancellation
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Build ordered result slice
	orderedResults := make([]*OCRImageResult, len(images))
	var firstError error

	for i := range images {
		if err := errorMap[i]; err != nil {
			if firstError == nil {
				firstError = fmt.Errorf("image %d: %w", i, err)
			}
			// Call error handler if provided
			if config.ErrorHandler != nil {
				config.ErrorHandler(i, images[i], err)
			}
		} else {
			orderedResults[i] = resultMap[i]
		}
	}

	return orderedResults, firstError
}

func (p *TestPipeline) ProcessImagesParallelBatched(images []image.Image, config ParallelConfig) ([]*OCRImageResult, error) {
	return p.ProcessImagesParallelBatchedContext(context.Background(), images, config)
}

func (p *TestPipeline) ProcessImagesParallelBatchedContext(ctx context.Context, images []image.Image, config ParallelConfig) ([]*OCRImageResult, error) {
	if config.BatchSize <= 1 {
		// No batching requested, use regular parallel processing
		return p.ProcessImagesParallelContext(ctx, images, config)
	}

	if len(images) == 0 {
		return nil, errors.New("no images provided")
	}

	// Initialize progress tracking
	if config.ProgressCallback != nil {
		config.ProgressCallback.OnStart(len(images))
		defer config.ProgressCallback.OnComplete()
	}

	var allResults []*OCRImageResult
	var resultMutex sync.Mutex
	var firstError error
	var errorMutex sync.Mutex

	// Process images in batches
	var wg sync.WaitGroup
	processedImages := 0
	var progressMutex sync.Mutex

	for start := 0; start < len(images); start += config.BatchSize {
		end := start + config.BatchSize
		if end > len(images) {
			end = len(images)
		}

		batch := images[start:end]
		batchStart := start

		wg.Add(1)
		go func(batch []image.Image, offset int) {
			defer wg.Done()

			// Process batch sequentially within this goroutine
			batchResults, err := p.ProcessImagesContext(ctx, batch)

			// Handle results
			resultMutex.Lock()
			if allResults == nil {
				allResults = make([]*OCRImageResult, len(images))
			}
			for i, result := range batchResults {
				allResults[offset+i] = result
			}
			resultMutex.Unlock()

			// Handle errors
			if err != nil {
				errorMutex.Lock()
				if firstError == nil {
					firstError = fmt.Errorf("batch starting at index %d: %w", offset, err)
				}
				errorMutex.Unlock()
			}

			// Update progress
			progressMutex.Lock()
			processedImages += len(batch)
			currentProcessed := processedImages
			progressMutex.Unlock()

			if config.ProgressCallback != nil {
				config.ProgressCallback.OnProgress(currentProcessed, len(images))
			}
		}(batch, batchStart)
	}

	wg.Wait()

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	return allResults, firstError
}

// testWorker processes images from the jobs channel for TestPipeline.
func (p *TestPipeline) testWorker(ctx context.Context, jobs <-chan imageJob, results chan<- imageResult, wg *sync.WaitGroup, config ParallelConfig) {
	defer wg.Done()

	for {
		select {
		case job, ok := <-jobs:
			if !ok {
				return // Channel closed
			}

			// Process image
			result, err := p.ProcessImageContext(ctx, job.image)

			// Send result
			select {
			case results <- imageResult{index: job.index, result: result, err: err}:
			case <-ctx.Done():
				return
			}

		case <-ctx.Done():
			return
		}
	}
}

func (p *TestPipeline) Close() error {
	var firstErr error
	if p.recognizer != nil {
		if err := p.recognizer.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if p.detector != nil {
		if err := p.detector.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// Helper functions for creating mock pipelines

func createMockPipeline(t *testing.T) *TestPipeline {
	t.Helper()
	return &TestPipeline{
		cfg:        Config{},
		detector:   &mockDetector{},
		recognizer: &mockRecognizer{},
	}
}

func createMockPipelineWithDelay(t *testing.T, delay time.Duration) *TestPipeline {
	t.Helper()
	return &TestPipeline{
		cfg:        Config{},
		detector:   &mockDetectorWithDelay{delay: delay},
		recognizer: &mockRecognizer{},
	}
}

func createMockPipelineWithErrors(t *testing.T, errors map[int]error) *TestPipeline {
	return &TestPipeline{
		cfg:        Config{},
		detector:   &mockDetector{},
		recognizer: &mockRecognizerWithErrors{errors: errors, callCount: 0},
	}
}

// Mock detector.
type mockDetector struct{}

func (m *mockDetector) DetectRegions(img image.Image) ([]detector.DetectedRegion, error) {
	// Return empty regions for simplicity
	return []detector.DetectedRegion{}, nil
}

func (m *mockDetector) Close() error                         { return nil }
func (m *mockDetector) GetModelInfo() map[string]interface{} { return map[string]interface{}{} }
func (m *mockDetector) Warmup(iterations int) error          { return nil }

// Mock detector with delay.
type mockDetectorWithDelay struct {
	delay time.Duration
}

func (m *mockDetectorWithDelay) DetectRegions(img image.Image) ([]detector.DetectedRegion, error) {
	time.Sleep(m.delay)
	return []detector.DetectedRegion{}, nil
}

func (m *mockDetectorWithDelay) Close() error { return nil }
func (m *mockDetectorWithDelay) GetModelInfo() map[string]interface{} {
	return map[string]interface{}{}
}
func (m *mockDetectorWithDelay) Warmup(iterations int) error { return nil }

// Mock recognizer.
type mockRecognizer struct{}

func (m *mockRecognizer) RecognizeBatch(img image.Image, regions []detector.DetectedRegion) ([]recognizer.Result, error) {
	// Return empty results for simplicity
	return []recognizer.Result{}, nil
}

func (m *mockRecognizer) Close() error                             { return nil }
func (m *mockRecognizer) GetModelInfo() map[string]interface{}     { return map[string]interface{}{} }
func (m *mockRecognizer) Warmup(iterations int) error              { return nil }
func (m *mockRecognizer) SetTextLineOrienter(orienter interface{}) {}

// Mock recognizer with errors.
type mockRecognizerWithErrors struct {
	errors    map[int]error
	callCount int
	mu        sync.Mutex
}

func (m *mockRecognizerWithErrors) RecognizeBatch(img image.Image, regions []detector.DetectedRegion) ([]recognizer.Result, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if err, exists := m.errors[m.callCount]; exists {
		m.callCount++
		return nil, err
	}
	m.callCount++
	return []recognizer.Result{}, nil
}

func (m *mockRecognizerWithErrors) Close() error { return nil }
func (m *mockRecognizerWithErrors) GetModelInfo() map[string]any {
	return map[string]interface{}{}
}
func (m *mockRecognizerWithErrors) Warmup(iterations int) error      { return nil }
func (m *mockRecognizerWithErrors) SetTextLineOrienter(orienter any) {}

// Mock recognizer that fails based on image properties.
type mockRecognizerWithImageError struct{}

func (m *mockRecognizerWithImageError) RecognizeBatch(img image.Image, regions []detector.DetectedRegion) ([]recognizer.Result, error) {
	// Fail for images with 200x200 dimensions
	bounds := img.Bounds()
	if bounds.Dx() == 200 && bounds.Dy() == 200 {
		return nil, errors.New("processing failed because image dimensions are 200x200")
	}
	return []recognizer.Result{}, nil
}

func (m *mockRecognizerWithImageError) Close() error { return nil }
func (m *mockRecognizerWithImageError) GetModelInfo() map[string]any {
	return map[string]interface{}{}
}
func (m *mockRecognizerWithImageError) Warmup(iterations int) error              { return nil }
func (m *mockRecognizerWithImageError) SetTextLineOrienter(orienter interface{}) {}
