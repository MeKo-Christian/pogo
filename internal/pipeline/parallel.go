package pipeline

import (
	"context"
	"errors"
	"fmt"
	"image"
	"log/slog"
	"runtime"
	"sync"
	"time"
)

// ParallelConfig holds configuration for parallel processing.
type ParallelConfig struct {
	MaxWorkers       int                           // Number of parallel workers (0 = runtime.NumCPU())
	BatchSize        int                           // Images per batch for micro-batching (0 = no batching)
	MemoryLimitBytes uint64                        // Memory limit in bytes (0 = no limit)
	ProgressCallback ProgressCallback              // Optional progress reporting
	ErrorHandler     func(int, image.Image, error) // Optional per-image error handler
}

// DefaultParallelConfig returns sensible defaults for parallel processing.
func DefaultParallelConfig() ParallelConfig {
	return ParallelConfig{
		MaxWorkers:       runtime.NumCPU(),
		BatchSize:        0, // No micro-batching by default
		MemoryLimitBytes: 0, // No memory limit by default
		ProgressCallback: nil,
		ErrorHandler:     nil,
	}
}

// imageJob represents a single image processing job.
type imageJob struct {
	index int
	image image.Image
}

// imageResult represents the result of processing a single image.
type imageResult struct {
	index  int
	result *OCRImageResult
	err    error
}

// ProcessImagesParallel processes multiple images in parallel using a worker pool.
// Returns results in the same order as input images.
func (p *Pipeline) ProcessImagesParallel(images []image.Image, config ParallelConfig) ([]*OCRImageResult, error) {
	return p.ProcessImagesParallelContext(context.Background(), images, config)
}

// ProcessImagesParallelContext processes images in parallel with context cancellation support.
func (p *Pipeline) ProcessImagesParallelContext(ctx context.Context, images []image.Image,
	config ParallelConfig) ([]*OCRImageResult, error) {
	if err := p.validateParallelProcessing(images); err != nil {
		return nil, err
	}

	config = p.applyConfigDefaults(config)

	slog.Debug("Starting parallel image processing",
		"image_count", len(images),
		"max_workers", config.MaxWorkers,
		"batch_size", config.BatchSize,
		"memory_limit_bytes", config.MemoryLimitBytes)

	// For single image or single worker, fall back to sequential processing
	if len(images) == 1 || config.MaxWorkers == 1 {
		slog.Debug("Falling back to sequential processing", "reason",
			map[bool]string{true: "single_image", false: "single_worker"}[len(images) == 1])
		return p.ProcessImagesContext(ctx, images)
	}

	return p.executeParallelProcessing(ctx, images, config)
}

func (p *Pipeline) validateParallelProcessing(images []image.Image) error {
	if len(images) == 0 {
		return errors.New("no images provided")
	}
	if p == nil || p.Detector == nil || p.Recognizer == nil {
		return errors.New("pipeline not initialized")
	}
	return nil
}

func (p *Pipeline) applyConfigDefaults(config ParallelConfig) ParallelConfig {
	if config.MaxWorkers <= 0 {
		config.MaxWorkers = runtime.NumCPU()
	}
	return config
}

func (p *Pipeline) executeParallelProcessing(ctx context.Context, images []image.Image,
	config ParallelConfig) ([]*OCRImageResult, error) {
	// Initialize progress tracking
	if config.ProgressCallback != nil {
		config.ProgressCallback.OnStart(len(images))
		defer config.ProgressCallback.OnComplete()
	}

	jobs, results := p.createChannels(len(images))
	wg := p.startWorkers(ctx, jobs, results, config)

	p.sendJobs(ctx, jobs, images)
	p.waitForCompletion(wg, results)

	return p.collectAndOrderResults(ctx, results, images, config)
}

func (p *Pipeline) createChannels(imageCount int) (chan imageJob, chan imageResult) {
	jobs := make(chan imageJob, imageCount)
	results := make(chan imageResult, imageCount)
	return jobs, results
}

func (p *Pipeline) startWorkers(ctx context.Context, jobs chan imageJob, results chan imageResult,
	config ParallelConfig) *sync.WaitGroup {
	var wg sync.WaitGroup
	slog.Debug("Starting worker pool", "worker_count", config.MaxWorkers)
	for range config.MaxWorkers {
		wg.Add(1)
		go p.worker(ctx, jobs, results, &wg, config)
	}
	return &wg
}

func (p *Pipeline) sendJobs(ctx context.Context, jobs chan imageJob, images []image.Image) {
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
}

func (p *Pipeline) waitForCompletion(wg *sync.WaitGroup, results chan imageResult) {
	go func() {
		wg.Wait()
		close(results)
	}()
}

func (p *Pipeline) collectAndOrderResults(ctx context.Context, results chan imageResult,
	images []image.Image, config ParallelConfig) ([]*OCRImageResult, error) {
	resultMap, errorMap := p.aggregateResults(results, len(images), config)

	// Check for context cancellation
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	return p.buildOrderedResults(resultMap, errorMap, images, config)
}

func (p *Pipeline) aggregateResults(results chan imageResult, totalImages int,
	config ParallelConfig) (map[int]*OCRImageResult, map[int]error) {
	resultMap := make(map[int]*OCRImageResult)
	errorMap := make(map[int]error)
	processedCount := 0

	for result := range results {
		resultMap[result.index] = result.result
		errorMap[result.index] = result.err
		processedCount++

		// Report progress
		if config.ProgressCallback != nil {
			config.ProgressCallback.OnProgress(processedCount, totalImages)
		}
	}

	return resultMap, errorMap
}

func (p *Pipeline) buildOrderedResults(resultMap map[int]*OCRImageResult, errorMap map[int]error,
	images []image.Image, config ParallelConfig) ([]*OCRImageResult, error) {
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

// worker processes images from the jobs channel.
func (p *Pipeline) worker(
	ctx context.Context,
	jobs <-chan imageJob,
	results chan<- imageResult,
	wg *sync.WaitGroup,
	_ ParallelConfig,
) {
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

// ProcessImagesParallelBatched processes images in parallel with micro-batching support.
// This can be more efficient for very large numbers of small images.
func (p *Pipeline) ProcessImagesParallelBatched(
	images []image.Image,
	config ParallelConfig,
) ([]*OCRImageResult, error) {
	return p.ProcessImagesParallelBatchedContext(context.Background(), images, config)
}

// processBatch processes a single batch of images and handles results/errors.
func (p *Pipeline) processBatch(
	ctx context.Context,
	batch []image.Image,
	offset int,
	resultMutex *sync.Mutex,
	errorMutex *sync.Mutex,
	allResults *[]*OCRImageResult,
	firstError *error,
	imagesLen int,
) {
	// Process batch sequentially within this goroutine
	batchResults, err := p.ProcessImagesContext(ctx, batch)

	// Handle results
	resultMutex.Lock()
	if *allResults == nil {
		*allResults = make([]*OCRImageResult, imagesLen)
	}
	for i, result := range batchResults {
		(*allResults)[offset+i] = result
	}
	resultMutex.Unlock()

	// Handle errors
	if err != nil {
		errorMutex.Lock()
		if *firstError == nil {
			*firstError = fmt.Errorf("batch starting at index %d: %w", offset, err)
		}
		errorMutex.Unlock()
	}
}

// updateProgress updates progress tracking in a thread-safe manner.
func updateProgress(
	progressMutex *sync.Mutex,
	processedImages *int,
	batchLen int,
	config ParallelConfig,
	totalImages int,
) {
	progressMutex.Lock()
	*processedImages += batchLen
	currentProcessed := *processedImages
	progressMutex.Unlock()

	if config.ProgressCallback != nil {
		config.ProgressCallback.OnProgress(currentProcessed, totalImages)
	}
}

// ProcessImagesParallelBatchedContext processes images in parallel batches with context support.
//

func (p *Pipeline) ProcessImagesParallelBatchedContext(ctx context.Context, images []image.Image,
	config ParallelConfig) ([]*OCRImageResult, error) {
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

			// Process batch and handle results/errors
			p.processBatch(ctx, batch, offset, &resultMutex, &errorMutex, &allResults, &firstError, len(images))

			// Update progress
			updateProgress(&progressMutex, &processedImages, len(batch), config, len(images))
		}(batch, batchStart)
	}

	wg.Wait()

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	return allResults, firstError
}

// ParallelStats holds statistics about parallel processing performance.
type ParallelStats struct {
	TotalImages      int           `json:"total_images"`
	ProcessedImages  int           `json:"processed_images"`
	FailedImages     int           `json:"failed_images"`
	WorkerCount      int           `json:"worker_count"`
	TotalDuration    time.Duration `json:"total_duration_ns"`
	AveragePerImage  time.Duration `json:"average_per_image_ns"`
	ThroughputPerSec float64       `json:"throughput_per_sec"`
}

// CalculateParallelStats calculates performance statistics for parallel processing.
func CalculateParallelStats(
	images []image.Image,
	results []*OCRImageResult,
	duration time.Duration,
	workerCount int,
) ParallelStats {
	totalImages := len(images)
	processedImages := 0
	failedImages := 0

	for _, result := range results {
		if result != nil {
			processedImages++
		} else {
			failedImages++
		}
	}

	var avgPerImage time.Duration
	var throughput float64

	if processedImages > 0 {
		avgPerImage = duration / time.Duration(processedImages)
		throughput = float64(processedImages) / duration.Seconds()
	}

	return ParallelStats{
		TotalImages:      totalImages,
		ProcessedImages:  processedImages,
		FailedImages:     failedImages,
		WorkerCount:      workerCount,
		TotalDuration:    duration,
		AveragePerImage:  avgPerImage,
		ThroughputPerSec: throughput,
	}
}
