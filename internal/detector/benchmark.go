package detector

import (
	"errors"
	"fmt"
	"image"
	"runtime"
	"time"

	"github.com/MeKo-Tech/pogo/internal/common"
)

// getMemStats returns current memory allocation.
func getMemStats() common.MemoryStats {
	return common.GetMemoryStats()
}

// SimpleBenchmarkResult holds basic benchmark results.
type SimpleBenchmarkResult struct {
	Name         string
	Duration     time.Duration
	MemoryBefore common.MemoryStats
	MemoryAfter  common.MemoryStats
	Iterations   int
	Error        error
}

// InferenceMetrics holds detailed performance metrics for detection inference.
type InferenceMetrics struct {
	PreprocessingTime  int64   // Time spent preprocessing images (nanoseconds)
	ModelExecutionTime int64   // Time spent in ONNX Runtime (nanoseconds)
	PostprocessingTime int64   // Time spent on result processing (nanoseconds)
	TotalTime          int64   // Total inference time (nanoseconds)
	ThroughputIPS      float64 // Images per second
	MemoryAllocMB      float64 // Memory allocated during inference (MB)
	TensorSizeMB       float64 // Size of input tensor (MB)
}

// RunInferenceWithMetrics performs detection inference with detailed profiling.
func (d *Detector) RunInferenceWithMetrics(img image.Image) (*DetectionResult, *InferenceMetrics, error) {
	if img == nil {
		return nil, nil, errors.New("input image is nil")
	}

	metrics := &InferenceMetrics{}
	totalTimer := common.NewTimer()
	memBefore := getMemStats()

	// Store original dimensions
	bounds := img.Bounds()
	originalWidth := bounds.Dx()
	originalHeight := bounds.Dy()

	// Preprocessing phase
	preprocessTimer := common.NewTimer()
	tensor, err := d.preprocessImage(img)
	preprocessTime := preprocessTimer.Stop().Nanoseconds()
	metrics.PreprocessingTime = preprocessTime

	if err != nil {
		return nil, metrics, fmt.Errorf("preprocessing failed: %w", err)
	}

	// Calculate tensor size in MB
	tensorSizeMB := float64(len(tensor.Data)*4) / (1024 * 1024) // 4 bytes per float32
	metrics.TensorSizeMB = tensorSizeMB

	// Model execution phase
	modelTimer := common.NewTimer()
	outputData, width, height, err := d.runInferenceInternal(tensor)
	modelTime := modelTimer.Stop().Nanoseconds()
	metrics.ModelExecutionTime = modelTime

	if err != nil {
		return nil, metrics, err
	}

	// Postprocessing phase (minimal in this case)
	postprocessTimer := common.NewTimer()

	result := &DetectionResult{
		ProbabilityMap: outputData,
		Width:          width,
		Height:         height,
		OriginalWidth:  originalWidth,
		OriginalHeight: originalHeight,
		ProcessingTime: modelTime, // Model execution time only
	}

	postprocessTime := postprocessTimer.Stop().Nanoseconds()
	metrics.PostprocessingTime = postprocessTime

	// Calculate final metrics
	totalTime := totalTimer.Stop().Nanoseconds()
	metrics.TotalTime = totalTime
	metrics.ThroughputIPS = 1.0 / (float64(totalTime) / 1e9)

	memAfter := getMemStats()
	metrics.MemoryAllocMB = float64(memAfter.Alloc-memBefore.Alloc) / (1024 * 1024)

	return result, metrics, nil
}

// BenchmarkDetection runs a benchmark with the given number of iterations.
func (d *Detector) BenchmarkDetection(img image.Image, iterations int) (*SimpleBenchmarkResult, error) {
	if img == nil {
		return nil, errors.New("input image is nil")
	}

	// Force garbage collection before measuring
	runtime.GC()
	memBefore := getMemStats()

	timer := common.NewTimer()
	var err error

	for range iterations {
		if e := d.runInferenceBenchmark(img); e != nil {
			err = e
			break
		}
	}

	duration := timer.Stop()
	memAfter := getMemStats()

	return &SimpleBenchmarkResult{
		Name:         "detection_inference",
		Duration:     duration,
		MemoryBefore: memBefore,
		MemoryAfter:  memAfter,
		Iterations:   iterations,
		Error:        err,
	}, nil
}

// runInferenceBenchmark is a helper for benchmark that just runs inference.
func (d *Detector) runInferenceBenchmark(img image.Image) error {
	_, err := d.RunInference(img)
	return err
}
