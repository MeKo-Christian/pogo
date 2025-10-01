package detector

import (
	"errors"
	"fmt"
	"image"
	"runtime"
	"time"

	"github.com/MeKo-Tech/pogo/internal/mempool"
	"github.com/MeKo-Tech/pogo/internal/onnx"
	"github.com/yalue/onnxruntime_go"
)

// BatchDetectionResult holds results from batch detection inference.
type BatchDetectionResult struct {
	Results       []*DetectionResult // Individual results for each image
	TotalTime     int64              // Total processing time in nanoseconds
	ThroughputIPS float64            // Images per second
	MemoryUsageMB float64            // Peak memory usage in MB
}

// preprocessBatchImages preprocesses a batch of images and returns tensors and result placeholders.
func (d *Detector) preprocessBatchImages(images []image.Image) ([][]float32, []*DetectionResult, int, int, error) {
	tensors := make([][]float32, 0, len(images))
	results := make([]*DetectionResult, 0, len(images))

	var commonHeight, commonWidth int
	for i, img := range images {
		if img == nil {
			return nil, nil, 0, 0, fmt.Errorf("image at index %d is nil", i)
		}

		// Store original dimensions
		bounds := img.Bounds()
		originalWidth := bounds.Dx()
		originalHeight := bounds.Dy()

		// Preprocess image
		tensor, err := d.preprocessImage(img)
		if err != nil {
			return nil, nil, 0, 0, fmt.Errorf("preprocessing failed for image %d: %w", i, err)
		}

		// Verify all images have same dimensions after preprocessing
		_, _, h, w := tensor.Shape[0], tensor.Shape[1], tensor.Shape[2], tensor.Shape[3]
		if i == 0 {
			commonHeight = int(h)
			commonWidth = int(w)
		} else if int(h) != commonHeight || int(w) != commonWidth {
			return nil, nil, 0, 0, fmt.Errorf("image %d has different dimensions after preprocessing: got %dx%d, expected %dx%d",
				i, w, h, commonWidth, commonHeight)
		}

		tensors = append(tensors, tensor.Data)

		// Create a result placeholder with original dimensions
		result := &DetectionResult{
			OriginalWidth:  originalWidth,
			OriginalHeight: originalHeight,
		}
		results = append(results, result)
	}

	return tensors, results, commonHeight, commonWidth, nil
}

// runBatchInferenceCore performs batch inference and returns output data and dimensions.
func (d *Detector) runBatchInferenceCore(batchTensor onnx.Tensor) ([]float32, int, int, int, int, error) {
	d.mu.RLock()
	session := d.session
	d.mu.RUnlock()

	if session == nil {
		return nil, 0, 0, 0, 0, errors.New("detector session is nil")
	}

	// Create input tensor for ONNX Runtime
	inputTensor, err := onnxruntime_go.NewTensor(onnxruntime_go.NewShape(batchTensor.Shape...), batchTensor.Data)
	if err != nil {
		return nil, 0, 0, 0, 0, fmt.Errorf("failed to create batch input tensor: %w", err)
	}
	defer func() {
		if err := inputTensor.Destroy(); err != nil {
			fmt.Printf("Failed to destroy batch input tensor: %v", err)
		}
	}()

	// Run batch inference
	outputs := []onnxruntime_go.Value{nil}
	err = session.Run([]onnxruntime_go.Value{inputTensor}, outputs)
	if err != nil {
		return nil, 0, 0, 0, 0, fmt.Errorf("batch inference failed: %w", err)
	}

	defer func() {
		for _, output := range outputs {
			if err := output.Destroy(); err != nil {
				fmt.Printf("Failed to destroy output tensor: %v", err)
			}
		}
	}()

	if len(outputs) != 1 {
		return nil, 0, 0, 0, 0, fmt.Errorf("expected 1 output, got %d", len(outputs))
	}

	// Extract output data
	outputTensor := outputs[0]
	floatTensor, ok := outputTensor.(*onnxruntime_go.Tensor[float32])
	if !ok {
		return nil, 0, 0, 0, 0, fmt.Errorf("expected float32 tensor, got %T", outputTensor)
	}

	outputData := floatTensor.GetData()
	actualOutputShape := outputTensor.GetShape()

	if len(actualOutputShape) != 4 {
		return nil, 0, 0, 0, 0, fmt.Errorf("expected 4D output tensor, got %dD", len(actualOutputShape))
	}

	batchSize := int(actualOutputShape[0])
	channels := int(actualOutputShape[1])
	outputHeight := int(actualOutputShape[2])
	outputWidth := int(actualOutputShape[3])

	return outputData, batchSize, channels, outputHeight, outputWidth, nil
}

// splitBatchOutput splits batch output data into individual results.
// Uses memory pooling for per-image probability maps.
func splitBatchOutput(outputData []float32, results []*DetectionResult,
	batchSize, channels, outputHeight, outputWidth int,
) {
	elementsPerImage := channels * outputHeight * outputWidth
	for i := range batchSize {
		startIdx := i * elementsPerImage
		endIdx := startIdx + elementsPerImage

		probabilityMap := mempool.GetFloat32(elementsPerImage)
		copy(probabilityMap, outputData[startIdx:endIdx])

		results[i].ProbabilityMap = probabilityMap
		results[i].Width = outputWidth
		results[i].Height = outputHeight
	}
}

// RunBatchInference performs detection inference on multiple images.
func (d *Detector) RunBatchInference(images []image.Image) (*BatchDetectionResult, error) {
	if len(images) == 0 {
		return nil, errors.New("no images provided")
	}

	start := time.Now()
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	// Preprocess all images
	tensors, results, commonHeight, commonWidth, err := d.preprocessBatchImages(images)
	if err != nil {
		return nil, err
	}

	// Create batch tensor
	batchTensor, err := onnx.NewBatchImageTensor(tensors, 3, commonHeight, commonWidth)
	if err != nil {
		return nil, fmt.Errorf("failed to create batch tensor: %w", err)
	}

	// Run batch inference
	outputData, batchSize, channels, outputHeight, outputWidth, err := d.runBatchInferenceCore(batchTensor)
	if err != nil {
		return nil, err
	}

	if batchSize != len(images) {
		return nil, fmt.Errorf("output batch size %d doesn't match input batch size %d", batchSize, len(images))
	}

	// Split batch output back to individual results
	splitBatchOutput(outputData, results, batchSize, channels, outputHeight, outputWidth)

	totalTime := time.Since(start).Nanoseconds()
	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)

	throughputIPS := float64(len(images)) / (float64(totalTime) / 1e9)
	memoryUsageMB := float64(memAfter.Alloc-memBefore.Alloc) / (1024 * 1024)

	avgTimePerImage := totalTime / int64(len(images))
	for _, result := range results {
		result.ProcessingTime = avgTimePerImage
	}

	return &BatchDetectionResult{
		Results:       results,
		TotalTime:     totalTime,
		ThroughputIPS: throughputIPS,
		MemoryUsageMB: memoryUsageMB,
	}, nil
}
