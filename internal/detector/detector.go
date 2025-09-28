package detector

import (
	"errors"
	"fmt"
	"image"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/onnx"
	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/yalue/onnxruntime_go"
)

// Config holds configuration for the text detector.
type Config struct {
	ModelPath      string         // Path to ONNX detection model
	DbThresh       float32        // DB threshold for binary thresholding (default: 0.3)
	DbBoxThresh    float32        // DB box threshold for filtering (default: 0.5)
	MaxImageSize   int            // Maximum image dimension (default: 960)
	UseServerModel bool           // Use server model instead of mobile (default: false)
	NumThreads     int            // Number of CPU threads (default: 0 for auto)
	UseNMS         bool           // Apply NMS on regions
	NMSThreshold   float64        // IoU threshold for NMS
	NMSMethod      string         // "hard" (default), "linear", or "gaussian" for Soft-NMS
	SoftNMSSigma   float64        // Sigma for Gaussian Soft-NMS
	SoftNMSThresh  float64        // Score threshold for Soft-NMS output filtering
	PolygonMode    string         // "minrect" (default) or "contour"
	GPU            onnx.GPUConfig // GPU acceleration configuration
}

// DefaultConfig returns a default detector configuration.
func DefaultConfig() Config {
	return Config{
		ModelPath:      models.GetDetectionModelPath("", false),
		DbThresh:       0.3,
		DbBoxThresh:    0.5,
		MaxImageSize:   960,
		UseServerModel: false,
		NumThreads:     0,
		UseNMS:         true,
		NMSThreshold:   0.3,
		NMSMethod:      "hard",
		SoftNMSSigma:   0.5,
		SoftNMSThresh:  0.1,
		PolygonMode:    "minrect",
		GPU:            onnx.DefaultGPUConfig(),
	}
}

// UpdateModelPath updates the ModelPath based on modelsDir and UseServerModel flag.
func (c *Config) UpdateModelPath(modelsDir string) {
	// Only update ModelPath if not already set (preserves overrides)
	if c.ModelPath == "" {
		c.ModelPath = models.GetDetectionModelPath(modelsDir, c.UseServerModel)
	}
}

// Detector performs text detection using ONNX Runtime.
type Detector struct {
	config           Config
	session          *onnxruntime_go.DynamicAdvancedSession
	inputInfo        onnxruntime_go.InputOutputInfo
	outputInfo       onnxruntime_go.InputOutputInfo
	imageConstraints utils.ImageConstraints
	mu               sync.RWMutex
}

// DetectionResult holds the output from detection inference.
type DetectionResult struct {
	ProbabilityMap []float32 // Raw probability map from model
	Width          int       // Width of probability map
	Height         int       // Height of probability map
	OriginalWidth  int       // Original image width
	OriginalHeight int       // Original image height
	ProcessingTime int64     // Inference time in nanoseconds
}

// validateConfig validates the detector configuration.
func validateConfig(config Config) error {
	if config.ModelPath == "" {
		return errors.New("model path cannot be empty")
	}
	return nil
}

// validateModelFile checks if the model file exists.
func validateModelFile(modelPath string) error {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return fmt.Errorf("model file not found: %s", modelPath)
	}
	return nil
}

// setupONNXEnvironment sets up the ONNX Runtime environment.
func setupONNXEnvironment(useGPU bool) error {
	if err := onnx.SetONNXLibraryPath(useGPU); err != nil {
		return fmt.Errorf("failed to set ONNX Runtime library path: %w", err)
	}

	if !onnxruntime_go.IsInitialized() {
		if err := onnxruntime_go.InitializeEnvironment(); err != nil {
			return fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
		}
	}
	return nil
}

// validateModelInfo gets and validates model input/output information.
func validateModelInfo(modelPath string) (onnxruntime_go.InputOutputInfo, onnxruntime_go.InputOutputInfo, error) {
	inputs, outputs, err := onnxruntime_go.GetInputOutputInfo(modelPath)
	if err != nil {
		return onnxruntime_go.InputOutputInfo{}, onnxruntime_go.InputOutputInfo{}, fmt.Errorf("failed to get model input/output info: %w", err)
	}

	if len(inputs) != 1 {
		return onnxruntime_go.InputOutputInfo{}, onnxruntime_go.InputOutputInfo{}, fmt.Errorf("expected 1 input, got %d", len(inputs))
	}
	if len(outputs) != 1 {
		return onnxruntime_go.InputOutputInfo{}, onnxruntime_go.InputOutputInfo{}, fmt.Errorf("expected 1 output, got %d", len(outputs))
	}

	inputInfo := inputs[0]
	outputInfo := outputs[0]

	if len(inputInfo.Dimensions) != 4 {
		return onnxruntime_go.InputOutputInfo{}, onnxruntime_go.InputOutputInfo{}, fmt.Errorf("expected 4D input tensor, got %dD", len(inputInfo.Dimensions))
	}

	return inputInfo, outputInfo, nil
}

// createSession creates the ONNX session with the given configuration.
func createSession(modelPath string, inputInfo, outputInfo onnxruntime_go.InputOutputInfo, config Config) (*onnxruntime_go.DynamicAdvancedSession, error) {
	sessionOptions, err := onnxruntime_go.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer func() {
		if err := sessionOptions.Destroy(); err != nil {
			fmt.Printf("Failed to destroy session options: %v", err)
		}
	}()

	if err := onnx.ConfigureSessionForGPU(sessionOptions, config.GPU); err != nil {
		return nil, fmt.Errorf("failed to configure GPU: %w", err)
	}

	if config.NumThreads > 0 {
		if err = sessionOptions.SetIntraOpNumThreads(config.NumThreads); err != nil {
			return nil, fmt.Errorf("failed to set thread count: %w", err)
		}
	}

	session, err := onnxruntime_go.NewDynamicAdvancedSession(modelPath,
		[]string{inputInfo.Name}, []string{outputInfo.Name}, sessionOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return session, nil
}

// setupImageConstraints creates image constraints based on config.
func setupImageConstraints(config Config) utils.ImageConstraints {
	return utils.ImageConstraints{
		MaxWidth:  config.MaxImageSize,
		MaxHeight: config.MaxImageSize,
		MinWidth:  32,
		MinHeight: 32,
	}
}

// NewDetector creates a new text detector with the given configuration.
func NewDetector(config Config) (*Detector, error) {
	if err := validateConfig(config); err != nil {
		return nil, err
	}

	if err := validateModelFile(config.ModelPath); err != nil {
		return nil, err
	}

	if err := setupONNXEnvironment(config.GPU.UseGPU); err != nil {
		return nil, err
	}

	inputInfo, outputInfo, err := validateModelInfo(config.ModelPath)
	if err != nil {
		return nil, err
	}

	session, err := createSession(config.ModelPath, inputInfo, outputInfo, config)
	if err != nil {
		return nil, err
	}

	imageConstraints := setupImageConstraints(config)

	detector := &Detector{
		config:           config,
		session:          session,
		inputInfo:        inputInfo,
		outputInfo:       outputInfo,
		imageConstraints: imageConstraints,
	}

	return detector, nil
}

// Close releases resources used by the detector.
func (d *Detector) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.session != nil {
		if err := d.session.Destroy(); err != nil {
			// Log but don't return error since we're in a Close method
			fmt.Printf("Failed to destroy detector session: %v", err)
		}

		d.session = nil
	}

	// Note: We don't call DestroyEnvironment here as it should only be called
	// when the entire application shuts down
	return nil
}

// GetConfig returns a copy of the detector's configuration.
func (d *Detector) GetConfig() Config {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.config
}

// GetInputShape returns the expected input tensor shape.
func (d *Detector) GetInputShape() []int64 {
	d.mu.RLock()
	defer d.mu.RUnlock()
	shape := make([]int64, len(d.inputInfo.Dimensions))
	copy(shape, d.inputInfo.Dimensions)
	return shape
}

// GetOutputShape returns the expected output tensor shape.
func (d *Detector) GetOutputShape() []int64 {
	d.mu.RLock()
	defer d.mu.RUnlock()
	shape := make([]int64, len(d.outputInfo.Dimensions))
	copy(shape, d.outputInfo.Dimensions)
	return shape
}

// preprocessImage prepares an image for detection inference.
func (d *Detector) preprocessImage(img image.Image) (onnx.Tensor, error) {
	// Resize image to fit within constraints while preserving aspect ratio
	resized, err := utils.ResizeImage(img, d.imageConstraints)
	if err != nil {
		return onnx.Tensor{}, fmt.Errorf("failed to resize image: %w", err)
	}

	// Normalize image to float32 tensor in NCHW format
	tensorData, width, height, err := utils.NormalizeImage(resized)
	if err != nil {
		return onnx.Tensor{}, fmt.Errorf("failed to normalize image: %w", err)
	}

	// Create tensor with shape [1, 3, H, W]
	tensor, err := onnx.NewImageTensor(tensorData, 3, height, width)
	if err != nil {
		return onnx.Tensor{}, fmt.Errorf("failed to create tensor: %w", err)
	}

	return tensor, nil
}

// RunInference performs detection inference on a single image.
func (d *Detector) RunInference(img image.Image) (*DetectionResult, error) {
	if img == nil {
		return nil, errors.New("input image is nil")
	}

	start := time.Now()

	// Store original dimensions
	bounds := img.Bounds()
	originalWidth := bounds.Dx()
	originalHeight := bounds.Dy()

	// Preprocess image
	tensor, err := d.preprocessImage(img)
	if err != nil {
		return nil, fmt.Errorf("preprocessing failed: %w", err)
	}

	// Verify tensor shape
	if err := onnx.VerifyImageTensor(tensor); err != nil {
		return nil, fmt.Errorf("invalid tensor: %w", err)
	}

	d.mu.RLock()
	session := d.session
	d.mu.RUnlock()

	if session == nil {
		return nil, errors.New("detector session is nil")
	}

	// Create input tensor for ONNX Runtime
	inputTensor, err := onnxruntime_go.NewTensor(onnxruntime_go.NewShape(tensor.Shape...), tensor.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer func() {
		if err := inputTensor.Destroy(); err != nil {
			fmt.Fprintf(os.Stderr, "Error destroying input tensor: %v\n", err)
		}
	}()

	// Run inference - ONNX Runtime will allocate output tensors automatically
	outputs := []onnxruntime_go.Value{nil}
	err = session.Run([]onnxruntime_go.Value{inputTensor}, outputs)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}
	defer func() {
		for _, output := range outputs {
			if err := output.Destroy(); err != nil {
				fmt.Fprintf(os.Stderr, "Error destroying output tensor: %v\n", err)
			}
		}
	}()

	if len(outputs) != 1 {
		return nil, fmt.Errorf("expected 1 output, got %d", len(outputs))
	}

	// Extract output data
	outputTensor := outputs[0]

	// Type assert to float32 tensor to access data
	floatTensor, ok := outputTensor.(*onnxruntime_go.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("expected float32 tensor, got %T", outputTensor)
	}

	outputData := floatTensor.GetData()
	actualOutputShape := outputTensor.GetShape()

	// Validate output shape (should be [N, C, H, W] where C=1 for probability map)
	if len(actualOutputShape) != 4 {
		return nil, fmt.Errorf("expected 4D output tensor, got %dD", len(actualOutputShape))
	}

	width := int(actualOutputShape[3])
	height := int(actualOutputShape[2])

	processingTime := time.Since(start).Nanoseconds()

	result := &DetectionResult{
		ProbabilityMap: outputData,
		Width:          width,
		Height:         height,
		OriginalWidth:  originalWidth,
		OriginalHeight: originalHeight,
		ProcessingTime: processingTime,
	}

	return result, nil
}

// BatchDetectionResult holds results from batch detection inference.
type BatchDetectionResult struct {
	Results       []*DetectionResult // Individual results for each image
	TotalTime     int64              // Total processing time in nanoseconds
	ThroughputIPS float64            // Images per second
	MemoryUsageMB float64            // Peak memory usage in MB
}

// RunBatchInference performs detection inference on multiple images.
func (d *Detector) RunBatchInference(images []image.Image) (*BatchDetectionResult, error) {
	if len(images) == 0 {
		return nil, errors.New("no images provided")
	}

	start := time.Now()
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	// Preprocess all images to tensors
	tensors := make([][]float32, 0, len(images))
	results := make([]*DetectionResult, 0, len(images))

	// First pass: preprocess all images and collect original dimensions
	var commonHeight, commonWidth int
	for i, img := range images {
		if img == nil {
			return nil, fmt.Errorf("image at index %d is nil", i)
		}

		// Store original dimensions
		bounds := img.Bounds()
		originalWidth := bounds.Dx()
		originalHeight := bounds.Dy()

		// Preprocess image
		tensor, err := d.preprocessImage(img)
		if err != nil {
			return nil, fmt.Errorf("preprocessing failed for image %d: %w", i, err)
		}

		// Verify all images have same dimensions after preprocessing
		_, _, h, w := tensor.Shape[0], tensor.Shape[1], tensor.Shape[2], tensor.Shape[3]
		if i == 0 {
			commonHeight = int(h)
			commonWidth = int(w)
		} else if int(h) != commonHeight || int(w) != commonWidth {
			return nil, fmt.Errorf("image %d has different dimensions after preprocessing: got %dx%d, expected %dx%d",
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

	// Create batch tensor
	batchTensor, err := onnx.NewBatchImageTensor(tensors, 3, commonHeight, commonWidth)
	if err != nil {
		return nil, fmt.Errorf("failed to create batch tensor: %w", err)
	}

	d.mu.RLock()
	session := d.session
	d.mu.RUnlock()

	if session == nil {
		return nil, errors.New("detector session is nil")
	}

	// Create input tensor for ONNX Runtime
	inputTensor, err := onnxruntime_go.NewTensor(onnxruntime_go.NewShape(batchTensor.Shape...), batchTensor.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to create batch input tensor: %w", err)
	}
	defer func() {
		if err := inputTensor.Destroy(); err != nil {
			// Log but don't return error in defer
			fmt.Printf("Failed to destroy batch input tensor: %v", err)
		}
	}()

	// Run batch inference - ONNX Runtime will allocate output tensors automatically
	outputs := []onnxruntime_go.Value{nil}
	err = session.Run([]onnxruntime_go.Value{inputTensor}, outputs)
	if err != nil {
		return nil, fmt.Errorf("batch inference failed: %w", err)
	}

	defer func() {
		for _, output := range outputs {
			if err := output.Destroy(); err != nil {
				// Log but don't return error in defer
				fmt.Printf("Failed to destroy output tensor: %v", err)
			}
		}
	}()

	if len(outputs) != 1 {
		return nil, fmt.Errorf("expected 1 output, got %d", len(outputs))
	}

	// Extract output data
	outputTensor := outputs[0]

	// Type assert to float32 tensor to access data
	floatTensor, ok := outputTensor.(*onnxruntime_go.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("expected float32 tensor, got %T", outputTensor)
	}

	outputData := floatTensor.GetData()
	actualOutputShape := outputTensor.GetShape()

	// Validate output shape (should be [N, C, H, W])
	if len(actualOutputShape) != 4 {
		return nil, fmt.Errorf("expected 4D output tensor, got %dD", len(actualOutputShape))
	}

	batchSize := int(actualOutputShape[0])
	channels := int(actualOutputShape[1])
	outputHeight := int(actualOutputShape[2])
	outputWidth := int(actualOutputShape[3])

	if batchSize != len(images) {
		return nil, fmt.Errorf("output batch size %d doesn't match input batch size %d", batchSize, len(images))
	}

	// Split batch output back to individual results
	elementsPerImage := channels * outputHeight * outputWidth
	for i := range batchSize {
		startIdx := i * elementsPerImage
		endIdx := startIdx + elementsPerImage

		// Extract probability map for this image
		probabilityMap := make([]float32, elementsPerImage)
		copy(probabilityMap, outputData[startIdx:endIdx])

		// Update the result with probability map data
		results[i].ProbabilityMap = probabilityMap
		results[i].Width = outputWidth
		results[i].Height = outputHeight
	}

	totalTime := time.Since(start).Nanoseconds()
	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)

	// Calculate throughput
	throughputIPS := float64(len(images)) / (float64(totalTime) / 1e9)

	// Calculate memory usage in MB
	memoryUsageMB := float64(memAfter.Alloc-memBefore.Alloc) / (1024 * 1024)

	// Set individual processing times (approximation)
	avgTimePerImage := totalTime / int64(len(images))
	for _, result := range results {
		result.ProcessingTime = avgTimePerImage
	}

	batchResult := &BatchDetectionResult{
		Results:       results,
		TotalTime:     totalTime,
		ThroughputIPS: throughputIPS,
		MemoryUsageMB: memoryUsageMB,
	}

	return batchResult, nil
}

// simpleTimer provides basic timing functionality.
type simpleTimer struct {
	start time.Time
}

// newTimer creates a new timer.
func newTimer() *simpleTimer {
	return &simpleTimer{start: time.Now()}
}

// stop returns the elapsed duration.
func (t *simpleTimer) stop() time.Duration {
	return time.Since(t.start)
}

// simpleMemStats holds basic memory statistics.
type simpleMemStats struct {
	AllocBytes uint64
}

// getMemStats returns current memory allocation.
func getMemStats() simpleMemStats {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return simpleMemStats{AllocBytes: m.Alloc}
}

// SimpleBenchmarkResult holds basic benchmark results.
type SimpleBenchmarkResult struct {
	Name         string
	Duration     time.Duration
	MemoryBefore simpleMemStats
	MemoryAfter  simpleMemStats
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
	totalTimer := newTimer()
	memBefore := getMemStats()

	// Store original dimensions
	bounds := img.Bounds()
	originalWidth := bounds.Dx()
	originalHeight := bounds.Dy()

	// Preprocessing phase
	preprocessTimer := newTimer()
	tensor, err := d.preprocessImage(img)
	preprocessTime := preprocessTimer.stop().Nanoseconds()
	metrics.PreprocessingTime = preprocessTime

	if err != nil {
		return nil, metrics, fmt.Errorf("preprocessing failed: %w", err)
	}

	// Calculate tensor size in MB
	tensorSizeMB := float64(len(tensor.Data)*4) / (1024 * 1024) // 4 bytes per float32
	metrics.TensorSizeMB = tensorSizeMB

	// Verify tensor shape
	if err := onnx.VerifyImageTensor(tensor); err != nil {
		return nil, metrics, fmt.Errorf("invalid tensor: %w", err)
	}

	d.mu.RLock()
	session := d.session
	d.mu.RUnlock()

	if session == nil {
		return nil, metrics, errors.New("detector session is nil")
	}

	// Create input tensor for ONNX Runtime
	inputTensor, err := onnxruntime_go.NewTensor(onnxruntime_go.NewShape(tensor.Shape...), tensor.Data)
	if err != nil {
		return nil, metrics, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer func() {
		if err := inputTensor.Destroy(); err != nil {
			fmt.Fprintf(os.Stderr, "Error destroying input tensor: %v\n", err)
		}
	}()

	// Model execution phase
	modelTimer := newTimer()
	outputs := []onnxruntime_go.Value{nil}
	err = session.Run([]onnxruntime_go.Value{inputTensor}, outputs)
	modelTime := modelTimer.stop().Nanoseconds()
	metrics.ModelExecutionTime = modelTime

	if err != nil {
		return nil, metrics, fmt.Errorf("inference failed: %w", err)
	}
	defer func() {
		for _, output := range outputs {
			if err := output.Destroy(); err != nil {
				fmt.Fprintf(os.Stderr, "Error destroying output tensor: %v\n", err)
			}
		}
	}()

	// Postprocessing phase
	postprocessTimer := newTimer()

	if len(outputs) != 1 {
		return nil, metrics, fmt.Errorf("expected 1 output, got %d", len(outputs))
	}

	// Extract output data
	outputTensor := outputs[0]

	// Type assert to float32 tensor to access data
	floatTensor, ok := outputTensor.(*onnxruntime_go.Tensor[float32])
	if !ok {
		return nil, metrics, fmt.Errorf("expected float32 tensor, got %T", outputTensor)
	}

	outputData := floatTensor.GetData()
	actualOutputShape := outputTensor.GetShape()

	// Validate output shape (should be [N, C, H, W] where C=1 for probability map)
	if len(actualOutputShape) != 4 {
		return nil, metrics, fmt.Errorf("expected 4D output tensor, got %dD", len(actualOutputShape))
	}

	width := int(actualOutputShape[3])
	height := int(actualOutputShape[2])

	result := &DetectionResult{
		ProbabilityMap: outputData,
		Width:          width,
		Height:         height,
		OriginalWidth:  originalWidth,
		OriginalHeight: originalHeight,
		ProcessingTime: modelTime, // Model execution time only
	}

	postprocessTime := postprocessTimer.stop().Nanoseconds()
	metrics.PostprocessingTime = postprocessTime

	// Calculate final metrics
	totalTime := totalTimer.stop().Nanoseconds()
	metrics.TotalTime = totalTime
	metrics.ThroughputIPS = 1.0 / (float64(totalTime) / 1e9)

	memAfter := getMemStats()
	metrics.MemoryAllocMB = float64(memAfter.AllocBytes-memBefore.AllocBytes) / (1024 * 1024)

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

	timer := newTimer()
	var err error

	for range iterations {
		if e := d.runInferenceBenchmark(img); e != nil {
			err = e
			break
		}
	}

	duration := timer.stop()
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

// GetModelInfo returns information about the loaded detection model.
func (d *Detector) GetModelInfo() map[string]interface{} {
	d.mu.RLock()
	defer d.mu.RUnlock()

	info := map[string]interface{}{
		"model_path":       d.config.ModelPath,
		"input_name":       d.inputInfo.Name,
		"output_name":      d.outputInfo.Name,
		"input_shape":      d.inputInfo.Dimensions,
		"output_shape":     d.outputInfo.Dimensions,
		"input_data_type":  d.inputInfo.DataType,
		"output_data_type": d.outputInfo.DataType,
		"db_thresh":        d.config.DbThresh,
		"db_box_thresh":    d.config.DbBoxThresh,
		"max_image_size":   d.config.MaxImageSize,
		"use_server_model": d.config.UseServerModel,
		"num_threads":      d.config.NumThreads,
		"gpu": map[string]interface{}{
			"enabled":                d.config.GPU.UseGPU,
			"device_id":              d.config.GPU.DeviceID,
			"memory_limit_bytes":     d.config.GPU.GPUMemLimit,
			"arena_extend_strategy":  d.config.GPU.ArenaExtendStrategy,
			"cudnn_conv_algo_search": d.config.GPU.CUDNNConvAlgoSearch,
			"copy_in_default_stream": d.config.GPU.DoCopyInDefaultStream,
		},
	}

	return info
}

// Warmup runs a number of forward passes with a blank image to reduce first-run latency.
func (d *Detector) Warmup(iterations int) error {
	if iterations <= 0 {
		return nil
	}
	// Derive a reasonable input size from inputInfo if available
	d.mu.RLock()
	in := d.inputInfo
	sess := d.session
	d.mu.RUnlock()
	if sess == nil {
		return errors.New("detector session is nil")
	}
	// Expect [N,C,H,W]
	h, w := 320, 320
	if len(in.Dimensions) == 4 {
		if in.Dimensions[2] > 0 {
			h = int(in.Dimensions[2])
		}
		if in.Dimensions[3] > 0 {
			w = int(in.Dimensions[3])
		}
	}
	// Create a black image of WxH
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	// Preprocess once
	tensor, err := d.preprocessImage(img)
	if err != nil {
		return err
	}
	for range iterations {
		inputTensor, err := onnxruntime_go.NewTensor(onnxruntime_go.NewShape(tensor.Shape...), tensor.Data)
		if err != nil {
			return err
		}
		outputs := []onnxruntime_go.Value{nil}
		runErr := sess.Run([]onnxruntime_go.Value{inputTensor}, outputs)
		if err := inputTensor.Destroy(); err != nil {
			fmt.Fprintf(os.Stderr, "Error destroying input tensor: %v\n", err)
		}
		if runErr == nil {
			for _, o := range outputs {
				if o != nil {
					if err := o.Destroy(); err != nil {
						fmt.Fprintf(os.Stderr, "Error destroying output tensor: %v\n", err)
					}
				}
			}
		} else {
			return runErr
		}
	}
	return nil
}
