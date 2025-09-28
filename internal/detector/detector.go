package detector

import (
	"errors"
	"fmt"
	"image"
	"log/slog"
	"os"
	"sync"
	"time"

	"github.com/MeKo-Tech/pogo/internal/onnx"
	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/yalue/onnxruntime_go"
)

// DetectionResult holds the output from detection inference.
type DetectionResult struct {
	ProbabilityMap []float32 // Raw probability map from model
	Width          int       // Width of probability map
	Height         int       // Height of probability map
	OriginalWidth  int       // Original image width
	OriginalHeight int       // Original image height
	ProcessingTime int64     // Inference time in nanoseconds
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

// NewDetector creates a new text detector with the given configuration.
func NewDetector(config Config) (*Detector, error) {
	if err := validateConfig(config); err != nil {
		return nil, err
	}

	if err := validateModelFile(config.ModelPath); err != nil {
		return nil, err
	}

	slog.Debug("Initializing detector",
		"model_path", config.ModelPath,
		"gpu_enabled", config.GPU.UseGPU,
		"max_image_size", config.MaxImageSize,
		"use_nms", config.UseNMS,
		"nms_method", config.NMSMethod)

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

	slog.Debug("Detector initialized successfully")
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

// runInferenceCore performs the ONNX inference and returns the output tensor.
func (d *Detector) runInferenceCore(inputTensor onnxruntime_go.Value) (onnxruntime_go.Value, error) {
	// Run inference - ONNX Runtime will allocate output tensors automatically
	outputs := []onnxruntime_go.Value{nil}
	err := d.session.Run([]onnxruntime_go.Value{inputTensor}, outputs)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}
	if len(outputs) != 1 {
		return nil, fmt.Errorf("expected 1 output, got %d", len(outputs))
	}
	return outputs[0], nil
}

// runInferenceInternal performs the core inference logic and returns output data and dimensions.
func (d *Detector) runInferenceInternal(tensor onnx.Tensor) ([]float32, int, int, error) {
	// Verify tensor shape
	if err := onnx.VerifyImageTensor(tensor); err != nil {
		return nil, 0, 0, fmt.Errorf("invalid tensor: %w", err)
	}

	d.mu.RLock()
	session := d.session
	d.mu.RUnlock()

	if session == nil {
		return nil, 0, 0, errors.New("detector session is nil")
	}

	// Create input tensor for ONNX Runtime
	inputTensor, err := onnxruntime_go.NewTensor(onnxruntime_go.NewShape(tensor.Shape...), tensor.Data)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer func() {
		if err := inputTensor.Destroy(); err != nil {
			fmt.Fprintf(os.Stderr, "Error destroying input tensor: %v\n", err)
		}
	}()

	outputTensor, err := d.runInferenceCore(inputTensor)
	if err != nil {
		return nil, 0, 0, err
	}
	defer func() {
		if err := outputTensor.Destroy(); err != nil {
			fmt.Fprintf(os.Stderr, "Error destroying output tensor: %v\n", err)
		}
	}()

	// Type assert to float32 tensor to access data
	floatTensor, ok := outputTensor.(*onnxruntime_go.Tensor[float32])
	if !ok {
		return nil, 0, 0, fmt.Errorf("expected float32 tensor, got %T", outputTensor)
	}

	outputData := floatTensor.GetData()
	actualOutputShape := outputTensor.GetShape()

	// Validate output shape (should be [N, C, H, W] where C=1 for probability map)
	if len(actualOutputShape) != 4 {
		return nil, 0, 0, fmt.Errorf("expected 4D output tensor, got %dD", len(actualOutputShape))
	}

	width := int(actualOutputShape[3])
	height := int(actualOutputShape[2])

	return outputData, width, height, nil
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

	outputData, width, height, err := d.runInferenceInternal(tensor)
	if err != nil {
		return nil, err
	}

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
