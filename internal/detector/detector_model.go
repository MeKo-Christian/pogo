package detector

import (
	"errors"
	"fmt"
	"os"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/onnx"
	"github.com/yalue/onnxruntime_go"
)

const (
	PolygonModeMinRect = "minrect"
	PolygonModeContour = "contour"
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

	// Class-agnostic NMS tuning
	UseAdaptiveNMS     bool    // Enable adaptive NMS thresholds
	AdaptiveNMSScale   float64 // Scale factor for adaptive IoU thresholds (default: 1.0)
	SizeAwareNMS       bool    // Enable size-based NMS tuning
	MinRegionSize      int     // Minimum region size for size-aware NMS (default: 32)
	MaxRegionSize      int     // Maximum region size for size-aware NMS (default: 1024)
	SizeNMSScaleFactor float64 // Scale factor for size-based IoU adjustment (default: 0.1)

	// Morphological operations configuration
	Morphology MorphConfig // Morphological operations on probability map

	// Adaptive threshold configuration
	AdaptiveThresholds AdaptiveThresholdConfig // Adaptive threshold calculation

	// Multi-scale inference configuration
	MultiScale MultiScaleConfig
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
		PolygonMode:    PolygonModeMinRect,
		GPU:            onnx.DefaultGPUConfig(),

		// Class-agnostic NMS tuning defaults
		UseAdaptiveNMS:     false,
		AdaptiveNMSScale:   1.0,
		SizeAwareNMS:       false,
		MinRegionSize:      32,
		MaxRegionSize:      1024,
		SizeNMSScaleFactor: 0.1,

		// Morphological operations defaults
		Morphology: DefaultMorphConfig(),

		// Adaptive threshold defaults
		AdaptiveThresholds: DefaultAdaptiveThresholdConfig(),

		// Multi-scale defaults
		MultiScale: DefaultMultiScaleConfig(),
	}
}

// UpdateModelPath updates the ModelPath based on modelsDir and UseServerModel flag.
func (c *Config) UpdateModelPath(modelsDir string) {
	c.ModelPath = models.GetDetectionModelPath(modelsDir, c.UseServerModel)
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

// validateModelInfo gets and validates model input/output information.
func validateModelInfo(modelPath string) (onnxruntime_go.InputOutputInfo, onnxruntime_go.InputOutputInfo, error) {
	inputs, outputs, err := onnxruntime_go.GetInputOutputInfo(modelPath)
	if err != nil {
		return onnxruntime_go.InputOutputInfo{}, onnxruntime_go.InputOutputInfo{},
			fmt.Errorf("failed to get model input/output info: %w", err)
	}

	if len(inputs) != 1 {
		return onnxruntime_go.InputOutputInfo{}, onnxruntime_go.InputOutputInfo{},
			fmt.Errorf("expected 1 input, got %d", len(inputs))
	}
	if len(outputs) != 1 {
		return onnxruntime_go.InputOutputInfo{}, onnxruntime_go.InputOutputInfo{},
			fmt.Errorf("expected 1 output, got %d", len(outputs))
	}

	inputInfo := inputs[0]
	outputInfo := outputs[0]

	if len(inputInfo.Dimensions) != 4 {
		return onnxruntime_go.InputOutputInfo{}, onnxruntime_go.InputOutputInfo{},
			fmt.Errorf("expected 4D input tensor, got %dD", len(inputInfo.Dimensions))
	}

	return inputInfo, outputInfo, nil
}

// MultiScaleConfig controls optional multi-scale detection.
type MultiScaleConfig struct {
	Enabled   bool      // Enable multi-scale inference
	Scales    []float64 // Relative scales (w.r.t original), e.g. [1.0, 0.75, 0.5]
	MergeIoU  float64   // IoU used for duplicate removal during merge (defaults to NMSThreshold if <=0)
	Adaptive  bool      // Enable adaptive scale generation based on image size
	MaxLevels int       // Maximum number of pyramid levels (including 1.0) when adaptive
	MinSide   int       // Stop generating when min(image side * scale) <= MinSide
	// Memory/merge behavior
	IncrementalMerge bool // Merge results after each scale to bound retained memory
}

// DefaultMultiScaleConfig returns disabled multi-scale with common downscales.
func DefaultMultiScaleConfig() MultiScaleConfig {
	return MultiScaleConfig{
		Enabled:          false,
		Scales:           []float64{1.0, 0.75, 0.5},
		MergeIoU:         0.3,
		Adaptive:         false,
		MaxLevels:        3,
		MinSide:          320,
		IncrementalMerge: true,
	}
}
