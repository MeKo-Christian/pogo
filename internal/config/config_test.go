package config

import (
	"testing"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/orientation"
	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/MeKo-Tech/pogo/internal/recognizer"
	"github.com/MeKo-Tech/pogo/internal/rectify"
)

const (
	customModelsDir   = "/custom/models"
	testTextlineModel = "/test/textline.onnx"
)

// TestDefaultConfig verifies that DefaultConfig returns expected values.
func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	// Global settings
	if cfg.ModelsDir != models.DefaultModelsDir {
		t.Errorf("Expected models_dir %s, got %s", models.DefaultModelsDir, cfg.ModelsDir)
	}
	if cfg.LogLevel != infoLevel {
		t.Errorf("Expected log_level '%s', got %s", infoLevel, cfg.LogLevel)
	}
	if cfg.Verbose {
		t.Error("Expected verbose to be false")
	}

	// Pipeline defaults
	if cfg.Pipeline.WarmupIterations != 0 {
		t.Errorf("Expected warmup_iterations 0, got %d", cfg.Pipeline.WarmupIterations)
	}

	// Output defaults
	if cfg.Output.Format != "text" {
		t.Errorf("Expected output format 'text', got %s", cfg.Output.Format)
	}
	if cfg.Output.ConfidencePrecision != 2 {
		t.Errorf("Expected confidence_precision 2, got %d", cfg.Output.ConfidencePrecision)
	}

	// Server defaults
	if cfg.Server.Host != "localhost" {
		t.Errorf("Expected server host 'localhost', got %s", cfg.Server.Host)
	}
	if cfg.Server.Port != 8080 {
		t.Errorf("Expected server port 8080, got %d", cfg.Server.Port)
	}

	// Batch defaults
	if cfg.Batch.Workers != 4 {
		t.Errorf("Expected batch workers 4, got %d", cfg.Batch.Workers)
	}

	// Features defaults
	if cfg.Features.OrientationEnabled {
		t.Error("Expected orientation to be disabled by default")
	}
	if cfg.Features.OrientationThreshold != 0.7 {
		t.Errorf("Expected orientation threshold 0.7, got %f", cfg.Features.OrientationThreshold)
	}

	// GPU defaults
	if cfg.GPU.Enabled {
		t.Error("Expected GPU to be disabled by default")
	}
	if cfg.GPU.Device != 0 {
		t.Errorf("Expected GPU device 0, got %d", cfg.GPU.Device)
	}
	if cfg.GPU.MemoryLimit != autoValue {
		t.Errorf("Expected GPU memory limit '%s', got %s", autoValue, cfg.GPU.MemoryLimit)
	}
}

// TestDefaultDetectorConfig verifies detector defaults.
func TestDefaultDetectorConfig(t *testing.T) {
	cfg := defaultDetectorConfig()
	detCfg := detector.DefaultConfig()

	if cfg.DbThresh != detCfg.DbThresh {
		t.Errorf("Expected DbThresh %f, got %f", detCfg.DbThresh, cfg.DbThresh)
	}
	if cfg.DbBoxThresh != detCfg.DbBoxThresh {
		t.Errorf("Expected DbBoxThresh %f, got %f", detCfg.DbBoxThresh, cfg.DbBoxThresh)
	}
	if cfg.PolygonMode != detCfg.PolygonMode {
		t.Errorf("Expected PolygonMode %s, got %s", detCfg.PolygonMode, cfg.PolygonMode)
	}
	if cfg.UseNMS != detCfg.UseNMS {
		t.Errorf("Expected UseNMS %v, got %v", detCfg.UseNMS, cfg.UseNMS)
	}
	if cfg.Morphology.Operation != noneOp {
		t.Errorf("Expected morphology operation '%s', got %s", noneOp, cfg.Morphology.Operation)
	}
	if cfg.AdaptiveThresholds.Method != histogramMethod {
		t.Errorf("Expected adaptive threshold method '%s', got %s", histogramMethod, cfg.AdaptiveThresholds.Method)
	}
}

// TestDefaultRecognizerConfig verifies recognizer defaults.
func TestDefaultRecognizerConfig(t *testing.T) {
	cfg := defaultRecognizerConfig()
	recCfg := recognizer.DefaultConfig()

	if cfg.Language != recCfg.Language {
		t.Errorf("Expected Language %s, got %s", recCfg.Language, cfg.Language)
	}
	if cfg.ImageHeight != recCfg.ImageHeight {
		t.Errorf("Expected ImageHeight %d, got %d", recCfg.ImageHeight, cfg.ImageHeight)
	}
	if cfg.MinConfidence != 0.0 {
		t.Errorf("Expected MinConfidence 0.0, got %f", cfg.MinConfidence)
	}
}

// TestDefaultParallelConfig verifies parallel defaults.
func TestDefaultParallelConfig(t *testing.T) {
	cfg := defaultParallelConfig()
	parallelCfg := pipeline.DefaultParallelConfig()

	if cfg.MaxWorkers != parallelCfg.MaxWorkers {
		t.Errorf("Expected MaxWorkers %d, got %d", parallelCfg.MaxWorkers, cfg.MaxWorkers)
	}
	if cfg.BatchSize != parallelCfg.BatchSize {
		t.Errorf("Expected BatchSize %d, got %d", parallelCfg.BatchSize, cfg.BatchSize)
	}
}

// TestDefaultResourceConfig verifies resource defaults.
func TestDefaultResourceConfig(t *testing.T) {
	cfg := defaultResourceConfig()
	resourceCfg := pipeline.DefaultResourceConfig()

	if cfg.MaxGoroutines != resourceCfg.MaxGoroutines {
		t.Errorf("Expected MaxGoroutines %d, got %d", resourceCfg.MaxGoroutines, cfg.MaxGoroutines)
	}
}

// TestValidateBasicEnums tests log level and output format validation.
func TestValidateBasicEnums(t *testing.T) {
	tests := []struct {
		name      string
		logLevel  string
		format    string
		wantError bool
	}{
		{"valid log level and format", infoLevel, "text", false},
		{"valid debug", debugLevel, "json", false},
		{"valid warn", warnLevel, "csv", false},
		{"valid error", "error", "text", false},
		{"invalid log level", "invalid", "text", true},
		{"invalid format", infoLevel, "xml", true},
		{"empty format is valid", infoLevel, "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.LogLevel = tt.logLevel
			cfg.Output.Format = tt.format

			err := cfg.validateBasicEnums()
			if (err != nil) != tt.wantError {
				t.Errorf("validateBasicEnums() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

// TestValidateThresholds tests threshold validation.
func TestValidateThresholds(t *testing.T) {
	tests := []struct {
		name      string
		setup     func(*Config)
		wantError bool
	}{
		{
			name:      "valid thresholds",
			setup:     func(c *Config) {},
			wantError: false,
		},
		{
			name: "detector db_thresh too high",
			setup: func(c *Config) {
				c.Pipeline.Detector.DbThresh = 1.5
			},
			wantError: true,
		},
		{
			name: "detector db_thresh negative",
			setup: func(c *Config) {
				c.Pipeline.Detector.DbThresh = -0.1
			},
			wantError: true,
		},
		{
			name: "detector box_thresh too high",
			setup: func(c *Config) {
				c.Pipeline.Detector.DbBoxThresh = 1.1
			},
			wantError: true,
		},
		{
			name: "nms threshold invalid",
			setup: func(c *Config) {
				c.Pipeline.Detector.NMSThreshold = -0.5
			},
			wantError: true,
		},
		{
			name: "recognizer min_confidence invalid",
			setup: func(c *Config) {
				c.Pipeline.Recognizer.MinConfidence = 2.0
			},
			wantError: true,
		},
		{
			name: "orientation threshold invalid",
			setup: func(c *Config) {
				c.Features.OrientationThreshold = -0.1
			},
			wantError: true,
		},
		{
			name: "textline threshold invalid",
			setup: func(c *Config) {
				c.Features.TextlineThreshold = 1.1
			},
			wantError: true,
		},
		{
			name: "rectification threshold invalid",
			setup: func(c *Config) {
				c.Features.RectificationThreshold = 1.5
			},
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			tt.setup(&cfg)

			err := cfg.validateThresholds()
			if (err != nil) != tt.wantError {
				t.Errorf("validateThresholds() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

// TestValidatePositiveIntegers tests positive integer validation.
func TestValidatePositiveIntegers(t *testing.T) {
	tests := []struct {
		name      string
		setup     func(*Config)
		wantError bool
	}{
		{
			name:      "valid integers",
			setup:     func(c *Config) {},
			wantError: false,
		},
		{
			name: "server port zero",
			setup: func(c *Config) {
				c.Server.Port = 0
			},
			wantError: true,
		},
		{
			name: "server port negative",
			setup: func(c *Config) {
				c.Server.Port = -1
			},
			wantError: true,
		},
		{
			name: "server port too high",
			setup: func(c *Config) {
				c.Server.Port = 70000
			},
			wantError: true,
		},
		{
			name: "max upload MB zero",
			setup: func(c *Config) {
				c.Server.MaxUploadMB = 0
			},
			wantError: true,
		},
		{
			name: "timeout zero",
			setup: func(c *Config) {
				c.Server.TimeoutSec = 0
			},
			wantError: true,
		},
		{
			name: "parallel workers zero",
			setup: func(c *Config) {
				c.Pipeline.Parallel.MaxWorkers = 0
			},
			wantError: true,
		},
		{
			name: "batch workers negative",
			setup: func(c *Config) {
				c.Batch.Workers = -1
			},
			wantError: true,
		},
		{
			name: "min region size zero",
			setup: func(c *Config) {
				c.Pipeline.Detector.MinRegionSize = 0
			},
			wantError: true,
		},
		{
			name: "max region size zero",
			setup: func(c *Config) {
				c.Pipeline.Detector.MaxRegionSize = 0
			},
			wantError: true,
		},
		{
			name: "max region size less than min",
			setup: func(c *Config) {
				c.Pipeline.Detector.MinRegionSize = 100
				c.Pipeline.Detector.MaxRegionSize = 50
			},
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			tt.setup(&cfg)

			err := cfg.validatePositiveIntegers()
			if (err != nil) != tt.wantError {
				t.Errorf("validatePositiveIntegers() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

// TestValidateEnums tests enum validation.
func TestValidateEnums(t *testing.T) {
	tests := []struct {
		name        string
		polygonMode string
		wantError   bool
	}{
		{"valid minrect", "minrect", false},
		{"valid contour", "contour", false},
		{"invalid mode", "invalid", true},
		{"empty mode", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.Pipeline.Detector.PolygonMode = tt.polygonMode

			err := cfg.validateEnums()
			if (err != nil) != tt.wantError {
				t.Errorf("validateEnums() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

// TestValidateGPU tests GPU validation.
func TestValidateGPU(t *testing.T) {
	tests := []struct {
		name        string
		memoryLimit string
		wantError   bool
	}{
		{"valid auto", autoValue, false},
		{"valid empty", "", false},
		{"valid B only", "1073741824B", false},
		{"invalid GB", "1GB", true},    // This will fail because "1G" is not a valid number
		{"invalid MB", "512MB", true},  // This will fail because "512M" is not a valid number
		{"invalid KB", "1024KB", true}, // This will fail because "1024K" is not a valid number
		{"invalid unit", "1TB", true},
		{"invalid format", "invalid", true},
		{"no number", "GB", true},
		{"no number B", "B", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.GPU.MemoryLimit = tt.memoryLimit

			err := cfg.validateGPU()
			if (err != nil) != tt.wantError {
				t.Errorf("validateGPU() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

// TestValidate tests the complete validation.
func TestValidate(t *testing.T) {
	t.Run("valid config", func(t *testing.T) {
		cfg := DefaultConfig()
		if err := cfg.Validate(); err != nil {
			t.Errorf("Validate() unexpected error: %v", err)
		}
	})

	t.Run("multiple errors", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.LogLevel = "invalid"
		cfg.Server.Port = 0
		cfg.Pipeline.Detector.DbThresh = 2.0

		err := cfg.Validate()
		if err == nil {
			t.Error("Validate() expected error, got nil")
		}
	})
}

// TestToPipelineConfig tests conversion to pipeline config.
func TestToPipelineConfig(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ModelsDir = "/test/models"
	cfg.Features.OrientationEnabled = true
	cfg.Features.TextlineEnabled = true
	cfg.Features.RectificationEnabled = true
	cfg.Pipeline.WarmupIterations = 3

	pipelineCfg := cfg.ToPipelineConfig()

	if pipelineCfg.ModelsDir != "/test/models" {
		t.Errorf("Expected ModelsDir '/test/models', got %s", pipelineCfg.ModelsDir)
	}
	if !pipelineCfg.EnableOrientation {
		t.Error("Expected orientation to be enabled")
	}
	if pipelineCfg.WarmupIterations != 3 {
		t.Errorf("Expected warmup iterations 3, got %d", pipelineCfg.WarmupIterations)
	}
}

// TestToOrientationConfig tests orientation config conversion.
func TestToOrientationConfig(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Features.OrientationEnabled = true
	cfg.Features.OrientationThreshold = 0.8
	cfg.Features.OrientationModelPath = "/test/orientation.onnx"

	orientCfg := cfg.toOrientationConfig()

	if !orientCfg.Enabled {
		t.Error("Expected orientation to be enabled")
	}
	if orientCfg.ConfidenceThreshold != 0.8 {
		t.Errorf("Expected confidence threshold 0.8, got %f", orientCfg.ConfidenceThreshold)
	}
	if orientCfg.ModelPath != "/test/orientation.onnx" {
		t.Errorf("Expected model path '/test/orientation.onnx', got %s", orientCfg.ModelPath)
	}
}

// TestToTextLineOrientationConfig tests text line orientation config conversion.
func TestToTextLineOrientationConfig(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Features.TextlineEnabled = true
	cfg.Features.TextlineThreshold = 0.6
	cfg.Features.TextlineModelPath = testTextlineModel

	textlineCfg := cfg.toTextLineOrientationConfig()

	if !textlineCfg.Enabled {
		t.Error("Expected textline orientation to be enabled")
	}
	if textlineCfg.ConfidenceThreshold != 0.6 {
		t.Errorf("Expected confidence threshold 0.6, got %f", textlineCfg.ConfidenceThreshold)
	}
	if textlineCfg.ModelPath != testTextlineModel {
		t.Errorf("Expected model path 'testTextlineModel', got %s", textlineCfg.ModelPath)
	}
}

// TestToRectificationConfig tests rectification config conversion.
func TestToRectificationConfig(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Features.RectificationEnabled = true
	cfg.Features.RectificationThreshold = 0.5
	cfg.Features.RectificationHeight = 2048
	cfg.Features.RectificationModelPath = "/test/rectify.onnx"
	cfg.Features.RectificationDebugDir = "/test/debug"

	rectCfg := cfg.toRectificationConfig()

	if !rectCfg.Enabled {
		t.Error("Expected rectification to be enabled")
	}
	if rectCfg.MaskThreshold != 0.5 {
		t.Errorf("Expected mask threshold 0.5, got %f", rectCfg.MaskThreshold)
	}
	if rectCfg.OutputHeight != 2048 {
		t.Errorf("Expected output height 2048, got %d", rectCfg.OutputHeight)
	}
	if rectCfg.ModelPath != "/test/rectify.onnx" {
		t.Errorf("Expected model path '/test/rectify.onnx', got %s", rectCfg.ModelPath)
	}
	if rectCfg.DebugDir != "/test/debug" {
		t.Errorf("Expected debug dir '/test/debug', got %s", rectCfg.DebugDir)
	}
}

// TestToDetectorConfig tests detector config conversion.
func TestToDetectorConfig(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Pipeline.Detector.DbThresh = 0.4
	cfg.Pipeline.Detector.ModelPath = "/test/detector.onnx"
	cfg.Pipeline.Detector.UseAdaptiveNMS = true
	cfg.Pipeline.Detector.SizeAwareNMS = true

	detCfg := cfg.toDetectorConfig()

	if detCfg.DbThresh != 0.4 {
		t.Errorf("Expected DbThresh 0.4, got %f", detCfg.DbThresh)
	}
	if detCfg.ModelPath != "/test/detector.onnx" {
		t.Errorf("Expected model path '/test/detector.onnx', got %s", detCfg.ModelPath)
	}
	if !detCfg.UseAdaptiveNMS {
		t.Error("Expected adaptive NMS to be enabled")
	}
	if !detCfg.SizeAwareNMS {
		t.Error("Expected size-aware NMS to be enabled")
	}
}

// TestToRecognizerConfig tests recognizer config conversion.
func TestToRecognizerConfig(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Pipeline.Recognizer.Language = "en"
	cfg.Pipeline.Recognizer.ImageHeight = 48
	cfg.Pipeline.Recognizer.ModelPath = "/test/recognizer.onnx"
	cfg.Pipeline.Recognizer.DictPath = "/test/dict.txt"

	recCfg := cfg.toRecognizerConfig()

	if recCfg.Language != "en" {
		t.Errorf("Expected language 'en', got %s", recCfg.Language)
	}
	if recCfg.ImageHeight != 48 {
		t.Errorf("Expected image height 48, got %d", recCfg.ImageHeight)
	}
	if recCfg.ModelPath != "/test/recognizer.onnx" {
		t.Errorf("Expected model path '/test/recognizer.onnx', got %s", recCfg.ModelPath)
	}
	if recCfg.DictPath != "/test/dict.txt" {
		t.Errorf("Expected dict path '/test/dict.txt', got %s", recCfg.DictPath)
	}
}

// TestToParallelConfig tests parallel config conversion.
func TestToParallelConfig(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Pipeline.Parallel.MaxWorkers = 8
	cfg.Pipeline.Parallel.BatchSize = 16

	parallelCfg := cfg.toParallelConfig()

	if parallelCfg.MaxWorkers != 8 {
		t.Errorf("Expected max workers 8, got %d", parallelCfg.MaxWorkers)
	}
	if parallelCfg.BatchSize != 16 {
		t.Errorf("Expected batch size 16, got %d", parallelCfg.BatchSize)
	}
}

// TestToResourceConfig tests resource config conversion.
func TestToResourceConfig(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Pipeline.Resource.MaxGoroutines = 100

	resourceCfg := cfg.toResourceConfig()

	if resourceCfg.MaxGoroutines != 100 {
		t.Errorf("Expected max goroutines 100, got %d", resourceCfg.MaxGoroutines)
	}
}

// TestContains tests the contains helper.
func TestContains(t *testing.T) {
	slice := []string{"foo", "bar", "baz"}

	if !contains(slice, "foo") {
		t.Error("Expected 'foo' to be in slice")
	}
	if !contains(slice, "bar") {
		t.Error("Expected 'bar' to be in slice")
	}
	if contains(slice, "qux") {
		t.Error("Did not expect 'qux' to be in slice")
	}
	if contains([]string{}, "foo") {
		t.Error("Did not expect 'foo' in empty slice")
	}
}

// TestValidateThreshold tests the threshold validation helper.
func TestValidateThreshold(t *testing.T) {
	tests := []struct {
		name      string
		value     float64
		fieldName string
		wantError bool
	}{
		{"valid 0.0", 0.0, "test", false},
		{"valid 0.5", 0.5, "test", false},
		{"valid 1.0", 1.0, "test", false},
		{"invalid negative", -0.1, "test", true},
		{"invalid too high", 1.1, "test", true},
		{"invalid way too high", 10.0, "test", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateThreshold(tt.value, tt.fieldName)
			if (err != nil) != tt.wantError {
				t.Errorf("validateThreshold(%f) error = %v, wantError %v", tt.value, err, tt.wantError)
			}
		})
	}
}

// TestValidateMemoryLimit tests the memory limit validation helper.
func TestValidateMemoryLimit(t *testing.T) {
	tests := []struct {
		name      string
		limit     string
		wantError bool
	}{
		{"empty string", "", false},
		{"auto", "auto", false},
		{"bytes", "1024B", false},
		{"invalid kilobytes", "512KB", true}, // "512K" is not a valid number
		{"invalid megabytes", "256MB", true}, // "256M" is not a valid number
		{"invalid gigabytes", "2GB", true},   // "2G" is not a valid number
		{"invalid lowercase", "1gb", true},   // "1g" is not a valid number
		{"invalid decimal", "1.5GB", true},   // "1.5G" is not a valid number
		{"invalid unit", "1TB", true},
		{"no unit", "1024", true},
		{"invalid number", "abcGB", true},
		{"just unit", "GB", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateMemoryLimit(tt.limit)
			if (err != nil) != tt.wantError {
				t.Errorf("validateMemoryLimit(%s) error = %v, wantError %v", tt.limit, err, tt.wantError)
			}
		})
	}
}

// TestParseMorphologyConfig tests morphology config parsing.
func TestParseMorphologyConfig(t *testing.T) {
	tests := []struct {
		name       string
		operation  string
		expectedOp detector.MorphologicalOp
		kernelSize int
		iterations int
	}{
		{"dilate", "dilate", detector.MorphDilate, 3, 1},
		{"erode", "erode", detector.MorphErode, 5, 2},
		{"opening", "opening", detector.MorphOpening, 3, 1},
		{"closing", "closing", detector.MorphClosing, 3, 1},
		{"smooth", "smooth", detector.MorphSmooth, 3, 1},
		{"none", "none", detector.MorphNone, 3, 1},
		{"invalid", "invalid", detector.MorphNone, 3, 1},
		{"uppercase", "DILATE", detector.MorphDilate, 3, 1},
		{"mixed case", "Erode", detector.MorphErode, 3, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := MorphologyConfig{
				Operation:  tt.operation,
				KernelSize: tt.kernelSize,
				Iterations: tt.iterations,
			}

			result := parseMorphologyConfig(cfg)

			if result.Operation != tt.expectedOp {
				t.Errorf("Expected operation %v, got %v", tt.expectedOp, result.Operation)
			}
			if result.KernelSize != tt.kernelSize {
				t.Errorf("Expected kernel size %d, got %d", tt.kernelSize, result.KernelSize)
			}
			if result.Iterations != tt.iterations {
				t.Errorf("Expected iterations %d, got %d", tt.iterations, result.Iterations)
			}
		})
	}
}

// TestParseAdaptiveThresholdsConfig tests adaptive threshold config parsing.
func TestParseAdaptiveThresholdsConfig(t *testing.T) {
	tests := []struct {
		name           string
		method         string
		expectedMethod detector.AdaptiveThresholdMethod
		enabled        bool
	}{
		{"otsu", "otsu", detector.AdaptiveMethodOtsu, true},
		{"histogram", "histogram", detector.AdaptiveMethodHistogram, true},
		{"dynamic", "dynamic", detector.AdaptiveMethodDynamic, true},
		{"invalid", "invalid", detector.AdaptiveMethodHistogram, false},
		{"uppercase", "OTSU", detector.AdaptiveMethodOtsu, true},
		{"mixed case", "Histogram", detector.AdaptiveMethodHistogram, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := AdaptiveThresholdsConfig{
				Enabled:      tt.enabled,
				Method:       tt.method,
				MinDbThresh:  0.1,
				MaxDbThresh:  0.8,
				MinBoxThresh: 0.3,
				MaxBoxThresh: 0.9,
			}

			result := parseAdaptiveThresholdsConfig(cfg)

			if result.Method != tt.expectedMethod {
				t.Errorf("Expected method %v, got %v", tt.expectedMethod, result.Method)
			}
			if result.Enabled != tt.enabled {
				t.Errorf("Expected enabled %v, got %v", tt.enabled, result.Enabled)
			}
			if result.MinDbThresh != 0.1 {
				t.Errorf("Expected MinDbThresh 0.1, got %f", result.MinDbThresh)
			}
		})
	}
}

// TestConfigConversionIntegration tests full config conversion to pipeline config.
func TestConfigConversionIntegration(t *testing.T) {
	cfg := DefaultConfig()

	// Modify some settings
	cfg.ModelsDir = customModelsDir
	cfg.Features.OrientationEnabled = true
	cfg.Features.OrientationThreshold = 0.9
	cfg.Features.TextlineEnabled = true
	cfg.Features.RectificationEnabled = true
	cfg.Pipeline.Detector.DbThresh = 0.35
	cfg.Pipeline.Recognizer.Language = "de"
	cfg.Pipeline.Parallel.MaxWorkers = 12
	cfg.Pipeline.WarmupIterations = 5

	// Convert to pipeline config
	pipelineCfg := cfg.ToPipelineConfig()

	// Verify conversion
	if pipelineCfg.ModelsDir != customModelsDir {
		t.Errorf("ModelsDir not converted correctly")
	}
	if !pipelineCfg.EnableOrientation {
		t.Error("EnableOrientation not set correctly")
	}
	if pipelineCfg.Orientation.ConfidenceThreshold != 0.9 {
		t.Error("Orientation threshold not converted correctly")
	}
	if !pipelineCfg.TextLineOrientation.Enabled {
		t.Error("TextLineOrientation not enabled")
	}
	if !pipelineCfg.Rectification.Enabled {
		t.Error("Rectification not enabled")
	}
	if pipelineCfg.Detector.DbThresh != 0.35 {
		t.Error("Detector DbThresh not converted correctly")
	}
	if pipelineCfg.Recognizer.Language != "de" {
		t.Error("Recognizer Language not converted correctly")
	}
	if pipelineCfg.Parallel.MaxWorkers != 12 {
		t.Error("Parallel MaxWorkers not converted correctly")
	}
	if pipelineCfg.WarmupIterations != 5 {
		t.Error("WarmupIterations not converted correctly")
	}
}

// TestToOrientationConfig_AllFields tests all orientation config fields are converted.
func TestToOrientationConfig_AllFields(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Features.OrientationEnabled = true
	cfg.Features.OrientationThreshold = 0.85
	cfg.Features.OrientationModelPath = "/custom/orientation.onnx"
	cfg.ModelsDir = customModelsDir

	orientCfg := cfg.toOrientationConfig()

	if !orientCfg.Enabled {
		t.Error("Expected orientation to be enabled")
	}
	if orientCfg.ConfidenceThreshold != 0.85 {
		t.Errorf("Expected confidence threshold 0.85, got %f", orientCfg.ConfidenceThreshold)
	}
	if orientCfg.ModelPath != "/custom/orientation.onnx" {
		t.Errorf("Expected model path '/custom/orientation.onnx', got %s", orientCfg.ModelPath)
	}
	// Verify defaults are preserved for non-configured fields
	if orientCfg.UseHeuristicFallback != true {
		t.Error("Expected UseHeuristicFallback to be true (default)")
	}
	if orientCfg.SkipSquareImages != true {
		t.Error("Expected SkipSquareImages to be true (default)")
	}
	if orientCfg.SquareThreshold != 1.2 {
		t.Errorf("Expected SquareThreshold 1.2 (default), got %f", orientCfg.SquareThreshold)
	}
}

// TestToOrientationConfig_EmptyModelPath tests orientation config with empty model path.
func TestToOrientationConfig_EmptyModelPath(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Features.OrientationEnabled = true
	cfg.Features.OrientationModelPath = "" // Empty, should use default

	orientCfg := cfg.toOrientationConfig()

	if !orientCfg.Enabled {
		t.Error("Expected orientation to be enabled")
	}
	// Should have the default model path from orientation.DefaultConfig()
	if orientCfg.ModelPath == "" {
		t.Error("Expected non-empty model path (default should be used)")
	}
}

// TestToTextLineOrientationConfig_AllFields tests all textline config fields are converted.
func TestToTextLineOrientationConfig_AllFields(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Features.TextlineEnabled = true
	cfg.Features.TextlineThreshold = 0.55
	cfg.Features.TextlineModelPath = "/custom/textline.onnx"

	textlineCfg := cfg.toTextLineOrientationConfig()

	if !textlineCfg.Enabled {
		t.Error("Expected textline orientation to be enabled")
	}
	if textlineCfg.ConfidenceThreshold != 0.55 {
		t.Errorf("Expected confidence threshold 0.55, got %f", textlineCfg.ConfidenceThreshold)
	}
	if textlineCfg.ModelPath != "/custom/textline.onnx" {
		t.Errorf("Expected model path '/custom/textline.onnx', got %s", textlineCfg.ModelPath)
	}
	// Verify textline-specific defaults
	if textlineCfg.UseHeuristicFallback != true {
		t.Error("Expected UseHeuristicFallback to be true (default)")
	}
}

// TestToOrientationConfig_DisabledState tests disabled orientation config.
func TestToOrientationConfig_DisabledState(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Features.OrientationEnabled = false
	cfg.Features.OrientationThreshold = 0.9

	orientCfg := cfg.toOrientationConfig()

	if orientCfg.Enabled {
		t.Error("Expected orientation to be disabled")
	}
	// Other settings should still be passed through
	if orientCfg.ConfidenceThreshold != 0.9 {
		t.Errorf("Expected confidence threshold 0.9, got %f", orientCfg.ConfidenceThreshold)
	}
}

// TestToTextLineOrientationConfig_DisabledState tests disabled textline config.
func TestToTextLineOrientationConfig_DisabledState(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Features.TextlineEnabled = false
	cfg.Features.TextlineThreshold = 0.65

	textlineCfg := cfg.toTextLineOrientationConfig()

	if textlineCfg.Enabled {
		t.Error("Expected textline orientation to be disabled")
	}
	if textlineCfg.ConfidenceThreshold != 0.65 {
		t.Errorf("Expected confidence threshold 0.65, got %f", textlineCfg.ConfidenceThreshold)
	}
}

// TestOrientationConfigThresholds tests various threshold values.
func TestOrientationConfigThresholds(t *testing.T) {
	tests := []struct {
		name      string
		threshold float64
		wantValid bool
	}{
		{"minimum threshold 0.0", 0.0, true},
		{"low threshold 0.3", 0.3, true},
		{"medium threshold 0.5", 0.5, true},
		{"default threshold 0.7", 0.7, true},
		{"high threshold 0.9", 0.9, true},
		{"maximum threshold 1.0", 1.0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.Features.OrientationThreshold = tt.threshold

			// Validate should pass for valid thresholds
			err := cfg.validateThresholds()
			if tt.wantValid && err != nil {
				t.Errorf("Expected valid threshold %f, got error: %v", tt.threshold, err)
			}

			// Conversion should preserve threshold
			orientCfg := cfg.toOrientationConfig()
			if orientCfg.ConfidenceThreshold != tt.threshold {
				t.Errorf("Expected threshold %f, got %f", tt.threshold, orientCfg.ConfidenceThreshold)
			}
		})
	}
}

// TestTextLineConfigThresholds tests various textline threshold values.
func TestTextLineConfigThresholds(t *testing.T) {
	tests := []struct {
		name      string
		threshold float64
	}{
		{"low threshold 0.4", 0.4},
		{"default threshold 0.6", 0.6},
		{"high threshold 0.8", 0.8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.Features.TextlineThreshold = tt.threshold

			textlineCfg := cfg.toTextLineOrientationConfig()
			if textlineCfg.ConfidenceThreshold != tt.threshold {
				t.Errorf("Expected threshold %f, got %f", tt.threshold, textlineCfg.ConfidenceThreshold)
			}
		})
	}
}

// TestOrientationConfigModelPaths tests custom model path handling.
func TestOrientationConfigModelPaths(t *testing.T) {
	tests := []struct {
		name      string
		modelPath string
	}{
		{"absolute path", "/absolute/path/to/model.onnx"},
		{"relative path", "models/orientation.onnx"},
		{"with spaces", "/path with spaces/model.onnx"},
		{"long path", "/very/long/path/to/models/directory/orientation_model_v2.onnx"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.Features.OrientationModelPath = tt.modelPath

			orientCfg := cfg.toOrientationConfig()
			if orientCfg.ModelPath != tt.modelPath {
				t.Errorf("Expected model path '%s', got '%s'", tt.modelPath, orientCfg.ModelPath)
			}
		})
	}
}

// TestToPipelineConfig_OrientationIntegration tests full pipeline config with orientation.
func TestToPipelineConfig_OrientationIntegration(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Features.OrientationEnabled = true
	cfg.Features.OrientationThreshold = 0.75
	cfg.Features.OrientationModelPath = "/test/orient.onnx"
	cfg.Features.TextlineEnabled = true
	cfg.Features.TextlineThreshold = 0.65
	cfg.Features.TextlineModelPath = testTextlineModel

	pipelineCfg := cfg.ToPipelineConfig()

	// Verify orientation is properly configured in pipeline
	if !pipelineCfg.EnableOrientation {
		t.Error("Expected orientation to be enabled in pipeline config")
	}
	if !pipelineCfg.Orientation.Enabled {
		t.Error("Expected orientation config to be enabled")
	}
	if pipelineCfg.Orientation.ConfidenceThreshold != 0.75 {
		t.Errorf("Expected orientation threshold 0.75, got %f", pipelineCfg.Orientation.ConfidenceThreshold)
	}
	if pipelineCfg.Orientation.ModelPath != "/test/orient.onnx" {
		t.Errorf("Expected orientation model path '/test/orient.onnx', got %s", pipelineCfg.Orientation.ModelPath)
	}

	// Verify textline orientation
	if !pipelineCfg.TextLineOrientation.Enabled {
		t.Error("Expected textline orientation to be enabled")
	}
	if pipelineCfg.TextLineOrientation.ConfidenceThreshold != 0.65 {
		t.Errorf("Expected textline threshold 0.65, got %f", pipelineCfg.TextLineOrientation.ConfidenceThreshold)
	}
	if pipelineCfg.TextLineOrientation.ModelPath != testTextlineModel {
		t.Errorf("Expected textline model path 'testTextlineModel', got %s", pipelineCfg.TextLineOrientation.ModelPath)
	}
}

// TestOrientationConfigConversionDefaults tests defaults are preserved during conversion.
func TestOrientationConfigConversionDefaults(t *testing.T) {
	cfg := DefaultConfig()
	// Only enable orientation, don't set other fields
	cfg.Features.OrientationEnabled = true

	orientCfg := cfg.toOrientationConfig()

	// Check that defaults from orientation.DefaultConfig() are used
	defaultOrient := orientation.DefaultConfig()
	if orientCfg.ConfidenceThreshold != defaultOrient.ConfidenceThreshold {
		t.Errorf("Expected default confidence threshold %f, got %f",
			defaultOrient.ConfidenceThreshold, orientCfg.ConfidenceThreshold)
	}
	if orientCfg.UseHeuristicFallback != defaultOrient.UseHeuristicFallback {
		t.Errorf("Expected default UseHeuristicFallback %v, got %v",
			defaultOrient.UseHeuristicFallback, orientCfg.UseHeuristicFallback)
	}
	if orientCfg.SkipSquareImages != defaultOrient.SkipSquareImages {
		t.Errorf("Expected default SkipSquareImages %v, got %v",
			defaultOrient.SkipSquareImages, orientCfg.SkipSquareImages)
	}
}

// TestOrientationConfig_GPUDefaults tests GPU defaults in orientation config.
func TestOrientationConfig_GPUDefaults(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Features.OrientationEnabled = true

	orientCfg := cfg.toOrientationConfig()

	// Verify GPU config has proper defaults from orientation.DefaultConfig
	if orientCfg.GPU.UseGPU {
		t.Error("Expected GPU to be disabled by default")
	}
	if orientCfg.GPU.DeviceID != 0 {
		t.Errorf("Expected GPU device ID 0, got %d", orientCfg.GPU.DeviceID)
	}
}

// TestTextLineOrientationConfig_GPUDefaults tests GPU defaults in textline config.
func TestTextLineOrientationConfig_GPUDefaults(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Features.TextlineEnabled = true

	textlineCfg := cfg.toTextLineOrientationConfig()

	// Verify GPU config has proper defaults
	if textlineCfg.GPU.UseGPU {
		t.Error("Expected GPU to be disabled by default for textline")
	}
	if textlineCfg.GPU.DeviceID != 0 {
		t.Errorf("Expected GPU device ID 0, got %d", textlineCfg.GPU.DeviceID)
	}
}

// TestConfigIntegration_OrientationWithOtherFeatures tests orientation works with other features.
func TestConfigIntegration_OrientationWithOtherFeatures(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Features.OrientationEnabled = true
	cfg.Features.OrientationThreshold = 0.8
	cfg.Features.TextlineEnabled = true
	cfg.Features.RectificationEnabled = true

	pipelineCfg := cfg.ToPipelineConfig()

	// All features should be enabled
	if !pipelineCfg.EnableOrientation {
		t.Error("Expected orientation to be enabled")
	}
	if !pipelineCfg.Orientation.Enabled {
		t.Error("Expected orientation config to be enabled")
	}
	if !pipelineCfg.TextLineOrientation.Enabled {
		t.Error("Expected textline orientation to be enabled")
	}
	if !pipelineCfg.Rectification.Enabled {
		t.Error("Expected rectification to be enabled")
	}
}

// TestOrientationConfig_EdgeCaseThresholds tests edge case threshold values.
func TestOrientationConfig_EdgeCaseThresholds(t *testing.T) {
	tests := []struct {
		name           string
		orientThresh   float64
		textlineThresh float64
		rectifyThresh  float64
	}{
		{"all minimum", 0.0, 0.0, 0.0},
		{"all maximum", 1.0, 1.0, 1.0},
		{"mixed values", 0.5, 0.75, 0.25},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.Features.OrientationThreshold = tt.orientThresh
			cfg.Features.TextlineThreshold = tt.textlineThresh
			cfg.Features.RectificationThreshold = tt.rectifyThresh

			// Should pass validation
			err := cfg.validateThresholds()
			if err != nil {
				t.Errorf("Expected valid config, got error: %v", err)
			}

			pipelineCfg := cfg.ToPipelineConfig()
			if pipelineCfg.Orientation.ConfidenceThreshold != tt.orientThresh {
				t.Errorf("Expected orientation threshold %f, got %f",
					tt.orientThresh, pipelineCfg.Orientation.ConfidenceThreshold)
			}
			if pipelineCfg.TextLineOrientation.ConfidenceThreshold != tt.textlineThresh {
				t.Errorf("Expected textline threshold %f, got %f",
					tt.textlineThresh, pipelineCfg.TextLineOrientation.ConfidenceThreshold)
			}
		})
	}
}

// TestOrientationConfig_OnlyOrientationEnabled tests when only orientation is enabled.
func TestOrientationConfig_OnlyOrientationEnabled(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Features.OrientationEnabled = true
	cfg.Features.TextlineEnabled = false
	cfg.Features.RectificationEnabled = false

	pipelineCfg := cfg.ToPipelineConfig()

	if !pipelineCfg.EnableOrientation {
		t.Error("Expected orientation to be enabled in pipeline")
	}
	if !pipelineCfg.Orientation.Enabled {
		t.Error("Expected orientation config to be enabled")
	}
	if pipelineCfg.TextLineOrientation.Enabled {
		t.Error("Expected textline orientation to be disabled")
	}
	if pipelineCfg.Rectification.Enabled {
		t.Error("Expected rectification to be disabled")
	}
}

// TestOrientationConfig_OnlyTextlineEnabled tests when only textline is enabled.
func TestOrientationConfig_OnlyTextlineEnabled(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Features.OrientationEnabled = false
	cfg.Features.TextlineEnabled = true
	cfg.Features.RectificationEnabled = false

	pipelineCfg := cfg.ToPipelineConfig()

	if pipelineCfg.EnableOrientation {
		t.Error("Expected orientation to be disabled in pipeline")
	}
	if pipelineCfg.Orientation.Enabled {
		t.Error("Expected orientation config to be disabled")
	}
	if !pipelineCfg.TextLineOrientation.Enabled {
		t.Error("Expected textline orientation to be enabled")
	}
	if pipelineCfg.Rectification.Enabled {
		t.Error("Expected rectification to be disabled")
	}
}

// Ensure we're importing the necessary packages.
var (
	_ = orientation.Config{}
	_ = rectify.Config{}
	_ = detector.Config{}
	_ = recognizer.Config{}
	_ = pipeline.Config{}
)
