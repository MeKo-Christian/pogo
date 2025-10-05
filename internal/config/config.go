package config

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/orientation"
	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/MeKo-Tech/pogo/internal/recognizer"
	"github.com/MeKo-Tech/pogo/internal/rectify"
)

const (
	// Common string constants to avoid repetition.
	autoValue       = "auto"
	infoLevel       = "info"
	debugLevel      = "debug"
	warnLevel       = "warn"
	dilateOp        = "dilate"
	histogramMethod = "histogram"
	noneOp          = "none"
)

// DefaultConfig returns a configuration with sensible defaults.
func DefaultConfig() Config {
	return Config{
		ModelsDir: models.DefaultModelsDir,
		LogLevel:  infoLevel,
		Verbose:   false,
		Pipeline: PipelineConfig{
			Detector:         defaultDetectorConfig(),
			Recognizer:       defaultRecognizerConfig(),
			Parallel:         defaultParallelConfig(),
			Resource:         defaultResourceConfig(),
			WarmupIterations: 0,
		},
		Output: OutputConfig{
			Format:              "text",
			ConfidencePrecision: 2,
			OverlayBoxColor:     "#FF0000",
			OverlayPolyColor:    "#00FF00",
		},
		Server: ServerConfig{
			Host:            "localhost",
			Port:            8080,
			CORSOrigin:      "*",
			MaxUploadMB:     50,
			TimeoutSec:      30,
			ShutdownTimeout: 10,
			OverlayEnabled:  true,
		},
		Batch: BatchConfig{
			Workers:         4,
			ContinueOnError: false,
		},
		Features: FeatureConfig{
			OrientationEnabled:     false,
			OrientationThreshold:   0.7,
			TextlineEnabled:        false,
			TextlineThreshold:      0.6,
			RectificationEnabled:   false,
			RectificationThreshold: 0.5,
			RectificationHeight:    1024,
			// Barcode defaults
			BarcodeEnabled: false,
			BarcodeTypes:   "",
			BarcodeMinSize: 0,
		},
		GPU: GPUConfig{
			Enabled:     false,
			Device:      0,
			MemoryLimit: autoValue,
		},
	}
}

// defaultDetectorConfig returns default detector configuration.
func defaultDetectorConfig() DetectorConfig {
	cfg := detector.DefaultConfig()
	return DetectorConfig{
		DbThresh:     cfg.DbThresh,
		DbBoxThresh:  cfg.DbBoxThresh,
		PolygonMode:  cfg.PolygonMode,
		UseNMS:       cfg.UseNMS,
		NMSThreshold: cfg.NMSThreshold,
		NumThreads:   cfg.NumThreads,
		MaxImageSize: cfg.MaxImageSize,

		// Class-agnostic NMS tuning defaults
		UseAdaptiveNMS:     cfg.UseAdaptiveNMS,
		AdaptiveNMSScale:   cfg.AdaptiveNMSScale,
		SizeAwareNMS:       cfg.SizeAwareNMS,
		MinRegionSize:      cfg.MinRegionSize,
		MaxRegionSize:      cfg.MaxRegionSize,
		SizeNMSScaleFactor: cfg.SizeNMSScaleFactor,

		// Morphological operations defaults
		Morphology: MorphologyConfig{
			Operation:  noneOp,
			KernelSize: cfg.Morphology.KernelSize,
			Iterations: cfg.Morphology.Iterations,
		},

		// Adaptive thresholds defaults
		AdaptiveThresholds: AdaptiveThresholdsConfig{
			Enabled:      cfg.AdaptiveThresholds.Enabled,
			Method:       histogramMethod,
			MinDbThresh:  cfg.AdaptiveThresholds.MinDbThresh,
			MaxDbThresh:  cfg.AdaptiveThresholds.MaxDbThresh,
			MinBoxThresh: cfg.AdaptiveThresholds.MinBoxThresh,
			MaxBoxThresh: cfg.AdaptiveThresholds.MaxBoxThresh,
		},

		// Multi-scale defaults
		MultiScale: MultiScaleConfig{
			Enabled:          cfg.MultiScale.Enabled,
			Scales:           cfg.MultiScale.Scales,
			MergeIoU:         cfg.MultiScale.MergeIoU,
			Adaptive:         cfg.MultiScale.Adaptive,
			MaxLevels:        cfg.MultiScale.MaxLevels,
			MinSide:          cfg.MultiScale.MinSide,
			IncrementalMerge: cfg.MultiScale.IncrementalMerge,
		},
	}
}

// defaultRecognizerConfig returns default recognizer configuration.
func defaultRecognizerConfig() RecognizerConfig {
	cfg := recognizer.DefaultConfig()
	return RecognizerConfig{
		Language:         cfg.Language,
		ImageHeight:      cfg.ImageHeight,
		MaxWidth:         cfg.MaxWidth,
		PadWidthMultiple: cfg.PadWidthMultiple,
		MinConfidence:    0.0,
		NumThreads:       cfg.NumThreads,
	}
}

// defaultParallelConfig returns default parallel configuration.
func defaultParallelConfig() ParallelConfig {
	cfg := pipeline.DefaultParallelConfig()
	return ParallelConfig{
		MaxWorkers: cfg.MaxWorkers,
		BatchSize:  cfg.BatchSize,
	}
}

// defaultResourceConfig returns default resource configuration.
func defaultResourceConfig() ResourceConfig {
	cfg := pipeline.DefaultResourceConfig()
	return ResourceConfig{
		MaxGoroutines: cfg.MaxGoroutines,
	}
}

// validateBasicEnums validates log level and output format.
func (c *Config) validateBasicEnums() error {
	// Validate log level
	validLogLevels := []string{"debug", "info", "warn", "error"}
	if !contains(validLogLevels, c.LogLevel) {
		return fmt.Errorf("invalid log level: %s (must be one of: %s)", c.LogLevel, strings.Join(validLogLevels, ", "))
	}

	// Validate output format
	validFormats := []string{"text", "json", "csv"}
	if c.Output.Format != "" && !contains(validFormats, c.Output.Format) {
		return fmt.Errorf("invalid output format: %s (must be one of: %s)", c.Output.Format, strings.Join(validFormats, ", "))
	}

	return nil
}

// validateThresholds validates all threshold values.
func (c *Config) validateThresholds() error {
	// Validate thresholds (must be between 0.0 and 1.0)
	if err := validateThreshold(float64(c.Pipeline.Detector.DbThresh), "detector.db_thresh"); err != nil {
		return err
	}
	if err := validateThreshold(float64(c.Pipeline.Detector.DbBoxThresh), "detector.db_box_thresh"); err != nil {
		return err
	}
	if err := validateThreshold(c.Pipeline.Detector.NMSThreshold, "detector.nms_threshold"); err != nil {
		return err
	}
	// Multi-scale merge IoU (optional; only validate if >0)
	if c.Pipeline.Detector.MultiScale.MergeIoU > 0 {
		if err := validateThreshold(c.Pipeline.Detector.MultiScale.MergeIoU, "detector.multi_scale.merge_iou"); err != nil {
			return err
		}
	}
	// Validate Multi-scale integers if adaptive enabled
	if c.Pipeline.Detector.MultiScale.MaxLevels < 0 {
		return fmt.Errorf("invalid detector.multi_scale.max_levels: %d (must be >= 0)", c.Pipeline.Detector.MultiScale.MaxLevels)
	}
	if c.Pipeline.Detector.MultiScale.MinSide < 0 {
		return fmt.Errorf("invalid detector.multi_scale.min_side: %d (must be >= 0)", c.Pipeline.Detector.MultiScale.MinSide)
	}
	if err := validateThreshold(c.Pipeline.Detector.AdaptiveNMSScale, "detector.adaptive_nms_scale"); err != nil {
		return err
	}
	if err := validateThreshold(c.Pipeline.Detector.SizeNMSScaleFactor, "detector.size_nms_scale_factor"); err != nil {
		return err
	}
	if err := validateThreshold(c.Pipeline.Recognizer.MinConfidence, "recognizer.min_confidence"); err != nil {
		return err
	}
	if err := validateThreshold(c.Features.OrientationThreshold, "features.orientation_threshold"); err != nil {
		return err
	}
	if err := validateThreshold(c.Features.TextlineThreshold, "features.textline_threshold"); err != nil {
		return err
	}
	if err := validateThreshold(c.Features.RectificationThreshold, "features.rectification_threshold"); err != nil {
		return err
	}

	return nil
}

// validatePositiveIntegers validates all positive integer values.
func (c *Config) validatePositiveIntegers() error {
	if c.Server.Port <= 0 || c.Server.Port > 65535 {
		return fmt.Errorf("invalid server port: %d (must be between 1 and 65535)", c.Server.Port)
	}
	if c.Server.MaxUploadMB <= 0 {
		return fmt.Errorf("invalid max upload size: %d (must be positive)", c.Server.MaxUploadMB)
	}
	if c.Server.TimeoutSec <= 0 {
		return fmt.Errorf("invalid timeout: %d (must be positive)", c.Server.TimeoutSec)
	}
	if c.Pipeline.Parallel.MaxWorkers <= 0 {
		return fmt.Errorf("invalid parallel max workers: %d (must be positive)", c.Pipeline.Parallel.MaxWorkers)
	}
	if c.Batch.Workers <= 0 {
		return fmt.Errorf("invalid batch workers: %d (must be positive)", c.Batch.Workers)
	}
	if c.Pipeline.Detector.MinRegionSize <= 0 {
		return fmt.Errorf("invalid detector min region size: %d (must be positive)", c.Pipeline.Detector.MinRegionSize)
	}
	if c.Pipeline.Detector.MaxRegionSize <= 0 {
		return fmt.Errorf("invalid detector max region size: %d (must be positive)", c.Pipeline.Detector.MaxRegionSize)
	}
	if c.Pipeline.Detector.MaxRegionSize < c.Pipeline.Detector.MinRegionSize {
		return fmt.Errorf("detector max region size (%d) must be >= min region size (%d)",
			c.Pipeline.Detector.MaxRegionSize, c.Pipeline.Detector.MinRegionSize)
	}

	// Barcode size cannot be negative
	if c.Features.BarcodeMinSize < 0 {
		return fmt.Errorf("invalid barcode_min_size: %d (must be >= 0)", c.Features.BarcodeMinSize)
	}

	return nil
}

// validateEnums validates enum-like fields.
func (c *Config) validateEnums() error {
	// Validate polygon mode
	validPolygonModes := []string{"minrect", "contour"}
	if !contains(validPolygonModes, c.Pipeline.Detector.PolygonMode) {
		return fmt.Errorf("invalid polygon mode: %s (must be one of: %s)",
			c.Pipeline.Detector.PolygonMode, strings.Join(validPolygonModes, ", "))
	}

	return nil
}

// validateGPU validates GPU-related settings.
func (c *Config) validateGPU() error {
	// Validate GPU memory limit format
	if c.GPU.MemoryLimit != autoValue && c.GPU.MemoryLimit != "" {
		if err := validateMemoryLimit(c.GPU.MemoryLimit); err != nil {
			return fmt.Errorf("invalid GPU memory limit: %w", err)
		}
	}

	return nil
}

// Validate validates the configuration and returns any errors.
func (c *Config) Validate() error {
	if err := c.validateBasicEnums(); err != nil {
		return err
	}
	if err := c.validateThresholds(); err != nil {
		return err
	}
	if err := c.validatePositiveIntegers(); err != nil {
		return err
	}
	if err := c.validateEnums(); err != nil {
		return err
	}
	if err := c.validateGPU(); err != nil {
		return err
	}

	return nil
}

// ToPipelineConfig converts the config to the internal pipeline configuration format.
func (c *Config) ToPipelineConfig() pipeline.Config {
    return pipeline.Config{
        ModelsDir:           c.ModelsDir,
        EnableOrientation:   c.Features.OrientationEnabled,
        Orientation:         c.toOrientationConfig(),
        TextLineOrientation: c.toTextLineOrientationConfig(),
        Rectification:       c.toRectificationConfig(),
        Detector:            c.toDetectorConfig(),
        Recognizer:          c.toRecognizerConfig(),
        WarmupIterations:    c.Pipeline.WarmupIterations,
        Parallel:            c.toParallelConfig(),
        Resource:            c.toResourceConfig(),
        Barcode:             c.toBarcodeConfig(),
    }
}

// toBarcodeConfig converts feature flags to pipeline.BarcodeConfig.
func (c *Config) toBarcodeConfig() pipeline.BarcodeConfig {
    bc := pipeline.DefaultBarcodeConfig()
    bc.Enabled = c.Features.BarcodeEnabled
    // Parse types as comma-separated
    if strings.TrimSpace(c.Features.BarcodeTypes) != "" {
        parts := strings.Split(c.Features.BarcodeTypes, ",")
        cleaned := make([]string, 0, len(parts))
        for _, p := range parts {
            p = strings.TrimSpace(p)
            if p != "" {
                cleaned = append(cleaned, p)
            }
        }
        bc.Types = cleaned
    }
    if c.Features.BarcodeMinSize > 0 {
        bc.MinSize = c.Features.BarcodeMinSize
    }
    return bc
}

// toOrientationConfig converts to orientation.Config.
func (c *Config) toOrientationConfig() orientation.Config {
	cfg := orientation.DefaultConfig()
	cfg.Enabled = c.Features.OrientationEnabled
	cfg.ConfidenceThreshold = c.Features.OrientationThreshold
	if c.Features.OrientationModelPath != "" {
		cfg.ModelPath = c.Features.OrientationModelPath
	}
	return cfg
}

// toTextLineOrientationConfig converts to text line orientation.Config.
func (c *Config) toTextLineOrientationConfig() orientation.Config {
	cfg := orientation.DefaultTextLineConfig()
	cfg.Enabled = c.Features.TextlineEnabled
	cfg.ConfidenceThreshold = c.Features.TextlineThreshold
	if c.Features.TextlineModelPath != "" {
		cfg.ModelPath = c.Features.TextlineModelPath
	}
	return cfg
}

// toRectificationConfig converts to rectify.Config.
func (c *Config) toRectificationConfig() rectify.Config {
	cfg := rectify.DefaultConfig()
	cfg.Enabled = c.Features.RectificationEnabled
	cfg.MaskThreshold = c.Features.RectificationThreshold
	cfg.OutputHeight = c.Features.RectificationHeight
	if c.Features.RectificationModelPath != "" {
		cfg.ModelPath = c.Features.RectificationModelPath
	}
	if c.Features.RectificationDebugDir != "" {
		cfg.DebugDir = c.Features.RectificationDebugDir
	}
	return cfg
}

// toDetectorConfig converts to detector.Config.
func (c *Config) toDetectorConfig() detector.Config {
	cfg := detector.DefaultConfig()
	cfg.DbThresh = c.Pipeline.Detector.DbThresh
	cfg.DbBoxThresh = c.Pipeline.Detector.DbBoxThresh
	cfg.UseNMS = c.Pipeline.Detector.UseNMS
	cfg.NMSThreshold = c.Pipeline.Detector.NMSThreshold
	cfg.NumThreads = c.Pipeline.Detector.NumThreads
	cfg.MaxImageSize = c.Pipeline.Detector.MaxImageSize
	cfg.PolygonMode = c.Pipeline.Detector.PolygonMode
	if c.Pipeline.Detector.ModelPath != "" {
		cfg.ModelPath = c.Pipeline.Detector.ModelPath
	}

	// Class-agnostic NMS tuning
	cfg.UseAdaptiveNMS = c.Pipeline.Detector.UseAdaptiveNMS
	cfg.AdaptiveNMSScale = c.Pipeline.Detector.AdaptiveNMSScale
	cfg.SizeAwareNMS = c.Pipeline.Detector.SizeAwareNMS
	cfg.MinRegionSize = c.Pipeline.Detector.MinRegionSize
	cfg.MaxRegionSize = c.Pipeline.Detector.MaxRegionSize
	cfg.SizeNMSScaleFactor = c.Pipeline.Detector.SizeNMSScaleFactor

	// Morphological operations
	cfg.Morphology = parseMorphologyConfig(c.Pipeline.Detector.Morphology)

	// Adaptive thresholds
	cfg.AdaptiveThresholds = parseAdaptiveThresholdsConfig(c.Pipeline.Detector.AdaptiveThresholds)

	// Multi-scale
	cfg.MultiScale.Enabled = c.Pipeline.Detector.MultiScale.Enabled
	if len(c.Pipeline.Detector.MultiScale.Scales) > 0 {
		cfg.MultiScale.Scales = c.Pipeline.Detector.MultiScale.Scales
	}
	if c.Pipeline.Detector.MultiScale.MergeIoU > 0 {
		cfg.MultiScale.MergeIoU = c.Pipeline.Detector.MultiScale.MergeIoU
	}
	cfg.MultiScale.Adaptive = c.Pipeline.Detector.MultiScale.Adaptive
	if c.Pipeline.Detector.MultiScale.MaxLevels > 0 {
		cfg.MultiScale.MaxLevels = c.Pipeline.Detector.MultiScale.MaxLevels
	}
	if c.Pipeline.Detector.MultiScale.MinSide > 0 {
		cfg.MultiScale.MinSide = c.Pipeline.Detector.MultiScale.MinSide
	}
	cfg.MultiScale.IncrementalMerge = c.Pipeline.Detector.MultiScale.IncrementalMerge

	return cfg
}

// toRecognizerConfig converts to recognizer.Config.
func (c *Config) toRecognizerConfig() recognizer.Config {
	cfg := recognizer.DefaultConfig()
	cfg.Language = c.Pipeline.Recognizer.Language
	cfg.ImageHeight = c.Pipeline.Recognizer.ImageHeight
	cfg.MaxWidth = c.Pipeline.Recognizer.MaxWidth
	cfg.PadWidthMultiple = c.Pipeline.Recognizer.PadWidthMultiple
	cfg.NumThreads = c.Pipeline.Recognizer.NumThreads
	if c.Pipeline.Recognizer.ModelPath != "" {
		cfg.ModelPath = c.Pipeline.Recognizer.ModelPath
	}
	if c.Pipeline.Recognizer.DictPath != "" {
		cfg.DictPath = c.Pipeline.Recognizer.DictPath
	}
	return cfg
}

// toParallelConfig converts to pipeline.ParallelConfig.
func (c *Config) toParallelConfig() pipeline.ParallelConfig {
	return pipeline.ParallelConfig{
		MaxWorkers: c.Pipeline.Parallel.MaxWorkers,
		BatchSize:  c.Pipeline.Parallel.BatchSize,
	}
}

// toResourceConfig converts to pipeline.ResourceConfig.
func (c *Config) toResourceConfig() pipeline.ResourceConfig {
	return pipeline.ResourceConfig{
		MaxGoroutines: c.Pipeline.Resource.MaxGoroutines,
	}
}

// Helper functions

// contains checks if a slice contains a string.
func contains(slice []string, item string) bool {
	return slices.Contains(slice, item)
}

// validateThreshold validates that a value is between 0.0 and 1.0.
func validateThreshold(value float64, name string) error {
	if value < 0.0 || value > 1.0 {
		return fmt.Errorf("invalid %s: %.2f (must be between 0.0 and 1.0)", name, value)
	}
	return nil
}

// validateMemoryLimit validates GPU memory limit format (e.g., "1GB", "512MB").
func validateMemoryLimit(limit string) error {
	if limit == "" || limit == autoValue {
		return nil
	}

	// Check if it ends with valid unit
	validUnits := []string{"B", "KB", "MB", "GB"}
	hasValidUnit := false
	for _, unit := range validUnits {
		if strings.HasSuffix(strings.ToUpper(limit), unit) {
			hasValidUnit = true
			numStr := strings.TrimSuffix(strings.ToUpper(limit), unit)
			if _, err := strconv.ParseFloat(numStr, 64); err != nil {
				return fmt.Errorf("invalid number in memory limit: %s", limit)
			}
			break
		}
	}

	if !hasValidUnit {
		return fmt.Errorf("memory limit must end with one of: %s", strings.Join(validUnits, ", "))
	}

	return nil
}

// parseMorphologyConfig converts MorphologyConfig to detector.MorphConfig.
func parseMorphologyConfig(config MorphologyConfig) detector.MorphConfig {
	morphConfig := detector.DefaultMorphConfig()
	morphConfig.KernelSize = config.KernelSize
	morphConfig.Iterations = config.Iterations

	// Parse operation string to MorphologicalOp
	switch strings.ToLower(config.Operation) {
	case dilateOp:
		morphConfig.Operation = detector.MorphDilate
	case "erode":
		morphConfig.Operation = detector.MorphErode
	case "opening":
		morphConfig.Operation = detector.MorphOpening
	case "closing":
		morphConfig.Operation = detector.MorphClosing
	case "smooth":
		morphConfig.Operation = detector.MorphSmooth
	default:
		morphConfig.Operation = detector.MorphNone
	}

	return morphConfig
}

// parseAdaptiveThresholdsConfig converts AdaptiveThresholdsConfig to detector.AdaptiveThresholdConfig.
func parseAdaptiveThresholdsConfig(config AdaptiveThresholdsConfig) detector.AdaptiveThresholdConfig {
	adaptiveConfig := detector.DefaultAdaptiveThresholdConfig()
	adaptiveConfig.Enabled = config.Enabled
	adaptiveConfig.MinDbThresh = config.MinDbThresh
	adaptiveConfig.MaxDbThresh = config.MaxDbThresh
	adaptiveConfig.MinBoxThresh = config.MinBoxThresh
	adaptiveConfig.MaxBoxThresh = config.MaxBoxThresh

	// Parse method string to AdaptiveThresholdMethod
	switch strings.ToLower(config.Method) {
	case "otsu":
		adaptiveConfig.Method = detector.AdaptiveMethodOtsu
	case histogramMethod:
		adaptiveConfig.Method = detector.AdaptiveMethodHistogram
	case "dynamic":
		adaptiveConfig.Method = detector.AdaptiveMethodDynamic
	default:
		adaptiveConfig.Method = detector.AdaptiveMethodHistogram
	}

	return adaptiveConfig
}
