package config

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/orientation"
	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/MeKo-Tech/pogo/internal/recognizer"
	"github.com/MeKo-Tech/pogo/internal/rectify"
)

// Config represents the complete configuration for the pogo OCR application.
// It includes settings for all commands (image, pdf, serve, batch) and
// supports loading from configuration files, environment variables, and command-line flags.
type Config struct {
	// Global settings
	ModelsDir string `mapstructure:"models_dir" yaml:"models_dir" json:"models_dir"`
	LogLevel  string `mapstructure:"log_level" yaml:"log_level" json:"log_level"`
	Verbose   bool   `mapstructure:"verbose" yaml:"verbose" json:"verbose"`

	// Pipeline configuration
	Pipeline PipelineConfig `mapstructure:"pipeline" yaml:"pipeline" json:"pipeline"`

	// Output configuration
	Output OutputConfig `mapstructure:"output" yaml:"output" json:"output"`

	// Server configuration (for serve command)
	Server ServerConfig `mapstructure:"server" yaml:"server" json:"server"`

	// Batch processing configuration
	Batch BatchConfig `mapstructure:"batch" yaml:"batch" json:"batch"`

	// Processing features
	Features FeatureConfig `mapstructure:"features" yaml:"features" json:"features"`

	// GPU configuration
	GPU GPUConfig `mapstructure:"gpu" yaml:"gpu" json:"gpu"`
}

// PipelineConfig contains OCR pipeline settings.
type PipelineConfig struct {
	// Detection settings
	Detector DetectorConfig `mapstructure:"detector" yaml:"detector" json:"detector"`

	// Recognition settings
	Recognizer RecognizerConfig `mapstructure:"recognizer" yaml:"recognizer" json:"recognizer"`

	// Parallel processing
	Parallel ParallelConfig `mapstructure:"parallel" yaml:"parallel" json:"parallel"`

	// Resource management
	Resource ResourceConfig `mapstructure:"resource" yaml:"resource" json:"resource"`

	// Warmup iterations
	WarmupIterations int `mapstructure:"warmup_iterations" yaml:"warmup_iterations" json:"warmup_iterations"`
}

// DetectorConfig contains text detection settings.
type DetectorConfig struct {
	ModelPath      string  `mapstructure:"model_path" yaml:"model_path" json:"model_path"`
	DbThresh       float32 `mapstructure:"db_thresh" yaml:"db_thresh" json:"db_thresh"`
	DbBoxThresh    float32 `mapstructure:"db_box_thresh" yaml:"db_box_thresh" json:"db_box_thresh"`
	PolygonMode    string  `mapstructure:"polygon_mode" yaml:"polygon_mode" json:"polygon_mode"`
	UseNMS         bool    `mapstructure:"use_nms" yaml:"use_nms" json:"use_nms"`
	NMSThreshold   float64 `mapstructure:"nms_threshold" yaml:"nms_threshold" json:"nms_threshold"`
	NumThreads     int     `mapstructure:"num_threads" yaml:"num_threads" json:"num_threads"`
	MaxImageSize   int     `mapstructure:"max_image_size" yaml:"max_image_size" json:"max_image_size"`
}

// RecognizerConfig contains text recognition settings.
type RecognizerConfig struct {
	ModelPath          string  `mapstructure:"model_path" yaml:"model_path" json:"model_path"`
	DictPath           string  `mapstructure:"dict_path" yaml:"dict_path" json:"dict_path"`
	DictLangs          string  `mapstructure:"dict_langs" yaml:"dict_langs" json:"dict_langs"`
	Language           string  `mapstructure:"language" yaml:"language" json:"language"`
	ImageHeight        int     `mapstructure:"image_height" yaml:"image_height" json:"image_height"`
	MaxWidth           int     `mapstructure:"max_width" yaml:"max_width" json:"max_width"`
	PadWidthMultiple   int     `mapstructure:"pad_width_multiple" yaml:"pad_width_multiple" json:"pad_width_multiple"`
	MinConfidence      float64 `mapstructure:"min_confidence" yaml:"min_confidence" json:"min_confidence"`
	NumThreads         int     `mapstructure:"num_threads" yaml:"num_threads" json:"num_threads"`
}

// ParallelConfig contains parallel processing settings.
type ParallelConfig struct {
	MaxWorkers int `mapstructure:"max_workers" yaml:"max_workers" json:"max_workers"`
	BatchSize  int `mapstructure:"batch_size" yaml:"batch_size" json:"batch_size"`
}

// ResourceConfig contains resource management settings.
type ResourceConfig struct {
	MaxGoroutines int `mapstructure:"max_goroutines" yaml:"max_goroutines" json:"max_goroutines"`
}

// OutputConfig contains output formatting settings.
type OutputConfig struct {
	Format             string  `mapstructure:"format" yaml:"format" json:"format"`
	File               string  `mapstructure:"file" yaml:"file" json:"file"`
	ConfidencePrecision int    `mapstructure:"confidence_precision" yaml:"confidence_precision" json:"confidence_precision"`
	OverlayDir         string  `mapstructure:"overlay_dir" yaml:"overlay_dir" json:"overlay_dir"`
	OverlayBoxColor    string  `mapstructure:"overlay_box_color" yaml:"overlay_box_color" json:"overlay_box_color"`
	OverlayPolyColor   string  `mapstructure:"overlay_poly_color" yaml:"overlay_poly_color" json:"overlay_poly_color"`
}

// ServerConfig contains HTTP server settings.
type ServerConfig struct {
	Host            string `mapstructure:"host" yaml:"host" json:"host"`
	Port            int    `mapstructure:"port" yaml:"port" json:"port"`
	CORSOrigin      string `mapstructure:"cors_origin" yaml:"cors_origin" json:"cors_origin"`
	MaxUploadMB     int    `mapstructure:"max_upload_mb" yaml:"max_upload_mb" json:"max_upload_mb"`
	TimeoutSec      int    `mapstructure:"timeout_sec" yaml:"timeout_sec" json:"timeout_sec"`
	ShutdownTimeout int    `mapstructure:"shutdown_timeout" yaml:"shutdown_timeout" json:"shutdown_timeout"`
	OverlayEnabled  bool   `mapstructure:"overlay_enabled" yaml:"overlay_enabled" json:"overlay_enabled"`
}

// BatchConfig contains batch processing settings.
type BatchConfig struct {
	Workers       int    `mapstructure:"workers" yaml:"workers" json:"workers"`
	OutputDir     string `mapstructure:"output_dir" yaml:"output_dir" json:"output_dir"`
	ContinueOnError bool `mapstructure:"continue_on_error" yaml:"continue_on_error" json:"continue_on_error"`
}

// FeatureConfig contains feature toggle settings.
type FeatureConfig struct {
	// Orientation detection
	OrientationEnabled   bool    `mapstructure:"orientation_enabled" yaml:"orientation_enabled" json:"orientation_enabled"`
	OrientationThreshold float64 `mapstructure:"orientation_threshold" yaml:"orientation_threshold" json:"orientation_threshold"`
	OrientationModelPath string  `mapstructure:"orientation_model_path" yaml:"orientation_model_path" json:"orientation_model_path"`

	// Text line orientation
	TextlineEnabled   bool    `mapstructure:"textline_enabled" yaml:"textline_enabled" json:"textline_enabled"`
	TextlineThreshold float64 `mapstructure:"textline_threshold" yaml:"textline_threshold" json:"textline_threshold"`
	TextlineModelPath string  `mapstructure:"textline_model_path" yaml:"textline_model_path" json:"textline_model_path"`

	// Document rectification
	RectificationEnabled    bool    `mapstructure:"rectification_enabled" yaml:"rectification_enabled" json:"rectification_enabled"`
	RectificationModelPath  string  `mapstructure:"rectification_model_path" yaml:"rectification_model_path" json:"rectification_model_path"`
	RectificationThreshold  float64 `mapstructure:"rectification_threshold" yaml:"rectification_threshold" json:"rectification_threshold"`
	RectificationHeight     int     `mapstructure:"rectification_height" yaml:"rectification_height" json:"rectification_height"`
	RectificationDebugDir   string  `mapstructure:"rectification_debug_dir" yaml:"rectification_debug_dir" json:"rectification_debug_dir"`
}

// GPUConfig contains GPU acceleration settings.
type GPUConfig struct {
	Enabled    bool   `mapstructure:"enabled" yaml:"enabled" json:"enabled"`
	Device     int    `mapstructure:"device" yaml:"device" json:"device"`
	MemoryLimit string `mapstructure:"memory_limit" yaml:"memory_limit" json:"memory_limit"`
}

// DefaultConfig returns a configuration with sensible defaults.
func DefaultConfig() Config {
	return Config{
		ModelsDir: models.DefaultModelsDir,
		LogLevel:  "info",
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
			OrientationEnabled:   false,
			OrientationThreshold: 0.7,
			TextlineEnabled:      false,
			TextlineThreshold:    0.6,
			RectificationEnabled: false,
			RectificationThreshold: 0.5,
			RectificationHeight: 1024,
		},
		GPU: GPUConfig{
			Enabled:     false,
			Device:      0,
			MemoryLimit: "auto",
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

// Validate validates the configuration and returns any errors.
func (c *Config) Validate() error {
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

	// Validate positive integers
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

	// Validate polygon mode
	validPolygonModes := []string{"minrect", "contour"}
	if !contains(validPolygonModes, c.Pipeline.Detector.PolygonMode) {
		return fmt.Errorf("invalid polygon mode: %s (must be one of: %s)", c.Pipeline.Detector.PolygonMode, strings.Join(validPolygonModes, ", "))
	}

	// Validate GPU memory limit format
	if c.GPU.MemoryLimit != "auto" && c.GPU.MemoryLimit != "" {
		if err := validateMemoryLimit(c.GPU.MemoryLimit); err != nil {
			return fmt.Errorf("invalid GPU memory limit: %v", err)
		}
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
	}
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
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
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
	if limit == "" || limit == "auto" {
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