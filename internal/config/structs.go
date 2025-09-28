//nolint:lll
package config

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
	ModelPath    string  `mapstructure:"model_path" yaml:"model_path" json:"model_path"`
	DbThresh     float32 `mapstructure:"db_thresh" yaml:"db_thresh" json:"db_thresh"`
	DbBoxThresh  float32 `mapstructure:"db_box_thresh" yaml:"db_box_thresh" json:"db_box_thresh"`
	PolygonMode  string  `mapstructure:"polygon_mode" yaml:"polygon_mode" json:"polygon_mode"`
	UseNMS       bool    `mapstructure:"use_nms" yaml:"use_nms" json:"use_nms"`
	NMSThreshold float64 `mapstructure:"nms_threshold" yaml:"nms_threshold" json:"nms_threshold"`
	NumThreads   int     `mapstructure:"num_threads" yaml:"num_threads" json:"num_threads"`
	MaxImageSize int     `mapstructure:"max_image_size" yaml:"max_image_size" json:"max_image_size"`

	// Class-agnostic NMS tuning
	UseAdaptiveNMS     bool    `mapstructure:"use_adaptive_nms" yaml:"use_adaptive_nms" json:"use_adaptive_nms"`
	AdaptiveNMSScale   float64 `mapstructure:"adaptive_nms_scale" yaml:"adaptive_nms_scale" json:"adaptive_nms_scale"`
	SizeAwareNMS       bool    `mapstructure:"size_aware_nms" yaml:"size_aware_nms" json:"size_aware_nms"`
	MinRegionSize      int     `mapstructure:"min_region_size" yaml:"min_region_size" json:"min_region_size"`
	MaxRegionSize      int     `mapstructure:"max_region_size" yaml:"max_region_size" json:"max_region_size"`
	SizeNMSScaleFactor float64 `mapstructure:"size_nms_scale_factor" yaml:"size_nms_scale_factor" json:"size_nms_scale_factor"`
}

// RecognizerConfig contains text recognition settings.
type RecognizerConfig struct {
	ModelPath        string  `mapstructure:"model_path" yaml:"model_path" json:"model_path"`
	DictPath         string  `mapstructure:"dict_path" yaml:"dict_path" json:"dict_path"`
	DictLangs        string  `mapstructure:"dict_langs" yaml:"dict_langs" json:"dict_langs"`
	Language         string  `mapstructure:"language" yaml:"language" json:"language"`
	ImageHeight      int     `mapstructure:"image_height" yaml:"image_height" json:"image_height"`
	MaxWidth         int     `mapstructure:"max_width" yaml:"max_width" json:"max_width"`
	PadWidthMultiple int     `mapstructure:"pad_width_multiple" yaml:"pad_width_multiple" json:"pad_width_multiple"`
	MinConfidence    float64 `mapstructure:"min_confidence" yaml:"min_confidence" json:"min_confidence"`
	NumThreads       int     `mapstructure:"num_threads" yaml:"num_threads" json:"num_threads"`
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
	Format              string `mapstructure:"format" yaml:"format" json:"format"`
	File                string `mapstructure:"file" yaml:"file" json:"file"`
	ConfidencePrecision int    `mapstructure:"confidence_precision" yaml:"confidence_precision" json:"confidence_precision"`
	OverlayDir          string `mapstructure:"overlay_dir" yaml:"overlay_dir" json:"overlay_dir"`
	OverlayBoxColor     string `mapstructure:"overlay_box_color" yaml:"overlay_box_color" json:"overlay_box_color"`
	OverlayPolyColor    string `mapstructure:"overlay_poly_color" yaml:"overlay_poly_color" json:"overlay_poly_color"`
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
	Workers         int    `mapstructure:"workers" yaml:"workers" json:"workers"`
	OutputDir       string `mapstructure:"output_dir" yaml:"output_dir" json:"output_dir"`
	ContinueOnError bool   `mapstructure:"continue_on_error" yaml:"continue_on_error" json:"continue_on_error"`
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
	RectificationEnabled   bool    `mapstructure:"rectification_enabled" yaml:"rectification_enabled" json:"rectification_enabled"`
	RectificationModelPath string  `mapstructure:"rectification_model_path" yaml:"rectification_model_path" json:"rectification_model_path"`
	RectificationThreshold float64 `mapstructure:"rectification_threshold" yaml:"rectification_threshold" json:"rectification_threshold"`
	RectificationHeight    int     `mapstructure:"rectification_height" yaml:"rectification_height" json:"rectification_height"`
	RectificationDebugDir  string  `mapstructure:"rectification_debug_dir" yaml:"rectification_debug_dir" json:"rectification_debug_dir"`
}

// GPUConfig contains GPU acceleration settings.
type GPUConfig struct {
	Enabled     bool   `mapstructure:"enabled" yaml:"enabled" json:"enabled"`
	Device      int    `mapstructure:"device" yaml:"device" json:"device"`
	MemoryLimit string `mapstructure:"memory_limit" yaml:"memory_limit" json:"memory_limit"`
}
