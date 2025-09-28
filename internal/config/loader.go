package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/viper"
)

const (
	// ConfigFileName is the base name for configuration files (without extension).
	ConfigFileName = "pogo"

	// EnvPrefix is the prefix for environment variables.
	EnvPrefix = "POGO"
)

// Loader handles loading configuration from various sources.
type Loader struct {
	v *viper.Viper
}

// NewLoader creates a new configuration loader.
func NewLoader() *Loader {
	// Use the global viper instance to ensure flag bindings work
	return &Loader{v: viper.GetViper()}
}

// Load loads configuration from files, environment variables, and sets defaults.
// It returns the loaded configuration and any error encountered.
func (l *Loader) Load() (*Config, error) {
	// Set configuration file details
	l.v.SetConfigName(ConfigFileName)
	l.v.SetConfigType("yaml") // Primary format, but viper supports multiple formats

	// Add configuration search paths
	l.addConfigPaths()

	// Set environment variable handling
	l.setupEnvironmentVariables()

	// Set defaults
	l.setDefaults()

	// Try to read configuration file
	if err := l.v.ReadInConfig(); err != nil {
		// It's okay if config file doesn't exist, we'll use defaults and env vars
		var configFileNotFoundError viper.ConfigFileNotFoundError
		if !errors.As(err, &configFileNotFoundError) {
			// Only return error if it's NOT a "config file not found" error
			return nil, fmt.Errorf("error reading config file: %w", err)
		}
		// Config file not found is OK, continue with defaults and env vars
	}

	// Unmarshal into our config struct
	var config Config
	if err := l.v.Unmarshal(&config); err != nil {
		return nil, fmt.Errorf("error unmarshaling config: %w", err)
	}

	// Validate the configuration
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}

	return &config, nil
}

// LoadWithoutValidation loads configuration from files, environment variables, and sets defaults.
// It returns the loaded configuration without validation.
func (l *Loader) LoadWithoutValidation() (*Config, error) {
	// Set configuration file details
	l.v.SetConfigName(ConfigFileName)
	l.v.SetConfigType("yaml") // Primary format, but viper supports multiple formats

	// Add configuration search paths
	l.addConfigPaths()

	// Set environment variable handling
	l.setupEnvironmentVariables()

	// Set defaults
	l.setDefaults()

	// Try to read configuration file
	if err := l.v.ReadInConfig(); err != nil {
		// It's okay if config file doesn't exist, we'll use defaults and env vars
		var configFileNotFoundError viper.ConfigFileNotFoundError
		if !errors.As(err, &configFileNotFoundError) {
			// Only return error if it's NOT a "config file not found" error
			return nil, fmt.Errorf("error reading config file: %w", err)
		}
		// Config file not found is OK, continue with defaults and env vars
	}

	// Unmarshal into our config struct
	var config Config
	if err := l.v.Unmarshal(&config); err != nil {
		return nil, fmt.Errorf("error unmarshaling config: %w", err)
	}

	return &config, nil
}

// LoadWithFile loads configuration from a specific file path.
func (l *Loader) LoadWithFile(configFile string) (*Config, error) {
	if configFile == "" {
		return l.Load()
	}

	// Check if file exists
	if _, err := os.Stat(configFile); os.IsNotExist(err) {
		return nil, fmt.Errorf("config file does not exist: %s", configFile)
	}

	// Set the specific config file
	l.v.SetConfigFile(configFile)

	// Set environment variable handling
	l.setupEnvironmentVariables()

	// Set defaults
	l.setDefaults()

	// Read the config file
	if err := l.v.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("error reading config file %s: %w", configFile, err)
	}

	// Unmarshal into our config struct
	var config Config
	if err := l.v.Unmarshal(&config); err != nil {
		return nil, fmt.Errorf("error unmarshaling config: %w", err)
	}

	// Validate the configuration
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}

	return &config, nil
}

// LoadWithFileWithoutValidation loads configuration from a specific file path without validation.
func (l *Loader) LoadWithFileWithoutValidation(configFile string) (*Config, error) {
	if configFile == "" {
		return l.LoadWithoutValidation()
	}

	// Check if file exists
	if _, err := os.Stat(configFile); os.IsNotExist(err) {
		return nil, fmt.Errorf("config file does not exist: %s", configFile)
	}

	// Set the specific config file
	l.v.SetConfigFile(configFile)

	// Set environment variable handling
	l.setupEnvironmentVariables()

	// Set defaults
	l.setDefaults()

	// Read the config file
	if err := l.v.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("error reading config file %s: %w", configFile, err)
	}

	// Unmarshal into our config struct
	var config Config
	if err := l.v.Unmarshal(&config); err != nil {
		return nil, fmt.Errorf("error unmarshaling config: %w", err)
	}

	return &config, nil
}

// BindFlag binds a command-line flag to a configuration key.
// This should be called after the flag has been defined.
func (l *Loader) BindFlag(key, flagName string) error {
	// Note: This method is for future use, actual binding happens in root command
	return nil
}

// BindFlagSet binds flags from a flag set to configuration keys.
func (l *Loader) BindFlagSet(flagSet interface{}) error {
	// This would be called after cobra flags are set up
	// The actual binding happens in the root command initialization
	return nil
}

// Get returns a value from the configuration.
func (l *Loader) Get(key string) interface{} {
	return l.v.Get(key)
}

// GetString returns a string value from the configuration.
func (l *Loader) GetString(key string) string {
	return l.v.GetString(key)
}

// Set sets a value in the configuration.
func (l *Loader) Set(key string, value interface{}) {
	l.v.Set(key, value)
}

// GetConfigFileUsed returns the path of the config file used.
func (l *Loader) GetConfigFileUsed() string {
	return l.v.ConfigFileUsed()
}

// GetViper returns the underlying viper instance for advanced usage.
func (l *Loader) GetViper() *viper.Viper {
	return l.v
}

// addConfigPaths adds the standard configuration search paths.
func (l *Loader) addConfigPaths() {
	// Current directory
	l.v.AddConfigPath(".")

	// User's home directory
	if home, err := os.UserHomeDir(); err == nil {
		l.v.AddConfigPath(home)
	}

	// System-wide configuration
	l.v.AddConfigPath("/etc/pogo")

	// XDG config directory
	if configDir, exists := os.LookupEnv("XDG_CONFIG_HOME"); exists {
		l.v.AddConfigPath(filepath.Join(configDir, "pogo"))
	} else if home, err := os.UserHomeDir(); err == nil {
		l.v.AddConfigPath(filepath.Join(home, ".config", "pogo"))
	}
}

// setupEnvironmentVariables configures environment variable handling.
func (l *Loader) setupEnvironmentVariables() {
	// Set the prefix for environment variables
	l.v.SetEnvPrefix(EnvPrefix)

	// Enable automatic environment variable binding
	l.v.AutomaticEnv()

	// Replace dots and dashes with underscores in env var names
	l.v.SetEnvKeyReplacer(strings.NewReplacer(".", "_", "-", "_"))
}

// setDefaults sets default values for all configuration options.
func (l *Loader) setDefaults() {
	defaults := DefaultConfig()

	// Global settings
	l.v.SetDefault("models_dir", defaults.ModelsDir)
	l.v.SetDefault("log_level", defaults.LogLevel)
	l.v.SetDefault("verbose", defaults.Verbose)

	// Pipeline defaults
	l.v.SetDefault("pipeline.detector.db_thresh", defaults.Pipeline.Detector.DbThresh)
	l.v.SetDefault("pipeline.detector.db_box_thresh", defaults.Pipeline.Detector.DbBoxThresh)
	l.v.SetDefault("pipeline.detector.polygon_mode", defaults.Pipeline.Detector.PolygonMode)
	l.v.SetDefault("pipeline.detector.use_nms", defaults.Pipeline.Detector.UseNMS)
	l.v.SetDefault("pipeline.detector.nms_threshold", defaults.Pipeline.Detector.NMSThreshold)
	l.v.SetDefault("pipeline.detector.num_threads", defaults.Pipeline.Detector.NumThreads)
	l.v.SetDefault("pipeline.detector.max_image_size", defaults.Pipeline.Detector.MaxImageSize)
	l.v.SetDefault("pipeline.detector.use_adaptive_nms", defaults.Pipeline.Detector.UseAdaptiveNMS)
	l.v.SetDefault("pipeline.detector.adaptive_nms_scale", defaults.Pipeline.Detector.AdaptiveNMSScale)
	l.v.SetDefault("pipeline.detector.size_aware_nms", defaults.Pipeline.Detector.SizeAwareNMS)
	l.v.SetDefault("pipeline.detector.min_region_size", defaults.Pipeline.Detector.MinRegionSize)
	l.v.SetDefault("pipeline.detector.max_region_size", defaults.Pipeline.Detector.MaxRegionSize)
	l.v.SetDefault("pipeline.detector.size_nms_scale_factor", defaults.Pipeline.Detector.SizeNMSScaleFactor)

	l.v.SetDefault("pipeline.recognizer.language", defaults.Pipeline.Recognizer.Language)
	l.v.SetDefault("pipeline.recognizer.image_height", defaults.Pipeline.Recognizer.ImageHeight)
	l.v.SetDefault("pipeline.recognizer.max_width", defaults.Pipeline.Recognizer.MaxWidth)
	l.v.SetDefault("pipeline.recognizer.pad_width_multiple", defaults.Pipeline.Recognizer.PadWidthMultiple)
	l.v.SetDefault("pipeline.recognizer.min_confidence", defaults.Pipeline.Recognizer.MinConfidence)
	l.v.SetDefault("pipeline.recognizer.num_threads", defaults.Pipeline.Recognizer.NumThreads)

	l.v.SetDefault("pipeline.parallel.max_workers", defaults.Pipeline.Parallel.MaxWorkers)
	l.v.SetDefault("pipeline.parallel.batch_size", defaults.Pipeline.Parallel.BatchSize)

	l.v.SetDefault("pipeline.resource.max_goroutines", defaults.Pipeline.Resource.MaxGoroutines)
	l.v.SetDefault("pipeline.warmup_iterations", defaults.Pipeline.WarmupIterations)

	// Output defaults
	l.v.SetDefault("output.format", defaults.Output.Format)
	l.v.SetDefault("output.confidence_precision", defaults.Output.ConfidencePrecision)
	l.v.SetDefault("output.overlay_box_color", defaults.Output.OverlayBoxColor)
	l.v.SetDefault("output.overlay_poly_color", defaults.Output.OverlayPolyColor)

	// Server defaults
	l.v.SetDefault("server.host", defaults.Server.Host)
	l.v.SetDefault("server.port", defaults.Server.Port)
	l.v.SetDefault("server.cors_origin", defaults.Server.CORSOrigin)
	l.v.SetDefault("server.max_upload_mb", defaults.Server.MaxUploadMB)
	l.v.SetDefault("server.timeout_sec", defaults.Server.TimeoutSec)
	l.v.SetDefault("server.shutdown_timeout", defaults.Server.ShutdownTimeout)
	l.v.SetDefault("server.overlay_enabled", defaults.Server.OverlayEnabled)

	// Batch defaults
	l.v.SetDefault("batch.workers", defaults.Batch.Workers)
	l.v.SetDefault("batch.continue_on_error", defaults.Batch.ContinueOnError)

	// Feature defaults
	l.v.SetDefault("features.orientation_enabled", defaults.Features.OrientationEnabled)
	l.v.SetDefault("features.orientation_threshold", defaults.Features.OrientationThreshold)
	l.v.SetDefault("features.textline_enabled", defaults.Features.TextlineEnabled)
	l.v.SetDefault("features.textline_threshold", defaults.Features.TextlineThreshold)
	l.v.SetDefault("features.rectification_enabled", defaults.Features.RectificationEnabled)
	l.v.SetDefault("features.rectification_threshold", defaults.Features.RectificationThreshold)
	l.v.SetDefault("features.rectification_height", defaults.Features.RectificationHeight)

	// GPU defaults
	l.v.SetDefault("gpu.enabled", defaults.GPU.Enabled)
	l.v.SetDefault("gpu.device", defaults.GPU.Device)
	l.v.SetDefault("gpu.memory_limit", defaults.GPU.MemoryLimit)
}

// GetResolvedConfig returns the current resolved configuration for debugging.
func (l *Loader) GetResolvedConfig() map[string]interface{} {
	return l.v.AllSettings()
}

// WriteConfigToFile writes the current configuration to a file.
func (l *Loader) WriteConfigToFile(filename string) error {
	return l.v.WriteConfigAs(filename)
}

// GenerateDefaultConfigFile generates a default configuration file.
func GenerateDefaultConfigFile(filename string) error {
	loader := NewLoader()
	loader.setDefaults()

	// If no filename provided, use default
	if filename == "" {
		filename = "pogo.yaml"
	}

	return loader.WriteConfigToFile(filename)
}

// GetConfigSearchPaths returns the paths where configuration files are searched.
func GetConfigSearchPaths() []string {
	paths := []string{"."}

	if home, err := os.UserHomeDir(); err == nil {
		paths = append(paths, home)
		paths = append(paths, filepath.Join(home, ".config", "pogo"))
	}

	if configDir, exists := os.LookupEnv("XDG_CONFIG_HOME"); exists {
		paths = append(paths, filepath.Join(configDir, "pogo"))
	}

	paths = append(paths, "/etc/pogo")

	return paths
}

// PrintConfigInfo prints information about configuration loading for debugging.
func (l *Loader) PrintConfigInfo() {
	fmt.Printf("Configuration file used: %s\n", l.GetConfigFileUsed())
	fmt.Printf("Configuration search paths: %v\n", GetConfigSearchPaths())
	fmt.Printf("Environment prefix: %s\n", EnvPrefix)
}
