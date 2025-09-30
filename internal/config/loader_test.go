package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	testValue = "test_value"
)

// clearPogoEnvVars clears all POGO_ environment variables and resets the global viper instance.
func clearPogoEnvVars() {
	for _, env := range os.Environ() {
		if len(env) > 5 && env[:5] == "POGO_" {
			parts := strings.SplitN(env, "=", 2)
			if len(parts) > 0 {
				_ = os.Unsetenv(parts[0]) // Ignore error in cleanup function
			}
		}
	}
	// Reset viper's global instance by creating a fresh one
	// Note: This is a workaround since viper caches environment variables
	// We can't fully reset it, but we can at least clear some state
}

// TestNewLoader tests loader creation.
func TestNewLoader(t *testing.T) {
	loader := NewLoader()
	if loader == nil {
		t.Fatal("NewLoader() returned nil")
	}
	if loader.v == nil {
		t.Error("Loader viper instance is nil")
	}
}

// TestLoadWithNoConfigFile tests loading with no config file present.
func TestLoadWithNoConfigFile(t *testing.T) {
	clearPogoEnvVars()

	// Create a temporary directory with no config file
	tmpDir := t.TempDir()
	originalWd, _ := os.Getwd()
	defer func() { _ = os.Chdir(originalWd) }() // Ignore error in cleanup

	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("Failed to change directory: %v", err)
	}

	loader := NewLoader()
	cfg, err := loader.Load()
	if err != nil {
		t.Errorf("Load() unexpected error: %v", err)
	}
	if cfg == nil {
		t.Fatal("Load() returned nil config")
	}

	// Should get default values
	if cfg.LogLevel != infoLevel {
		t.Errorf("Expected default log level '%s', got %s", infoLevel, cfg.LogLevel)
	}
	if cfg.Server.Port != 8080 {
		t.Errorf("Expected default port 8080, got %d", cfg.Server.Port)
	}
}

// TestLoadWithValidYAMLFile tests loading from a valid YAML file.
func TestLoadWithValidYAMLFile(t *testing.T) {
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "pogo.yaml")

	yamlContent := `
log_level: debug
verbose: true
models_dir: /custom/models
server:
  host: 0.0.0.0
  port: 9090
pipeline:
  detector:
    db_thresh: 0.4
  recognizer:
    language: de
`

	if err := os.WriteFile(configFile, []byte(yamlContent), 0o644); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}

	loader := NewLoader()
	cfg, err := loader.LoadWithFile(configFile)
	if err != nil {
		t.Errorf("LoadWithFile() unexpected error: %v", err)
	}
	if cfg.LogLevel != debugLevel {
		t.Errorf("Expected log level '%s', got %s", debugLevel, cfg.LogLevel)
	}
	if !cfg.Verbose {
		t.Error("Expected verbose to be true")
	}
	if cfg.ModelsDir != "/custom/models" {
		t.Errorf("Expected models dir '/custom/models', got %s", cfg.ModelsDir)
	}
	if cfg.Server.Host != "0.0.0.0" {
		t.Errorf("Expected host '0.0.0.0', got %s", cfg.Server.Host)
	}
	if cfg.Server.Port != 9090 {
		t.Errorf("Expected port 9090, got %d", cfg.Server.Port)
	}
	if cfg.Pipeline.Detector.DbThresh != 0.4 {
		t.Errorf("Expected db_thresh 0.4, got %f", cfg.Pipeline.Detector.DbThresh)
	}
	if cfg.Pipeline.Recognizer.Language != "de" {
		t.Errorf("Expected language 'de', got %s", cfg.Pipeline.Recognizer.Language)
	}
}

// TestLoadWithInvalidYAMLFile tests loading from an invalid YAML file.
func TestLoadWithInvalidYAMLFile(t *testing.T) {
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "pogo.yaml")

	invalidYAML := `
log_level: debug
  invalid indentation
    more bad indentation
`

	if err := os.WriteFile(configFile, []byte(invalidYAML), 0o644); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}

	loader := NewLoader()
	_, err := loader.LoadWithFile(configFile)

	if err == nil {
		t.Error("LoadWithFile() expected error for invalid YAML, got nil")
	}
}

// TestLoadWithNonExistentFile tests loading from a non-existent file.
func TestLoadWithNonExistentFile(t *testing.T) {
	loader := NewLoader()
	_, err := loader.LoadWithFile("/nonexistent/path/to/config.yaml")

	if err == nil {
		t.Error("LoadWithFile() expected error for non-existent file, got nil")
	}
}

// TestLoadWithValidationFailure tests loading with validation failure.
func TestLoadWithValidationFailure(t *testing.T) {
	clearPogoEnvVars()

	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "pogo.yaml")

	yamlContent := `
log_level: invalid_level
server:
  port: 0
`

	if err := os.WriteFile(configFile, []byte(yamlContent), 0o644); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}

	loader := NewLoader()
	_, err := loader.LoadWithFile(configFile)

	if err == nil {
		t.Error("LoadWithFile() expected validation error, got nil")
	}
}

// TestLoadWithoutValidation tests loading without validation.
func TestLoadWithoutValidation(t *testing.T) {
	clearPogoEnvVars()
	defer clearPogoEnvVars() // Clean up after the test

	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "pogo.yaml")

	yamlContent := `
log_level: invalid_level
server:
  port: -1
pipeline:
  detector:
    db_thresh: 5.0
`

	if err := os.WriteFile(configFile, []byte(yamlContent), 0o644); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}

	loader := NewLoader()
	cfg, err := loader.LoadWithFileWithoutValidation(configFile)
	// Should load successfully without validation
	if err != nil {
		t.Errorf("LoadWithFileWithoutValidation() unexpected error: %v", err)
	}
	if cfg == nil {
		t.Fatal("LoadWithFileWithoutValidation() returned nil config")
	}

	// Values should be loaded even if invalid
	if cfg.LogLevel != "invalid_level" {
		t.Errorf("Expected log level 'invalid_level', got %s", cfg.LogLevel)
	}
	if cfg.Server.Port != -1 {
		t.Errorf("Expected port -1, got %d", cfg.Server.Port)
	}
}

// TestEnvironmentVariableOverride tests environment variable override.
func TestEnvironmentVariableOverride(t *testing.T) {
	clearPogoEnvVars()
	defer clearPogoEnvVars() // Clean up after the test

	// Set environment variables
	envVars := map[string]string{
		"POGO_LOG_LEVEL":   "debug",
		"POGO_SERVER_PORT": "9999",
		"POGO_VERBOSE":     "true",
	}

	for key, value := range envVars {
		if err := os.Setenv(key, value); err != nil {
			t.Fatalf("Failed to set env var %s: %v", key, err)
		}
	}

	tmpDir := t.TempDir()
	originalWd, _ := os.Getwd()
	defer func() { _ = os.Chdir(originalWd) }() // Ignore error in cleanup

	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("Failed to change directory: %v", err)
	}

	loader := NewLoader()
	cfg, err := loader.Load()
	if err != nil {
		t.Errorf("Load() unexpected error: %v", err)
	}
	if cfg == nil {
		t.Fatal("Load() returned nil config")
	}

	if cfg.LogLevel != "debug" {
		t.Errorf("Expected log level 'debug' from env, got %s", cfg.LogLevel)
	}
	if cfg.Server.Port != 9999 {
		t.Errorf("Expected port 9999 from env, got %d", cfg.Server.Port)
	}
	if !cfg.Verbose {
		t.Error("Expected verbose true from env")
	}
}

// TestEnvironmentVariableWithUnderscores tests nested config with underscores.
func TestEnvironmentVariableWithUnderscores(t *testing.T) {
	clearPogoEnvVars()
	defer clearPogoEnvVars() // Clean up after the test

	envVars := map[string]string{
		"POGO_PIPELINE_DETECTOR_DB_THRESH":    "0.45",
		"POGO_PIPELINE_RECOGNIZER_LANGUAGE":   "fr",
		"POGO_FEATURES_ORIENTATION_ENABLED":   "true",
		"POGO_FEATURES_ORIENTATION_THRESHOLD": "0.85",
	}

	for key, value := range envVars {
		if err := os.Setenv(key, value); err != nil {
			t.Fatalf("Failed to set env var %s: %v", key, err)
		}
	}

	tmpDir := t.TempDir()
	originalWd, _ := os.Getwd()
	defer func() { _ = os.Chdir(originalWd) }() // Ignore error in cleanup

	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("Failed to change directory: %v", err)
	}

	loader := NewLoader()
	cfg, err := loader.Load()
	if err != nil {
		t.Errorf("Load() unexpected error: %v", err)
	}
	if cfg == nil {
		t.Fatal("Load() returned nil config")
	}

	if cfg.Pipeline.Detector.DbThresh != 0.45 {
		t.Errorf("Expected db_thresh 0.45 from env, got %f", cfg.Pipeline.Detector.DbThresh)
	}
	if cfg.Pipeline.Recognizer.Language != "fr" {
		t.Errorf("Expected language 'fr' from env, got %s", cfg.Pipeline.Recognizer.Language)
	}
	if !cfg.Features.OrientationEnabled {
		t.Error("Expected orientation enabled from env")
	}
	if cfg.Features.OrientationThreshold != 0.85 {
		t.Errorf("Expected orientation threshold 0.85 from env, got %f", cfg.Features.OrientationThreshold)
	}
}

// TestGetSetConfigValues tests Get and Set methods.
func TestGetSetConfigValues(t *testing.T) {
	loader := NewLoader()

	// Set a value
	loader.Set("test_key", testValue)

	// Get the value
	value := loader.GetString("test_key")
	if value != testValue {
		t.Errorf("Expected '%s', got %s", testValue, value)
	}

	// Get with generic Get
	genericValue := loader.Get("test_key")
	if genericValue != testValue {
		t.Errorf("Expected '%s', got %v", testValue, genericValue)
	}
}

// TestGetConfigFileUsed tests getting the config file path.
func TestGetConfigFileUsed(t *testing.T) {
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "pogo.yaml")

	yamlContent := `log_level: debug`
	if err := os.WriteFile(configFile, []byte(yamlContent), 0o644); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}

	loader := NewLoader()
	_, err := loader.LoadWithFile(configFile)
	if err != nil {
		t.Fatalf("LoadWithFile() error: %v", err)
	}

	usedFile := loader.GetConfigFileUsed()
	if usedFile != configFile {
		t.Errorf("Expected config file %s, got %s", configFile, usedFile)
	}
}

// TestGetViper tests getting the viper instance.
func TestGetViper(t *testing.T) {
	loader := NewLoader()
	v := loader.GetViper()

	if v == nil {
		t.Error("GetViper() returned nil")
	}
	if v != loader.v {
		t.Error("GetViper() returned different instance")
	}
}

// TestGetResolvedConfig tests getting all resolved config.
func TestGetResolvedConfig(t *testing.T) {
	loader := NewLoader()
	loader.Set("test_key", testValue)

	resolved := loader.GetResolvedConfig()
	if resolved == nil {
		t.Error("GetResolvedConfig() returned nil")
	}

	if value, ok := resolved["test_key"]; !ok || value != testValue {
		t.Errorf("Expected test_key='%s' in resolved config, got %v", testValue, value)
	}
}

// TestWriteConfigToFile tests writing config to file.
func TestWriteConfigToFile(t *testing.T) {
	tmpDir := t.TempDir()
	outputFile := filepath.Join(tmpDir, "output.yaml")

	loader := NewLoader()
	loader.Set("log_level", "debug")
	loader.Set("verbose", true)

	err := loader.WriteConfigToFile(outputFile)
	if err != nil {
		t.Errorf("WriteConfigToFile() error: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(outputFile); os.IsNotExist(err) {
		t.Error("Config file was not written")
	}
}

// TestGenerateDefaultConfigFile tests generating a default config file.
func TestGenerateDefaultConfigFile(t *testing.T) {
	tmpDir := t.TempDir()
	outputFile := filepath.Join(tmpDir, "default.yaml")

	err := GenerateDefaultConfigFile(outputFile)
	if err != nil {
		t.Errorf("GenerateDefaultConfigFile() error: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(outputFile); os.IsNotExist(err) {
		t.Error("Default config file was not generated")
	}

	// Try to load the generated file
	loader := NewLoader()
	cfg, err := loader.LoadWithFile(outputFile)
	if err != nil {
		t.Errorf("Failed to load generated config: %v", err)
	}
	if cfg == nil {
		t.Error("Loaded config is nil")
	}
}

// TestGenerateDefaultConfigFileWithEmptyFilename tests default filename.
func TestGenerateDefaultConfigFileWithEmptyFilename(t *testing.T) {
	tmpDir := t.TempDir()
	originalWd, _ := os.Getwd()
	defer func() { _ = os.Chdir(originalWd) }() // Ignore error in cleanup

	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("Failed to change directory: %v", err)
	}

	err := GenerateDefaultConfigFile("")
	if err != nil {
		t.Errorf("GenerateDefaultConfigFile(\"\") error: %v", err)
	}

	// Should create pogo.yaml
	expectedFile := filepath.Join(tmpDir, "pogo.yaml")
	if _, err := os.Stat(expectedFile); os.IsNotExist(err) {
		t.Error("Default pogo.yaml was not generated")
	}
}

// TestGetConfigSearchPaths tests getting config search paths.
func TestGetConfigSearchPaths(t *testing.T) {
	paths := GetConfigSearchPaths()

	if len(paths) == 0 {
		t.Error("GetConfigSearchPaths() returned empty slice")
	}

	// Should include current directory
	hasCurrentDir := false
	for _, path := range paths {
		if path == "." {
			hasCurrentDir = true
			break
		}
	}
	if !hasCurrentDir {
		t.Error("Search paths don't include current directory")
	}
}

// TestPrintConfigInfo tests printing config info (no assertions, just coverage).
func TestPrintConfigInfo(t *testing.T) {
	loader := NewLoader()

	// Just call it to ensure it doesn't panic
	loader.PrintConfigInfo()
}

// TestLoadWithEmptyConfigFile tests loading with empty config file.
func TestLoadWithEmptyConfigFile(t *testing.T) {
	clearPogoEnvVars()

	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "pogo.yaml")

	// Create empty file
	if err := os.WriteFile(configFile, []byte(""), 0o644); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}

	loader := NewLoader()
	cfg, err := loader.LoadWithFile(configFile)
	if err != nil {
		t.Errorf("LoadWithFile() unexpected error: %v", err)
	}

	// Should get default values
	if cfg.LogLevel != infoLevel {
		t.Errorf("Expected default log level '%s', got %s", infoLevel, cfg.LogLevel)
	}
}

// TestMultipleConfigSourcesPrecedence tests precedence of config sources.
func TestMultipleConfigSourcesPrecedence(t *testing.T) {
	clearPogoEnvVars()
	defer clearPogoEnvVars()

	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "pogo.yaml")

	// Create config file with log_level=warn
	yamlContent := `log_level: warn`
	if err := os.WriteFile(configFile, []byte(yamlContent), 0o644); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}

	// Set environment variable with log_level=debug (should override file)
	if err := os.Setenv("POGO_LOG_LEVEL", "debug"); err != nil {
		t.Fatalf("Failed to set env var: %v", err)
	}

	loader := NewLoader()
	cfg, err := loader.LoadWithFile(configFile)
	if err != nil {
		t.Errorf("LoadWithFile() error: %v", err)
	}

	// Environment variable should take precedence
	if cfg.LogLevel != "debug" {
		t.Errorf("Expected log level 'debug' from env (should override file), got %s", cfg.LogLevel)
	}
}

// TestLoadWithEmptyFilenameUsesDefaultLoad tests that LoadWithFile("") uses Load().
func TestLoadWithEmptyFilenameUsesDefaultLoad(t *testing.T) {
	clearPogoEnvVars()

	tmpDir := t.TempDir()
	originalWd, _ := os.Getwd()
	defer func() { _ = os.Chdir(originalWd) }() // Ignore error in cleanup

	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("Failed to change directory: %v", err)
	}

	loader := NewLoader()
	cfg, err := loader.LoadWithFile("")
	if err != nil {
		t.Errorf("LoadWithFile(\"\") unexpected error: %v", err)
	}
	if cfg == nil {
		t.Fatal("LoadWithFile(\"\") returned nil config")
	}

	// Should use defaults
	if cfg.LogLevel != infoLevel {
		t.Errorf("Expected default log level, got %s", cfg.LogLevel)
	}
}

// TestLoadWithoutValidationUsesDefaults tests LoadWithoutValidation with no file.
func TestLoadWithoutValidationUsesDefaults(t *testing.T) {
	clearPogoEnvVars()

	tmpDir := t.TempDir()
	originalWd, _ := os.Getwd()
	defer func() { _ = os.Chdir(originalWd) }() // Ignore error in cleanup

	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("Failed to change directory: %v", err)
	}

	loader := NewLoader()
	cfg, err := loader.LoadWithoutValidation()
	if err != nil {
		t.Errorf("LoadWithoutValidation() unexpected error: %v", err)
	}
	if cfg == nil {
		t.Fatal("LoadWithoutValidation() returned nil config")
	}

	if cfg.LogLevel != infoLevel {
		t.Errorf("Expected default log level, got %s", cfg.LogLevel)
	}
}

// TestLoadWithFileWithoutValidationEmptyString tests empty string behavior.
func TestLoadWithFileWithoutValidationEmptyString(t *testing.T) {
	tmpDir := t.TempDir()
	originalWd, _ := os.Getwd()
	defer func() { _ = os.Chdir(originalWd) }() // Ignore error in cleanup

	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("Failed to change directory: %v", err)
	}

	loader := NewLoader()
	cfg, err := loader.LoadWithFileWithoutValidation("")
	if err != nil {
		t.Errorf("LoadWithFileWithoutValidation(\"\") unexpected error: %v", err)
	}
	if cfg == nil {
		t.Fatal("LoadWithFileWithoutValidation(\"\") returned nil config")
	}
}

// TestBindFlag tests BindFlag (currently a no-op).
func TestBindFlag(t *testing.T) {
	loader := NewLoader()
	err := loader.BindFlag("test.key", "test-flag")
	if err != nil {
		t.Errorf("BindFlag() unexpected error: %v", err)
	}
}

// TestBindFlagSet tests BindFlagSet (currently a no-op).
func TestBindFlagSet(t *testing.T) {
	loader := NewLoader()
	err := loader.BindFlagSet(nil)
	if err != nil {
		t.Errorf("BindFlagSet() unexpected error: %v", err)
	}
}
