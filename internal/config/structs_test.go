package config

import (
	"encoding/json"
	"testing"

	"gopkg.in/yaml.v3"
)

const (
	testModelsDir = "/test/models"
	testHost      = "0.0.0.0"
	testDictPath  = "/test/dict.txt"
)

// TestConfigJSONMarshaling tests marshaling Config to JSON.
func TestConfigJSONMarshaling(t *testing.T) {
	cfg := DefaultConfig()
	cfg.LogLevel = debugLevel
	cfg.Verbose = true
	cfg.Server.Port = 9090

	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("json.Marshal() error: %v", err)
	}

	if len(data) == 0 {
		t.Error("Marshaled JSON is empty")
	}

	// Verify it contains expected fields
	var result map[string]interface{}
	if err := json.Unmarshal(data, &result); err != nil {
		t.Fatalf("json.Unmarshal() error: %v", err)
	}

	if result["log_level"] != debugLevel {
		t.Errorf("Expected log_level '%s', got %v", debugLevel, result["log_level"])
	}
	if result["verbose"] != true {
		t.Errorf("Expected verbose true, got %v", result["verbose"])
	}
}

// TestConfigJSONUnmarshaling tests unmarshaling Config from JSON.
func TestConfigJSONUnmarshaling(t *testing.T) {
	jsonData := `{
		"log_level": "debug",
		"verbose": true,
		"models_dir": "/test/models",
		"server": {
			"host": "0.0.0.0",
			"port": 9090
		},
		"pipeline": {
			"detector": {
				"db_thresh": 0.4
			},
			"recognizer": {
				"language": "en"
			}
		}
	}`

	var cfg Config
	err := json.Unmarshal([]byte(jsonData), &cfg)
	if err != nil {
		t.Fatalf("json.Unmarshal() error: %v", err)
	}

	if cfg.LogLevel != debugLevel {
		t.Errorf("Expected log_level '%s', got %s", debugLevel, cfg.LogLevel)
	}
	if !cfg.Verbose {
		t.Error("Expected verbose true")
	}
	if cfg.ModelsDir != testModelsDir {
		t.Errorf("Expected models_dir '%s', got %s", testModelsDir, cfg.ModelsDir)
	}
	if cfg.Server.Host != testHost {
		t.Errorf("Expected host '%s', got %s", testHost, cfg.Server.Host)
	}
	if cfg.Server.Port != 9090 {
		t.Errorf("Expected port 9090, got %d", cfg.Server.Port)
	}
	if cfg.Pipeline.Detector.DbThresh != 0.4 {
		t.Errorf("Expected db_thresh 0.4, got %f", cfg.Pipeline.Detector.DbThresh)
	}
}

// TestConfigYAMLMarshaling tests marshaling Config to YAML.
func TestConfigYAMLMarshaling(t *testing.T) {
	cfg := DefaultConfig()
	cfg.LogLevel = warnLevel
	cfg.Verbose = false
	cfg.Server.Port = 8888

	data, err := yaml.Marshal(cfg)
	if err != nil {
		t.Fatalf("yaml.Marshal() error: %v", err)
	}

	if len(data) == 0 {
		t.Error("Marshaled YAML is empty")
	}

	// Verify it contains expected fields
	var result map[string]interface{}
	if err := yaml.Unmarshal(data, &result); err != nil {
		t.Fatalf("yaml.Unmarshal() error: %v", err)
	}

	if result["log_level"] != warnLevel {
		t.Errorf("Expected log_level '%s', got %v", warnLevel, result["log_level"])
	}
}

// TestConfigYAMLUnmarshaling tests unmarshaling Config from YAML.
func TestConfigYAMLUnmarshaling(t *testing.T) {
	yamlData := `
log_level: error
verbose: true
models_dir: /yaml/models
server:
  host: 127.0.0.1
  port: 7070
pipeline:
  detector:
    db_thresh: 0.35
  recognizer:
    language: de
features:
  orientation_enabled: true
  orientation_threshold: 0.8
`

	var cfg Config
	err := yaml.Unmarshal([]byte(yamlData), &cfg)
	if err != nil {
		t.Fatalf("yaml.Unmarshal() error: %v", err)
	}

	if cfg.LogLevel != "error" {
		t.Errorf("Expected log_level 'error', got %s", cfg.LogLevel)
	}
	if !cfg.Verbose {
		t.Error("Expected verbose true")
	}
	if cfg.ModelsDir != "/yaml/models" {
		t.Errorf("Expected models_dir '/yaml/models', got %s", cfg.ModelsDir)
	}
	if cfg.Server.Host != "127.0.0.1" {
		t.Errorf("Expected host '127.0.0.1', got %s", cfg.Server.Host)
	}
	if cfg.Server.Port != 7070 {
		t.Errorf("Expected port 7070, got %d", cfg.Server.Port)
	}
	if cfg.Pipeline.Detector.DbThresh != 0.35 {
		t.Errorf("Expected db_thresh 0.35, got %f", cfg.Pipeline.Detector.DbThresh)
	}
	if cfg.Pipeline.Recognizer.Language != "de" {
		t.Errorf("Expected language 'de', got %s", cfg.Pipeline.Recognizer.Language)
	}
	if !cfg.Features.OrientationEnabled {
		t.Error("Expected orientation enabled")
	}
	if cfg.Features.OrientationThreshold != 0.8 {
		t.Errorf("Expected orientation threshold 0.8, got %f", cfg.Features.OrientationThreshold)
	}
}

// TestConfigRoundTripJSON tests JSON round-trip serialization.
func TestConfigRoundTripJSON(t *testing.T) {
	original := DefaultConfig()
	original.LogLevel = debugLevel
	original.Verbose = true
	original.Server.Port = 9999
	original.Pipeline.Detector.DbThresh = 0.42
	original.Features.OrientationEnabled = true

	// Marshal to JSON
	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("json.Marshal() error: %v", err)
	}

	// Unmarshal back
	var decoded Config
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error: %v", err)
	}

	// Compare key fields
	if decoded.LogLevel != original.LogLevel {
		t.Errorf("LogLevel mismatch: expected %s, got %s", original.LogLevel, decoded.LogLevel)
	}
	if decoded.Verbose != original.Verbose {
		t.Errorf("Verbose mismatch: expected %v, got %v", original.Verbose, decoded.Verbose)
	}
	if decoded.Server.Port != original.Server.Port {
		t.Errorf("Port mismatch: expected %d, got %d", original.Server.Port, decoded.Server.Port)
	}
	if decoded.Pipeline.Detector.DbThresh != original.Pipeline.Detector.DbThresh {
		t.Errorf("DbThresh mismatch: expected %f, got %f", original.Pipeline.Detector.DbThresh, decoded.Pipeline.Detector.DbThresh)
	}
	if decoded.Features.OrientationEnabled != original.Features.OrientationEnabled {
		t.Errorf("OrientationEnabled mismatch: expected %v, got %v", original.Features.OrientationEnabled, decoded.Features.OrientationEnabled)
	}
}

// TestConfigRoundTripYAML tests YAML round-trip serialization.
func TestConfigRoundTripYAML(t *testing.T) {
	original := DefaultConfig()
	original.LogLevel = warnLevel
	original.Verbose = false
	original.Server.Host = "192.168.1.1"
	original.Server.Port = 8888
	original.Pipeline.Recognizer.Language = "fr"
	original.GPU.Enabled = true
	original.GPU.Device = 1
	original.GPU.MemoryLimit = "2GB"

	// Marshal to YAML
	data, err := yaml.Marshal(original)
	if err != nil {
		t.Fatalf("yaml.Marshal() error: %v", err)
	}

	// Unmarshal back
	var decoded Config
	if err := yaml.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("yaml.Unmarshal() error: %v", err)
	}

	// Compare key fields
	if decoded.LogLevel != original.LogLevel {
		t.Errorf("LogLevel mismatch: expected %s, got %s", original.LogLevel, decoded.LogLevel)
	}
	if decoded.Verbose != original.Verbose {
		t.Errorf("Verbose mismatch: expected %v, got %v", original.Verbose, decoded.Verbose)
	}
	if decoded.Server.Host != original.Server.Host {
		t.Errorf("Host mismatch: expected %s, got %s", original.Server.Host, decoded.Server.Host)
	}
	if decoded.Server.Port != original.Server.Port {
		t.Errorf("Port mismatch: expected %d, got %d", original.Server.Port, decoded.Server.Port)
	}
	if decoded.Pipeline.Recognizer.Language != original.Pipeline.Recognizer.Language {
		t.Errorf("Language mismatch: expected %s, got %s", original.Pipeline.Recognizer.Language, decoded.Pipeline.Recognizer.Language)
	}
	if decoded.GPU.Enabled != original.GPU.Enabled {
		t.Errorf("GPU Enabled mismatch: expected %v, got %v", original.GPU.Enabled, decoded.GPU.Enabled)
	}
	if decoded.GPU.Device != original.GPU.Device {
		t.Errorf("GPU Device mismatch: expected %d, got %d", original.GPU.Device, decoded.GPU.Device)
	}
	if decoded.GPU.MemoryLimit != original.GPU.MemoryLimit {
		t.Errorf("GPU MemoryLimit mismatch: expected %s, got %s", original.GPU.MemoryLimit, decoded.GPU.MemoryLimit)
	}
}

// TestPipelineConfigStructure tests PipelineConfig structure.
func TestPipelineConfigStructure(t *testing.T) {
	cfg := PipelineConfig{
		Detector: DetectorConfig{
			DbThresh: 0.3,
		},
		Recognizer: RecognizerConfig{
			Language: "en",
		},
		Parallel: ParallelConfig{
			MaxWorkers: 4,
		},
		Resource: ResourceConfig{
			MaxGoroutines: 100,
		},
		WarmupIterations: 5,
	}

	if cfg.Detector.DbThresh != 0.3 {
		t.Errorf("Expected DbThresh 0.3, got %f", cfg.Detector.DbThresh)
	}
	if cfg.Recognizer.Language != "en" {
		t.Errorf("Expected Language 'en', got %s", cfg.Recognizer.Language)
	}
	if cfg.Parallel.MaxWorkers != 4 {
		t.Errorf("Expected MaxWorkers 4, got %d", cfg.Parallel.MaxWorkers)
	}
	if cfg.Resource.MaxGoroutines != 100 {
		t.Errorf("Expected MaxGoroutines 100, got %d", cfg.Resource.MaxGoroutines)
	}
	if cfg.WarmupIterations != 5 {
		t.Errorf("Expected WarmupIterations 5, got %d", cfg.WarmupIterations)
	}
}

// TestDetectorConfigStructure tests DetectorConfig structure.
func TestDetectorConfigStructure(t *testing.T) {
	cfg := DetectorConfig{
		ModelPath:      "/test/model.onnx",
		DbThresh:       0.4,
		DbBoxThresh:    0.6,
		PolygonMode:    "minrect",
		UseNMS:         true,
		NMSThreshold:   0.5,
		NumThreads:     4,
		MaxImageSize:   2048,
		UseAdaptiveNMS: true,
		MinRegionSize:  10,
		MaxRegionSize:  1000,
		Morphology: MorphologyConfig{
			Operation:  dilateOp,
			KernelSize: 5,
			Iterations: 2,
		},
		AdaptiveThresholds: AdaptiveThresholdsConfig{
			Enabled:      true,
			Method:       "histogram",
			MinDbThresh:  0.1,
			MaxDbThresh:  0.8,
			MinBoxThresh: 0.3,
			MaxBoxThresh: 0.9,
		},
	}

	if cfg.ModelPath != "/test/model.onnx" {
		t.Errorf("Expected ModelPath '/test/model.onnx', got %s", cfg.ModelPath)
	}
	if cfg.Morphology.Operation != dilateOp {
		t.Errorf("Expected morphology operation '%s', got %s", dilateOp, cfg.Morphology.Operation)
	}
	if !cfg.AdaptiveThresholds.Enabled {
		t.Error("Expected adaptive thresholds enabled")
	}
}

// TestRecognizerConfigStructure tests RecognizerConfig structure.
func TestRecognizerConfigStructure(t *testing.T) {
	cfg := RecognizerConfig{
		ModelPath:        "/test/rec.onnx",
		DictPath:         testDictPath,
		DictLangs:        "en,de",
		Language:         "en",
		ImageHeight:      48,
		MaxWidth:         512,
		PadWidthMultiple: 32,
		MinConfidence:    0.5,
		NumThreads:       2,
	}

	if cfg.ModelPath != "/test/rec.onnx" {
		t.Errorf("Expected ModelPath '/test/rec.onnx', got %s", cfg.ModelPath)
	}
	if cfg.DictPath != testDictPath {
		t.Errorf("Expected DictPath '%s', got %s", testDictPath, cfg.DictPath)
	}
	if cfg.DictLangs != "en,de" {
		t.Errorf("Expected DictLangs 'en,de', got %s", cfg.DictLangs)
	}
	if cfg.Language != "en" {
		t.Errorf("Expected Language 'en', got %s", cfg.Language)
	}
}

// TestOutputConfigStructure tests OutputConfig structure.
func TestOutputConfigStructure(t *testing.T) {
	cfg := OutputConfig{
		Format:              "json",
		File:                "/output/results.json",
		ConfidencePrecision: 3,
		OverlayDir:          "/output/overlays",
		OverlayBoxColor:     "#FF0000",
		OverlayPolyColor:    "#00FF00",
	}

	if cfg.Format != "json" {
		t.Errorf("Expected Format 'json', got %s", cfg.Format)
	}
	if cfg.File != "/output/results.json" {
		t.Errorf("Expected File '/output/results.json', got %s", cfg.File)
	}
	if cfg.ConfidencePrecision != 3 {
		t.Errorf("Expected ConfidencePrecision 3, got %d", cfg.ConfidencePrecision)
	}
}

// TestServerConfigStructure tests ServerConfig structure.
func TestServerConfigStructure(t *testing.T) {
	cfg := ServerConfig{
		Host:              "0.0.0.0",
		Port:              9090,
		CORSOrigin:        "*",
		MaxUploadMB:       100,
		TimeoutSec:        60,
		ShutdownTimeout:   30,
		OverlayEnabled:    true,
		RateLimitEnabled:  true,
		RequestsPerMinute: 60,
		RequestsPerHour:   1000,
		MaxRequestsPerDay: 10000,
		MaxDataPerDay:     1073741824, // 1GB
	}

	if cfg.Host != "0.0.0.0" {
		t.Errorf("Expected Host '0.0.0.0', got %s", cfg.Host)
	}
	if cfg.Port != 9090 {
		t.Errorf("Expected Port 9090, got %d", cfg.Port)
	}
	if !cfg.RateLimitEnabled {
		t.Error("Expected RateLimitEnabled true")
	}
	if cfg.RequestsPerMinute != 60 {
		t.Errorf("Expected RequestsPerMinute 60, got %d", cfg.RequestsPerMinute)
	}
}

// TestFeatureConfigStructure tests FeatureConfig structure.
func TestFeatureConfigStructure(t *testing.T) {
	cfg := FeatureConfig{
		OrientationEnabled:     true,
		OrientationThreshold:   0.7,
		OrientationModelPath:   "/models/orientation.onnx",
		TextlineEnabled:        true,
		TextlineThreshold:      0.6,
		TextlineModelPath:      "/models/textline.onnx",
		RectificationEnabled:   true,
		RectificationModelPath: "/models/rectify.onnx",
		RectificationThreshold: 0.5,
		RectificationHeight:    1024,
		RectificationDebugDir:  "/debug",
	}

	if !cfg.OrientationEnabled {
		t.Error("Expected OrientationEnabled true")
	}
	if !cfg.TextlineEnabled {
		t.Error("Expected TextlineEnabled true")
	}
	if !cfg.RectificationEnabled {
		t.Error("Expected RectificationEnabled true")
	}
	if cfg.RectificationHeight != 1024 {
		t.Errorf("Expected RectificationHeight 1024, got %d", cfg.RectificationHeight)
	}
}

// TestGPUConfigStructure tests GPUConfig structure.
func TestGPUConfigStructure(t *testing.T) {
	cfg := GPUConfig{
		Enabled:     true,
		Device:      1,
		MemoryLimit: "2GB",
	}

	if !cfg.Enabled {
		t.Error("Expected Enabled true")
	}
	if cfg.Device != 1 {
		t.Errorf("Expected Device 1, got %d", cfg.Device)
	}
	if cfg.MemoryLimit != "2GB" {
		t.Errorf("Expected MemoryLimit '2GB', got %s", cfg.MemoryLimit)
	}
}

// TestBatchConfigStructure tests BatchConfig structure.
func TestBatchConfigStructure(t *testing.T) {
	cfg := BatchConfig{
		Workers:         8,
		OutputDir:       "/batch/output",
		ContinueOnError: true,
	}

	if cfg.Workers != 8 {
		t.Errorf("Expected Workers 8, got %d", cfg.Workers)
	}
	if cfg.OutputDir != "/batch/output" {
		t.Errorf("Expected OutputDir '/batch/output', got %s", cfg.OutputDir)
	}
	if !cfg.ContinueOnError {
		t.Error("Expected ContinueOnError true")
	}
}

// TestZeroValuesVsDefaults tests zero values vs defaults.
func TestZeroValuesVsDefaults(t *testing.T) {
	var zero Config
	defaults := DefaultConfig()

	// Zero values should be different from defaults
	if zero.LogLevel == defaults.LogLevel {
		t.Error("Zero LogLevel should differ from default")
	}
	if zero.Server.Port == defaults.Server.Port {
		t.Error("Zero Port should differ from default")
	}
	if zero.Batch.Workers == defaults.Batch.Workers {
		t.Error("Zero Workers should differ from default")
	}
}

// TestStructTags tests that all struct fields have proper tags.
func TestStructTags(t *testing.T) {
	// This is a simple sanity check that the structs can be marshaled
	cfg := DefaultConfig()

	// Test JSON tags
	jsonData, err := json.Marshal(cfg)
	if err != nil {
		t.Errorf("Failed to marshal config to JSON: %v", err)
	}
	if len(jsonData) == 0 {
		t.Error("JSON marshaling produced empty output")
	}

	// Test YAML tags
	yamlData, err := yaml.Marshal(cfg)
	if err != nil {
		t.Errorf("Failed to marshal config to YAML: %v", err)
	}
	if len(yamlData) == 0 {
		t.Error("YAML marshaling produced empty output")
	}
}

// TestNestedStructInitialization tests nested struct initialization.
func TestNestedStructInitialization(t *testing.T) {
	cfg := Config{
		Pipeline: PipelineConfig{
			Detector: DetectorConfig{
				Morphology: MorphologyConfig{
					Operation:  dilateOp,
					KernelSize: 5,
				},
				AdaptiveThresholds: AdaptiveThresholdsConfig{
					Enabled: true,
					Method:  histogramMethod,
				},
			},
		},
	}

	if cfg.Pipeline.Detector.Morphology.Operation != dilateOp {
		t.Error("Nested morphology config not initialized correctly")
	}
	if !cfg.Pipeline.Detector.AdaptiveThresholds.Enabled {
		t.Error("Nested adaptive thresholds config not initialized correctly")
	}
}
