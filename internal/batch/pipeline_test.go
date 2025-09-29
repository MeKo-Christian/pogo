package batch

import (
	"testing"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildPipeline_BasicConfig(t *testing.T) {
	config := &Config{
		ModelsDir:       "/test/models",
		Workers:         2,
		Confidence:      0.5,
		MinRecConf:      0.0,
		BatchSize:       4,
		MemoryLimitStr:  "512MB",
		MaxGoroutines:   8,
		AdaptiveScaling: true,
		Backpressure:    true,
	}

	pl, err := buildPipeline(config, nil)
	require.NoError(t, err)
	require.NotNil(t, pl)

	// Pipeline should be properly initialized
	assert.NotNil(t, pl.Detector)
	assert.NotNil(t, pl.Recognizer)
}

func TestBuildPipeline_WithModelPaths(t *testing.T) {
	config := &Config{
		ModelsDir:  "/test/models",
		DetModel:   "/custom/det.onnx",
		RecModel:   "/custom/rec.onnx",
		DictCSV:    "en.txt,de.txt",
		Workers:    1,
		Confidence: 0.3,
	}

	pl, err := buildPipeline(config, nil)
	require.NoError(t, err)
	require.NotNil(t, pl)
}

func TestBuildPipeline_WithLanguageDicts(t *testing.T) {
	config := &Config{
		ModelsDir:  "/test/models",
		DictLangs:  "en,de",
		Workers:    1,
		Confidence: 0.3,
	}

	pl, err := buildPipeline(config, nil)
	require.NoError(t, err)
	require.NotNil(t, pl)
}

func TestBuildPipeline_WithFeatures(t *testing.T) {
	config := &Config{
		ModelsDir:         "/test/models",
		DetectOrientation: true,
		DetectTextline:    true,
		Rectify:           true,
		RecHeight:         64,
		Workers:           1,
		Confidence:        0.3,
	}

	pl, err := buildPipeline(config, nil)
	require.NoError(t, err)
	require.NotNil(t, pl)
}

func TestBuildPipeline_WithThresholds(t *testing.T) {
	config := &Config{
		ModelsDir:       "/test/models",
		OrientThresh:    0.8,
		TextlineThresh:  0.7,
		RectifyMask:     0.6,
		RectifyHeight:   480,
		RectifyModel:    "/custom/rectify.onnx",
		RectifyDebugDir: "/tmp/debug",
		Workers:         1,
		Confidence:      0.3,
	}

	pl, err := buildPipeline(config, nil)
	require.NoError(t, err)
	require.NotNil(t, pl)
}

func TestBuildPipeline_InvalidModelsDir(t *testing.T) {
	config := &Config{
		ModelsDir: "/nonexistent/models",
		Workers:   1,
	}

	pl, err := buildPipeline(config, nil)
	assert.Error(t, err)
	assert.Nil(t, pl)
}

func TestConfigurePipelineModels_NoCustomModels(t *testing.T) {
	config := &Config{
		ModelsDir: "/test/models",
	}

	builder := pipeline.NewBuilder()
	result := configurePipelineModels(builder, config)

	// Should return the same builder
	assert.Equal(t, builder, result)
}

func TestConfigurePipelineModels_WithDetModel(t *testing.T) {
	config := &Config{
		ModelsDir: "/test/models",
		DetModel:  "/custom/det.onnx",
	}

	builder := pipeline.NewBuilder()
	result := configurePipelineModels(builder, config)

	assert.Equal(t, builder, result) // Should return the same builder instance
}

func TestConfigurePipelineModels_WithDictCSV(t *testing.T) {
	config := &Config{
		ModelsDir: "/test/models",
		DictCSV:   "dict1.txt,dict2.txt",
	}

	builder := pipeline.NewBuilder()
	result := configurePipelineModels(builder, config)

	assert.Equal(t, builder, result)
}

func TestConfigurePipelineFeatures_NoFeatures(t *testing.T) {
	config := &Config{}

	builder := pipeline.NewBuilder()
	result := configurePipelineFeatures(builder, config)

	assert.Equal(t, builder, result)
}

func TestConfigurePipelineFeatures_WithOrientation(t *testing.T) {
	config := &Config{
		DetectOrientation: true,
	}

	builder := pipeline.NewBuilder()
	result := configurePipelineFeatures(builder, config)

	assert.Equal(t, builder, result)
}

func TestConfigurePipelineFeatures_WithRectification(t *testing.T) {
	config := &Config{
		Rectify:   true,
		RecHeight: 64,
	}

	builder := pipeline.NewBuilder()
	result := configurePipelineFeatures(builder, config)

	assert.Equal(t, builder, result)
}

func TestConfigurePipelineThresholds_DefaultValues(t *testing.T) {
	config := &Config{}

	builder := pipeline.NewBuilder()
	result := configurePipelineThresholds(builder, config)

	assert.Equal(t, builder, result)
}

func TestConfigurePipelineThresholds_CustomValues(t *testing.T) {
	config := &Config{
		OrientThresh:   0.8,
		TextlineThresh: 0.7,
		RectifyMask:    0.6,
	}

	builder := pipeline.NewBuilder()
	result := configurePipelineThresholds(builder, config)

	assert.Equal(t, builder, result)
}

func TestParseMemoryLimitOrDefault_EmptyString(t *testing.T) {
	result := parseMemoryLimitOrDefault("")
	assert.Equal(t, uint64(0), result)
}

func TestParseMemoryLimitOrDefault_ValidString(t *testing.T) {
	result := parseMemoryLimitOrDefault("256MB")
	assert.Equal(t, uint64(256*1024*1024), result)
}

func TestParseMemoryLimitOrDefault_InvalidString(t *testing.T) {
	result := parseMemoryLimitOrDefault("invalid")
	assert.Equal(t, uint64(0), result)
}

func TestParseMemoryLimit_Bytes(t *testing.T) {
	result, err := parseMemoryLimit("1024")
	require.NoError(t, err)
	assert.Equal(t, uint64(1024), result)
}

func TestParseMemoryLimit_Kilobytes(t *testing.T) {
	result, err := parseMemoryLimit("512KB")
	require.NoError(t, err)
	assert.Equal(t, uint64(512*1024), result)
}

func TestParseMemoryLimit_Megabytes(t *testing.T) {
	result, err := parseMemoryLimit("256MB")
	require.NoError(t, err)
	assert.Equal(t, uint64(256*1024*1024), result)
}

func TestParseMemoryLimit_Gigabytes(t *testing.T) {
	result, err := parseMemoryLimit("2GB")
	require.NoError(t, err)
	assert.Equal(t, uint64(2*1024*1024*1024), result)
}

func TestParseMemoryLimit_Terabytes(t *testing.T) {
	result, err := parseMemoryLimit("1TB")
	require.NoError(t, err)
	assert.Equal(t, uint64(1024*1024*1024*1024), result)
}

func TestParseMemoryLimit_CaseInsensitive(t *testing.T) {
	result, err := parseMemoryLimit("128mb")
	require.NoError(t, err)
	assert.Equal(t, uint64(128*1024*1024), result)
}

func TestParseMemoryLimit_InvalidFormat(t *testing.T) {
	testCases := []string{
		"invalid",
		"123XYZ",
		"MB",
		"GB",
		"",
	}

	for _, tc := range testCases {
		_, err := parseMemoryLimit(tc)
		assert.Error(t, err, "should fail for: %s", tc)
	}
}

func TestParseMemoryLimit_Whitespace(t *testing.T) {
	result, err := parseMemoryLimit("  128 MB  ")
	require.NoError(t, err)
	assert.Equal(t, uint64(128*1024*1024), result)
}
