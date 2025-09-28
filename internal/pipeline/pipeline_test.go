package pipeline

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	customDetectorPath   = "/custom/detector.onnx"
	customRecognizerPath = "/custom/recognizer.onnx"
)

// testModelPathSetter is a helper function to test model path setter methods.
func testModelPathSetter(t *testing.T, setter func(*Builder, string) *Builder, getter func(Config) string,
	methodName string,
) {
	t.Helper()
	tests := []struct {
		name         string
		path         string
		expectedPath string
	}{
		{
			name:         fmt.Sprintf("sets %s model path", methodName),
			path:         fmt.Sprintf("/custom/%s.onnx", methodName),
			expectedPath: fmt.Sprintf("/custom/%s.onnx", methodName),
		},
		{
			name:         "ignores empty path",
			path:         "",
			expectedPath: "", // Should remain default
		},
		{
			name:         "sets relative path",
			path:         fmt.Sprintf("./models/custom_%s.onnx", methodName),
			expectedPath: fmt.Sprintf("./models/custom_%s.onnx", methodName),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := NewBuilder()
			originalPath := getter(b.Config())

			// Apply the method
			result := setter(b, tt.path)

			// Should return the same builder for chaining
			assert.Equal(t, b, result)

			// Check the configuration
			cfg := b.Config()
			if tt.path == "" {
				// Empty path should not change the original path
				assert.Equal(t, originalPath, getter(cfg))
			} else {
				assert.Equal(t, tt.expectedPath, getter(cfg))
			}
		})
	}
}

func TestBuilder_WithDetectorModelPath(t *testing.T) {
	testModelPathSetter(t,
		func(b *Builder, path string) *Builder { return b.WithDetectorModelPath(path) },
		func(cfg Config) string { return cfg.Detector.ModelPath },
		"detector",
	)
}

func TestBuilder_WithRecognizerModelPath(t *testing.T) {
	testModelPathSetter(t,
		func(b *Builder, path string) *Builder { return b.WithRecognizerModelPath(path) },
		func(cfg Config) string { return cfg.Recognizer.ModelPath },
		"recognizer",
	)
}

func TestBuilder_Validate_MissingModels(t *testing.T) {
	b := NewBuilder().WithModelsDir("/nonexistent/path")
	err := b.Validate()
	require.Error(t, err)
	// Depending on validation order, detector model or dictionary may be reported first.
	assert.Regexp(t, `(?i)(model|dictionary) not found`, err.Error())
}

func TestBuilder_Validate_WithTempModels(t *testing.T) {
	dir := t.TempDir()

	// Create organized structure with dummy files
	detPath := models.ResolveModelPath(dir, models.TypeDetection, models.VariantMobile, models.DetectionMobile)
	recPath := models.ResolveModelPath(dir, models.TypeRecognition, models.VariantMobile, models.RecognitionMobile)
	dictPath := models.ResolveModelPath(dir, models.TypeDictionaries, "", models.DictionaryPPOCRKeysV1)

	for _, p := range []string{detPath, recPath, dictPath} {
		require.NoError(t, os.MkdirAll(filepath.Dir(p), 0o755))
		require.NoError(t, os.WriteFile(p, []byte("dummy"), 0o644))
	}

	b := NewBuilder().WithModelsDir(dir)
	// Only DictPath is guaranteed to be updated by UpdateModelPath; Detector/Recognizer
	// keep overrides unless cleared. Validate should pass using the dummy files.
	cfg := b.Config()
	assert.True(t, strings.HasPrefix(cfg.Recognizer.DictPath, dir))

	// Validate should succeed with dummy files
	require.NoError(t, b.Validate())
}

func TestBuilder_Build_SkipIfNoONNXOrNoRealModels(t *testing.T) {
	// Attempt to build only if real default models are present
	det := models.GetDetectionModelPath("", false)
	rec := models.GetRecognitionModelPath("", false)
	dict := models.GetDictionaryPath("", models.DictionaryPPOCRKeysV1)
	if _, err := os.Stat(det); err != nil {
		t.Skip("detection model not available; skipping Build test")
	}
	if _, err := os.Stat(rec); err != nil {
		t.Skip("recognition model not available; skipping Build test")
	}
	if _, err := os.Stat(dict); err != nil {
		t.Skip("dictionary not available; skipping Build test")
	}

	b := NewBuilder()
	p, err := b.Build()
	if err != nil {
		// Likely due to missing ONNX runtime or build tags
		if strings.Contains(err.Error(), "build with -tags onnx") || strings.Contains(err.Error(), "ONNX") {
			t.Skipf("Build failed due to ONNX availability: %v", err)
		}
	}
	if p != nil {
		_ = p.Close()
	}
}

func TestBuilder_WithDetectorModelPath_Integration(t *testing.T) {
	// Test that the detector model path override works with other builder methods
	// Note: WithModelsDir will override explicit model paths due to UpdateModelPath call
	b := NewBuilder()

	// Set a custom detector path and then apply other configurations (avoiding WithModelsDir)
	b.WithDetectorModelPath(customDetectorPath).
		WithThreads(4).
		WithImageHeight(48)

	cfg := b.Config()

	// The custom detector path should be preserved when not calling WithModelsDir
	assert.Equal(t, customDetectorPath, cfg.Detector.ModelPath)
	assert.Equal(t, 4, cfg.Detector.NumThreads)
	assert.Equal(t, 48, cfg.Recognizer.ImageHeight)
}

func TestBuilder_WithRecognizerModelPath_Integration(t *testing.T) {
	// Test that the recognizer model path override works with other builder methods
	// Note: WithModelsDir will override explicit model paths due to UpdateModelPath call
	b := NewBuilder()

	// Set a custom recognizer path and then apply other configurations (avoiding WithModelsDir)
	b.WithRecognizerModelPath(customRecognizerPath).
		WithImageHeight(48).
		WithThreads(2)

	cfg := b.Config()

	// The custom recognizer path should be preserved when not calling WithModelsDir
	assert.Equal(t, customRecognizerPath, cfg.Recognizer.ModelPath)
	assert.Equal(t, 48, cfg.Recognizer.ImageHeight)
	assert.Equal(t, 2, cfg.Recognizer.NumThreads)
}

func TestBuilder_ModelPathPrecedence(t *testing.T) {
	// Test that explicit model paths take precedence over WithModelsDir
	b := NewBuilder()

	// First set models dir, then override with explicit paths
	b.WithModelsDir("/models/dir").
		WithDetectorModelPath(customDetectorPath).
		WithRecognizerModelPath(customRecognizerPath)

	cfg := b.Config()

	// Explicit paths should override the models dir defaults
	assert.Equal(t, customDetectorPath, cfg.Detector.ModelPath)
	assert.Equal(t, customRecognizerPath, cfg.Recognizer.ModelPath)
	assert.Equal(t, "/models/dir", cfg.ModelsDir)
}

func TestBuilder_ModelPathOverrideAfterModelsDir(t *testing.T) {
	// Test that setting model paths after WithModelsDir works correctly
	b := NewBuilder()

	// Start with a models directory
	b.WithModelsDir("/models/dir")

	originalDetectorPath := b.Config().Detector.ModelPath
	originalRecognizerPath := b.Config().Recognizer.ModelPath

	// Now override with explicit paths
	b.WithDetectorModelPath(customDetectorPath).
		WithRecognizerModelPath(customRecognizerPath)

	cfg := b.Config()

	// Paths should be overridden
	assert.Equal(t, customDetectorPath, cfg.Detector.ModelPath)
	assert.Equal(t, customRecognizerPath, cfg.Recognizer.ModelPath)
	assert.NotEqual(t, originalDetectorPath, cfg.Detector.ModelPath)
	assert.NotEqual(t, originalRecognizerPath, cfg.Recognizer.ModelPath)
}

func TestBuilder_WithModelsDirOverridesExplicitPaths(t *testing.T) {
	// Test that WithModelsDir overrides explicitly set model paths due to UpdateModelPath call
	b := NewBuilder()

	// First set explicit paths
	b.WithDetectorModelPath(customDetectorPath).
		WithRecognizerModelPath(customRecognizerPath)

	// Verify they are set
	cfg1 := b.Config()
	assert.Equal(t, customDetectorPath, cfg1.Detector.ModelPath)
	assert.Equal(t, customRecognizerPath, cfg1.Recognizer.ModelPath)

	// Now call WithModelsDir which should override the explicit paths
	b.WithModelsDir("/models/dir")

	cfg2 := b.Config()
	// WithModelsDir calls UpdateModelPath which generates new paths based on the models directory
	assert.NotEqual(t, customDetectorPath, cfg2.Detector.ModelPath)
	assert.NotEqual(t, customRecognizerPath, cfg2.Recognizer.ModelPath)
	assert.Contains(t, cfg2.Detector.ModelPath, "/models/dir")
	assert.Contains(t, cfg2.Recognizer.ModelPath, "/models/dir")
	assert.Equal(t, "/models/dir", cfg2.ModelsDir)
}
