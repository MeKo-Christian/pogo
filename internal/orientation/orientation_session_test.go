package orientation

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Verifies project-relative ONNX library path resolution works with repo layout.
func TestSetONNXLibraryPath_ProjectRelative(t *testing.T) {
	// The repository contains onnxruntime/lib via scripts; expect no error.
	err := setONNXLibraryPath()
	require.NoError(t, err)
}

// Simulate session init failure paths falling back to heuristic or erroring without fallback.
func TestNewClassifier_SessionInitFailures_FallbackAndError(t *testing.T) {
	// Create a temporary dummy file to satisfy os.Stat(modelPath) but not a real ONNX model.
	tmpDir := t.TempDir()
	dummyModel := filepath.Join(tmpDir, "bad-model.onnx")
	require.NoError(t, os.WriteFile(dummyModel, []byte("not a real model"), 0o644))

	// With fallback enabled: should not error and should be heuristic-only.
	cfgFallback := DefaultConfig()
	cfgFallback.Enabled = true
	cfgFallback.ModelPath = dummyModel
	cfgFallback.UseHeuristicFallback = true
	cls, err := NewClassifier(cfgFallback)
	require.NoError(t, err)
	require.NotNil(t, cls)
	// Predict on a small synthetic image should work via heuristic path.
	imgCfg := testutil.DefaultTestImageConfig()
	img, genErr := testutil.GenerateTextImage(imgCfg)
	require.NoError(t, genErr)
	_, pErr := cls.Predict(img)
	require.NoError(t, pErr)

	// Without fallback: should return an error during initialization
	cfgStrict := cfgFallback
	cfgStrict.UseHeuristicFallback = false
	cls2, err2 := NewClassifier(cfgStrict)
	require.Error(t, err2)
	assert.Nil(t, cls2)
}

func TestPredict_NilImage(t *testing.T) {
	cls, err := NewClassifier(Config{Enabled: false, UseHeuristicFallback: true})
	require.NoError(t, err)
	_, perr := cls.Predict(nil)
	require.Error(t, perr)
}

// Ensure UpdateModelPath keeps filename and relocates properly for both doc/textline defaults.
func TestUpdateModelPath_Variants(t *testing.T) {
	dir := t.TempDir()
	c1 := DefaultConfig()
	base1 := filepath.Base(c1.ModelPath)
	c1.UpdateModelPath(dir)
	assert.Equal(t, models.GetLayoutModelPath(dir, base1), c1.ModelPath)

	c2 := DefaultTextLineConfig()
	base2 := filepath.Base(c2.ModelPath)
	c2.UpdateModelPath(dir)
	assert.Equal(t, models.GetLayoutModelPath(dir, base2), c2.ModelPath)
}

func TestTryCreateONNXClassifier_MoreErrorPaths(t *testing.T) {
	// Test with a real-looking but invalid ONNX file
	tmpDir := t.TempDir()
	invalidModel := filepath.Join(tmpDir, "invalid.onnx")

	// Create a file that exists but is not a valid ONNX model
	invalidData := []byte("This is not a valid ONNX model file")
	require.NoError(t, os.WriteFile(invalidModel, invalidData, 0o644))

	cfg := DefaultConfig()
	cfg.ModelPath = invalidModel
	cfg.NumThreads = 4

	// This should fail at ONNX parsing stage
	_, err := tryCreateONNXClassifier(cfg)
	assert.Error(t, err)
	// The error should be related to ONNX parsing or IO info retrieval
	assert.Error(t, err)
}

func TestGetModelIOInfo_ErrorPath(t *testing.T) {
	// Test with non-existent file
	_, _, err := getModelIOInfo("/non/existent/file.onnx")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "io info")
}
