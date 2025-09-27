package pipeline

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
