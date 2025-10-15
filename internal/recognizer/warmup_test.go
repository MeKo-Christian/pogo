package recognizer

import (
	"os"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/stretchr/testify/require"
)

func TestRecognizer_Warmup_SkipIfNoModel(t *testing.T) {
    modelPath := models.GetRecognitionModelPath("", false)
    dictPath := models.GetDictionaryPath("", models.DictionaryPPOCRv5)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Recognition model not available, skipping warmup test")
	}
	if _, err := os.Stat(dictPath); os.IsNotExist(err) {
		t.Skip("Dictionary not available, skipping warmup test")
	}
	cfg := DefaultConfig()
	cfg.ModelPath = modelPath
	cfg.DictPath = dictPath
	// Let recognizer infer height from the model (e.g., 48)
	cfg.ImageHeight = 0
	rec, err := NewRecognizer(cfg)
	require.NoError(t, err)
	defer func() { _ = rec.Close() }()

	require.NoError(t, rec.Warmup(0))
	require.NoError(t, rec.Warmup(2))
}
