package recognizer

import (
	"os"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	assert.Equal(t, models.GetRecognitionModelPath("", false), cfg.ModelPath)
	assert.Equal(t, models.GetDictionaryPath("", models.DictionaryPPOCRKeysV1), cfg.DictPath)
	assert.Equal(t, 48, cfg.ImageHeight)
	assert.False(t, cfg.UseServerModel)
	assert.Equal(t, 0, cfg.NumThreads)
}

func TestNewRecognizer_EmptyPaths(t *testing.T) {
	cfg := Config{}
	r, err := NewRecognizer(cfg)
	require.Error(t, err)
	require.Nil(t, r)
	assert.Contains(t, err.Error(), "model path cannot be empty")
}

func TestNewRecognizer_InvalidModelPath(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ModelPath = "no/such/model.onnx"
	cfg.DictPath = "no/such/dict.txt"
	r, err := NewRecognizer(cfg)
	require.Error(t, err)
	require.Nil(t, r)
	assert.Contains(t, err.Error(), "model file not found")
}

func TestNewRecognizer_MissingDictionary(t *testing.T) {
	// Skip if model not available to avoid double failure
	modelPath := models.GetRecognitionModelPath("", false)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Recognition model not available, skipping test")
	}
	cfg := DefaultConfig()
	cfg.ModelPath = modelPath
	cfg.DictPath = "no/such/dict.txt"
	r, err := NewRecognizer(cfg)
	require.Error(t, err)
	require.Nil(t, r)
	assert.Contains(t, err.Error(), "dictionary file not found")
}

func TestNewRecognizer_ValidModel(t *testing.T) {
	modelPath := models.GetRecognitionModelPath("", false)
	dictPath := models.GetDictionaryPath("", models.DictionaryPPOCRKeysV1)

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Recognition model not available, skipping test")
	}
	if _, err := os.Stat(dictPath); os.IsNotExist(err) {
		t.Skip("Dictionary not available, skipping test")
	}

	cfg := DefaultConfig()
	cfg.ModelPath = modelPath
	cfg.DictPath = dictPath

	r, err := NewRecognizer(cfg)
	require.NoError(t, err)
	require.NotNil(t, r)
	defer func() {
		require.NoError(t, r.Close())
	}()

	// Basic assertions
	assert.Equal(t, cfg, r.GetConfig())
	inShape := r.GetInputShape()
	outShape := r.GetOutputShape()
	assert.NotEmpty(t, inShape)
	assert.NotEmpty(t, outShape)

	info := r.GetModelInfo()
	assert.Equal(t, modelPath, info["model_path"])
	assert.Equal(t, dictPath, info["dict_path"])
	assert.Positive(t, info["charset_size"].(int))
}
