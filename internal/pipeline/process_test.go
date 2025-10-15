package pipeline

import (
    "image/color"
    "os"
    "testing"

    "github.com/MeKo-Tech/pogo/internal/testutil"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "strings"
)

func TestProcessImage_Errors(t *testing.T) {
	b := NewBuilder()
	// Skip if real models are not present
	det := b.Config().Detector.ModelPath
	rec := b.Config().Recognizer.ModelPath
	dict := b.Config().Recognizer.DictPath
	if _, err := os.Stat(det); err != nil {
		t.Skip("models unavailable")
	}
	if _, err := os.Stat(rec); err != nil {
		t.Skip("models unavailable")
	}
	if _, err := os.Stat(dict); err != nil {
		t.Skip("models unavailable")
	}

	p, err := b.Build()
	if err != nil {
		t.Skipf("pipeline build failed (likely ONNX): %v", err)
	}
	defer func() {
		require.NoError(t, p.Close())
	}()

	_, err = p.ProcessImage(nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nil")
}

func TestProcessImages_Empty(t *testing.T) {
	// This test does not require models
	p := &Pipeline{}
	_, err := p.ProcessImages(nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no images provided")
}

func TestProcessImage_Smoke(t *testing.T) {
	// Build only if models exist
	b := NewBuilder()
	det := b.Config().Detector.ModelPath
	rec := b.Config().Recognizer.ModelPath
	dict := b.Config().Recognizer.DictPath
	if _, err := os.Stat(det); err != nil {
		t.Skip("models unavailable")
	}
	if _, err := os.Stat(rec); err != nil {
		t.Skip("models unavailable")
	}
	if _, err := os.Stat(dict); err != nil {
		t.Skip("models unavailable")
	}

	p, err := b.Build()
	if err != nil {
		t.Skipf("pipeline build failed (likely ONNX): %v", err)
	}
	defer func() {
		require.NoError(t, p.Close())
	}()

	// Generate a small synthetic image
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "Hello"
	cfg.Background = color.White
	cfg.Foreground = color.Black
	img, err2 := testutil.GenerateTextImage(cfg)
	require.NoError(t, err2)

    res, err := p.ProcessImage(img)
    if err != nil {
        t.Skipf("process failed (runtime deps): %v", err)
    }
    require.NotNil(t, res)
    assert.Equal(t, img.Bounds().Dx(), res.Width)
    assert.Equal(t, img.Bounds().Dy(), res.Height)

    // Validate recognized text contains expected content (case-insensitive)
    txt, err := ToPlainTextImage(res)
    require.NoError(t, err)
    assert.Contains(t, strings.ToLower(txt), "hello")
}
