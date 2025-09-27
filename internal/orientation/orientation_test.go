package orientation

import (
	"image/color"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHeuristicOrientation_Rotated90(t *testing.T) {
	// Generate a rotated 90-degree text image
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "Rotated Text"
	cfg.Rotation = 90
	cfg.Background = color.White
	cfg.Foreground = color.Black
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	// Use heuristic-only classifier (no ONNX required)
	cls, err := NewClassifier(Config{Enabled: false, UseHeuristicFallback: true})
	require.NoError(t, err)

	res, err := cls.Predict(img)
	require.NoError(t, err)
	// Validate output shape: angle in allowed set, confidence in [0,1]
	allowed := map[int]bool{0: true, 90: true, 180: true, 270: true}
	assert.True(t, allowed[res.Angle], "angle not in {0,90,180,270}: %d", res.Angle)
	assert.True(t, res.Confidence >= 0 && res.Confidence <= 1, "confidence out of range: %f", res.Confidence)
}

func TestNewClassifier_FallbackWhenModelMissing(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Enabled = true
	// Point to a non-existing model path, but with fallback enabled
	cfg.ModelPath = "/non/existent/model.onnx"
	cfg.UseHeuristicFallback = true
	cls, err := NewClassifier(cfg)
	require.NoError(t, err)
	require.NotNil(t, cls)
	// Heuristic should be active
	imgCfg := testutil.DefaultTestImageConfig()
	img, err := testutil.GenerateTextImage(imgCfg)
	require.NoError(t, err)
	_, err = cls.Predict(img)
	require.NoError(t, err)
}
