package orientation

import (
	"image/color"
	"path/filepath"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConfigUpdateModelPath(t *testing.T) {
	tmp := t.TempDir()
	c := DefaultConfig()
	old := c.ModelPath
	c.UpdateModelPath(tmp)
	assert.NotEqual(t, old, c.ModelPath)
	assert.True(t, filepath.IsAbs(c.ModelPath))
	assert.GreaterOrEqual(t, len(c.ModelPath), len(tmp))
	// Should keep the filename under the new dir
	assert.Equal(t, models.GetLayoutModelPath(tmp, filepath.Base(old)), c.ModelPath)
}

func TestHeuristicOrientation_LandscapePrefers0(t *testing.T) {
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "Landscape"
	cfg.Background = color.White
	cfg.Foreground = color.Black
	cfg.Size = testutil.MediumSize
	// no rotation -> likely 0
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)
	cls, err := NewClassifier(Config{Enabled: false, UseHeuristicFallback: true, ConfidenceThreshold: 0.0})
	require.NoError(t, err)
	res, err := cls.Predict(img)
	require.NoError(t, err)
	assert.Contains(t, []int{0, 90, 180, 270}, res.Angle)
	// Heuristic should return a valid angle with a sane confidence bound
	assert.GreaterOrEqual(t, res.Confidence, 0.0)
	assert.LessOrEqual(t, res.Confidence, 1.0)
}

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
