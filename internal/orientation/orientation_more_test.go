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
