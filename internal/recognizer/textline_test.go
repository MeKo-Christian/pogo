package recognizer

import (
	"image/color"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/orientation"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCropRegionImageWithOrienter_Heuristic(t *testing.T) {
	// Create an image with 90-degree rotated text
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "Rotated Text"
	cfg.Rotation = 90
	cfg.Background = color.White
	cfg.Foreground = color.Black
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	// Region covering the full image
	b := img.Bounds()
	region := detector.DetectedRegion{
		Polygon: []utils.Point{
			{X: 0, Y: 0},
			{X: float64(b.Dx()), Y: 0},
			{X: float64(b.Dx()), Y: float64(b.Dy())},
			{X: 0, Y: float64(b.Dy())},
		},
		Box: utils.NewBox(0, 0, float64(b.Dx()), float64(b.Dy())),
	}

	// Use heuristic-only classifier
	cls, err := orientation.NewClassifier(orientation.Config{Enabled: false, UseHeuristicFallback: true})
	require.NoError(t, err)

	patch, rotated, err := CropRegionImageWithOrienter(img, region, cls, false)
	require.NoError(t, err)
	require.NotNil(t, patch)
	assert.True(t, rotated)
	pb := patch.Bounds()
	assert.Greater(t, pb.Dx(), pb.Dy())
}
