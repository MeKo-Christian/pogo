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

func TestCropRegionImage_BoxAndPolygon(t *testing.T) {
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "Hello"
	cfg.Size = testutil.SmallSize
	cfg.Background = color.White
	cfg.Foreground = color.Black
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	// Define a region roughly in the center of the image
	x1, y1 := 80.0, 90.0
	x2, y2 := 240.0, 150.0
	poly := []utils.Point{{X: x1, Y: y1}, {X: x2, Y: y1}, {X: x2, Y: y2}, {X: x1, Y: y2}}
	region := detector.DetectedRegion{
		Polygon:    poly,
		Box:        utils.NewBox(x1, y1, x2, y2),
		Confidence: 0.9,
	}

	patch, rotated, err := CropRegionImage(img, region, true)
	require.NoError(t, err)
	require.NotNil(t, patch)
	assert.False(t, rotated)

	pb := patch.Bounds()
	// Expect approximately the same size as defined by box
	assert.InDelta(t, int(x2-x1), pb.Dx(), float64(pb.Dx())*0.2)
	assert.InDelta(t, int(y2-y1), pb.Dy(), float64(pb.Dy())*0.2)
}

func TestCropRegionImage_RotateIfVertical(t *testing.T) {
	// Create a tall rectangle region to trigger rotation
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "Vertical"
	cfg.Size = testutil.SmallSize
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	x1, y1 := 150.0, 40.0
	x2, y2 := 190.0, 200.0 // tall region
	region := detector.DetectedRegion{
		Polygon: []utils.Point{{X: x1, Y: y1}, {X: x2, Y: y1}, {X: x2, Y: y2}, {X: x1, Y: y2}},
		Box:     utils.NewBox(x1, y1, x2, y2),
	}
	patch, rotated, err := CropRegionImage(img, region, true)
	require.NoError(t, err)
	require.NotNil(t, patch)
	assert.True(t, rotated)
	pb := patch.Bounds()
	assert.Greater(t, pb.Dx(), pb.Dy())
}

func TestResizeForRecognition(t *testing.T) {
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "Resize"
	cfg.Size = testutil.SmallSize
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	targetH := 32
	maxW := 256
	padMult := 8

	resized, outW, outH, err := ResizeForRecognition(img, targetH, maxW, padMult)
	require.NoError(t, err)
	require.NotNil(t, resized)
	assert.Equal(t, targetH, outH)
	// outW must be multiple of padMult and <= maxW
	assert.Equal(t, 0, outW%padMult)
	if maxW > 0 {
		assert.LessOrEqual(t, outW, maxW)
	}
}

func TestNormalizeForRecognition(t *testing.T) {
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "Norm"
	cfg.Size = testutil.SmallSize
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	resized, outW, outH, err := ResizeForRecognition(img, 32, 256, 8)
	require.NoError(t, err)
	require.NotNil(t, resized)

	ten, err := NormalizeForRecognition(resized)
	require.NoError(t, err)
	require.Equal(t, int64(1), ten.Shape[0])
	require.Equal(t, int64(3), ten.Shape[1])
	require.Equal(t, int64(outH), ten.Shape[2])
	require.Equal(t, int64(outW), ten.Shape[3])
	for _, v := range ten.Data {
		assert.GreaterOrEqual(t, v, float32(0))
		assert.LessOrEqual(t, v, float32(1))
	}
}

func TestBatchCropRegions(t *testing.T) {
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "Batch"
	cfg.Size = testutil.MediumSize
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	regions := []detector.DetectedRegion{
		{Polygon: []utils.Point{{X: 50, Y: 50}, {X: 150, Y: 50}, {X: 150, Y: 100}, {X: 50, Y: 100}}, Box: utils.NewBox(50, 50, 150, 100)},
		{Polygon: []utils.Point{{X: 200, Y: 120}, {X: 300, Y: 120}, {X: 300, Y: 180}, {X: 200, Y: 180}}, Box: utils.NewBox(200, 120, 300, 180)},
	}

	patches, rotated, err := BatchCropRegions(img, regions, true)
	require.NoError(t, err)
	require.Len(t, patches, len(regions))
	require.Len(t, rotated, len(regions))
	for _, p := range patches {
		b := p.Bounds()
		assert.Positive(t, b.Dx())
		assert.Positive(t, b.Dy())
	}
}

func TestCropRegionImageWithOrienter(t *testing.T) {
	// Create an image with text rotated 90 degrees; the heuristic orienter should request rotation
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "Rotated"
	cfg.Size = testutil.MediumSize
	cfg.Rotation = 90
	cfg.Background = color.White
	cfg.Foreground = color.Black
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	// Region covering most of the image
	b := img.Bounds()
	region := detector.DetectedRegion{
		Polygon: []utils.Point{{X: 10, Y: 10}, {X: float64(b.Dx() - 10), Y: 10}, {X: float64(b.Dx() - 10), Y: float64(b.Dy() - 10)}, {X: 10, Y: float64(b.Dy() - 10)}},
		Box:     utils.NewBox(10, 10, float64(b.Dx()-10), float64(b.Dy()-10)),
	}
	// Heuristic-only classifier
	oCfg := orientation.DefaultTextLineConfig()
	oCfg.Enabled = false
	oCfg.UseHeuristicFallback = true
	oCfg.ConfidenceThreshold = 0.1
	cls, err := orientation.NewClassifier(oCfg)
	require.NoError(t, err)
	patch, rotated, err := CropRegionImageWithOrienter(img, region, cls, false)
	require.NoError(t, err)
	require.NotNil(t, patch)
	// Should be rotated for vertical text
	assert.True(t, rotated)
}
