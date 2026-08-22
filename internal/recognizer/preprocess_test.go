package recognizer

import (
	"image"
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

	// PaddleOCR recognition input is (x/255 - 0.5) / 0.5, i.e. [-1, 1].
	// Compare against the plain [0,1] normalization of the same image.
	ref, _, _, err := utils.NormalizeImage(resized)
	require.NoError(t, err)
	require.Len(t, ten.Data, len(ref))
	sawNegative := false
	for i, v := range ten.Data {
		assert.GreaterOrEqual(t, v, float32(-1))
		assert.LessOrEqual(t, v, float32(1))
		assert.InDelta(t, (float64(ref[i])-0.5)/0.5, v, 1e-6)
		if v < 0 {
			sawNegative = true
		}
	}
	assert.True(t, sawNegative, "recognition input must reach negative values")
}

func TestBatchCropRegions(t *testing.T) {
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "Batch"
	cfg.Size = testutil.MediumSize
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	regions := []detector.DetectedRegion{
		{
			Polygon: []utils.Point{{X: 50, Y: 50}, {X: 150, Y: 50}, {X: 150, Y: 100}, {X: 50, Y: 100}},
			Box:     utils.NewBox(50, 50, 150, 100),
		},
		{
			Polygon: []utils.Point{{X: 200, Y: 120}, {X: 300, Y: 120}, {X: 300, Y: 180}, {X: 200, Y: 180}},
			Box:     utils.NewBox(200, 120, 300, 180),
		},
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
		Polygon: []utils.Point{
			{X: 10, Y: 10},
			{X: float64(b.Dx() - 10), Y: 10},
			{X: float64(b.Dx() - 10), Y: float64(b.Dy() - 10)},
			{X: 10, Y: float64(b.Dy() - 10)},
		},
		Box: utils.NewBox(10, 10, float64(b.Dx()-10), float64(b.Dy()-10)),
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

func TestNormalizeForRecognitionWithPool_BufferAndTensor(t *testing.T) {
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "Pool"
	cfg.Size = testutil.SmallSize
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)
	// Ensure a solid background; generating default already has white background.
	_ = color.White

	resized, outW, outH, err := ResizeForRecognition(img, 32, 0, 8)
	require.NoError(t, err)
	ten, buf, err := NormalizeForRecognitionWithPool(resized)
	require.NoError(t, err)
	require.NotNil(t, buf)
	// Tensor shape should match out dims
	require.Equal(t, int64(1), ten.Shape[0])
	require.Equal(t, int64(3), ten.Shape[1])
	require.Equal(t, int64(outH), ten.Shape[2])
	require.Equal(t, int64(outW), ten.Shape[3])
	// Data range in [-1,1] and identical to the non-pooled entry point.
	ref, err := NormalizeForRecognition(resized)
	require.NoError(t, err)
	require.Len(t, ten.Data, len(ref.Data))
	sawNegative := false
	for i, v := range ten.Data {
		assert.GreaterOrEqual(t, v, float32(-1))
		assert.LessOrEqual(t, v, float32(1))
		assert.InDelta(t, ref.Data[i], v, 0)
		if v < 0 {
			sawNegative = true
		}
	}
	assert.True(t, sawNegative, "recognition input must reach negative values")
}

func TestNormalizeForRecognitionWithPoolAnd_InvalidParams(t *testing.T) {
	cfg := testutil.DefaultTestImageConfig()
	cfg.Size = testutil.SmallSize
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	tests := []struct {
		name string
		img  image.Image
		p    utils.NormalizeParams
	}{
		{
			name: "negative channel count",
			img:  img,
			p:    utils.NormalizeParams{Channels: -1, Scale: 1, Std: [3]float32{1, 1, 1}},
		},
		{
			name: "zero channel count",
			img:  img,
			p:    utils.NormalizeParams{Channels: 0, Scale: 1, Std: [3]float32{1, 1, 1}},
		},
		{
			name: "nil image",
			img:  nil,
			p:    DefaultNormalizeParams(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Sizing the pooled buffer must not panic; the validation error wins.
			ten, buf, err := NormalizeForRecognitionWithPoolAnd(tt.img, tt.p)
			require.Error(t, err)
			assert.Nil(t, buf)
			assert.Nil(t, ten.Data)
		})
	}
}
