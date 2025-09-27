package pipeline

import (
	"encoding/json"
	"image"
	"image/color"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func sampleResult() *OCRImageResult {
	res := &OCRImageResult{Width: 200, Height: 100}
	res.Regions = []OCRRegionResult{
		{
			Polygon:       []struct{ X, Y float64 }{{10, 10}, {60, 10}, {60, 30}, {10, 30}},
			Box:           struct{ X, Y, W, H int }{X: 10, Y: 10, W: 50, H: 20},
			DetConfidence: 0.8,
			Text:          "Hello",
			RecConfidence: 0.9,
		},
		{
			Polygon:       []struct{ X, Y float64 }{{70, 40}, {140, 40}, {140, 60}, {70, 60}},
			Box:           struct{ X, Y, W, H int }{X: 70, Y: 40, W: 70, H: 20},
			DetConfidence: 0.7,
			Text:          "World",
			RecConfidence: 0.85,
		},
	}
	return res
}

func TestToJSONAndBack(t *testing.T) {
	res := sampleResult()
	s, err := ToJSONImage(res)
	require.NoError(t, err)
	require.NotEmpty(t, s)

	var back OCRImageResult
	require.NoError(t, json.Unmarshal([]byte(s), &back))
	assert.Equal(t, res.Width, back.Width)
	assert.Equal(t, res.Height, back.Height)
	assert.Len(t, back.Regions, 2)
}

func TestPlainTextAndCSV(t *testing.T) {
	res := sampleResult()
	txt, err := ToPlainTextImage(res)
	require.NoError(t, err)
	assert.Equal(t, "Hello\nWorld", txt)

	csv, err := ToCSVImage(res)
	require.NoError(t, err)
	// header + 2 rows
	assert.Contains(t, csv, "x,y,w,h,det_conf,text,rec_conf")
	assert.GreaterOrEqual(t, len(csv), len("x,y,w,h,det_conf,text,rec_conf\n"))
}

func TestValidateOCRImageResult(t *testing.T) {
	res := sampleResult()
	require.NoError(t, ValidateOCRImageResult(res))

	bad := *res
	bad.Width = -1
	assert.Error(t, ValidateOCRImageResult(&bad))
}

func TestRenderOverlay(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 200, 100))
	res := sampleResult()
	overlay := RenderOverlay(img, res, color.RGBA{255, 0, 0, 255}, color.RGBA{0, 255, 0, 255})
	require.NotNil(t, overlay)
	assert.Equal(t, img.Bounds().Dx(), overlay.Bounds().Dx())
	assert.Equal(t, img.Bounds().Dy(), overlay.Bounds().Dy())
}

func TestRenderOverlay_OrientationMapping(t *testing.T) {
	// Original image 80x50 (WxH)
	w, h := 80, 50
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	// Fake OCR result produced on a 90°-rotated working image: working dims are 50x80
	res := &OCRImageResult{}
	res.Orientation.Angle = 90
	res.Orientation.Applied = true
	// A box at (x=10,y=5,w=20,h=10) in rotated coords should map to original using inverse of CCW90
	// Inverse mapping for a point (x,y): x0 = W0-1 - y; y0 = x
	bx, by, bw, bh := 10, 5, 20, 10
	res.Regions = []OCRRegionResult{{
		Box: struct{ X, Y, W, H int }{X: bx, Y: by, W: bw, H: bh},
	}}
	ov := RenderOverlay(img, res, color.RGBA{255, 0, 0, 255}, nil)
	require.NotNil(t, ov)
	// Sample a few expected edge pixels to be red (rough check)
	// Compute expected mapped rect bounds
	x1 := float64(w - 1 - (by + bh))
	y1 := float64(bx)
	x2 := float64(w - 1 - by)
	// Round to ints
	ix1, iy1 := int(x1+0.5), int(y1+0.5)
	ix2 := int(x2 + 0.5)
	// Check pixels along top edge
	topY := iy1
	for x := ix1; x < ix2; x++ {
		r, g, b, a := ov.At(x, topY).RGBA()
		if a != 0 {
			assert.True(t, r > g && r > b, "expected red-ish at (%d,%d)", x, topY)
			break
		}
	}
}
