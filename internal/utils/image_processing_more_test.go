package utils

import (
	"image"
	"image/color"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Additional coverage for preprocessing utilities beyond image_processing_test.go

func TestPadImage_CropsWhenLarger(t *testing.T) {
	// 300x200 image padded/cropped to 100x100 should return 100x100
	img := image.NewRGBA(image.Rect(0, 0, 300, 200))
	// Fill to avoid zero image edge cases
	for y := range 200 {
		for x := range 300 {
			img.Set(x, y, color.RGBA{R: 10, G: 20, B: 30, A: 255})
		}
	}

	got, err := PadImage(img, 100, 100)
	require.NoError(t, err)
	require.NotNil(t, got)
	b := got.Bounds()
	assert.Equal(t, 100, b.Dx())
	assert.Equal(t, 100, b.Dy())
}

func TestNormalizeImage_NonRGBAInputs(t *testing.T) {
	// Gray image
	gray := image.NewGray(image.Rect(0, 0, 32, 16))
	for y := range 16 {
		for x := range 32 {
			gray.SetGray(x, y, color.Gray{Y: 128})
		}
	}
	dataG, wG, hG, err := NormalizeImage(gray)
	require.NoError(t, err)
	assert.Equal(t, 32, wG)
	assert.Equal(t, 16, hG)
	// Values should be within [0,1]
	for _, v := range dataG {
		assert.GreaterOrEqual(t, v, float32(0))
		assert.LessOrEqual(t, v, float32(1))
	}

	// NRGBA image (note: color.RGBA().RGBA() returns premultiplied components)
	nrgba := image.NewNRGBA(image.Rect(0, 0, 10, 10))
	for y := range 10 {
		for x := range 10 {
			nrgba.SetNRGBA(x, y, color.NRGBA{R: 64, G: 128, B: 192, A: 200})
		}
	}
	dataN, wN, hN, err := NormalizeImage(nrgba)
	require.NoError(t, err)
	assert.Equal(t, 10, wN)
	assert.Equal(t, 10, hN)
	// Check first pixel channel order NCHW (R then G then B planes)
	// Values are premultiplied by alpha when obtained via RGBA(),
	// so expected ~= base * (alpha/255).
	r := dataN[0]
	g := dataN[wN*hN]
	b := dataN[2*wN*hN]
	a := 200.0 / 255.0
	assert.InDelta(t, (64.0/255.0)*a, r, 0.02)
	assert.InDelta(t, (128.0/255.0)*a, g, 0.02)
	assert.InDelta(t, (192.0/255.0)*a, b, 0.02)
}

func TestNormalizeImageIntoBuffer_ReuseAndMatch(t *testing.T) {
	base := image.NewRGBA(image.Rect(0, 0, 20, 10))
	for y := range 10 {
		for x := range 20 {
			base.Set(x, y, color.RGBA{R: 200, G: 100, B: 50, A: 255})
		}
	}

	// Baseline using Allocate path
	ref, w, h, err := NormalizeImage(base)
	require.NoError(t, err)

	// Insufficient buffer -> should allocate internally
	small := make([]float32, 0, 10) // too small on purpose
	out1, w1, h1, err := NormalizeImageIntoBuffer(base, small)
	require.NoError(t, err)
	assert.Equal(t, w, w1)
	assert.Equal(t, h, h1)
	// Data should match baseline
	require.Len(t, out1, len(ref))
	for i := range ref {
		assert.InDelta(t, ref[i], out1[i], 1e-6)
	}

	// Sufficient buffer -> should reuse provided slice capacity
	need := 3 * w * h
	buf := make([]float32, 0, need)
	out2, w2, h2, err := NormalizeImageIntoBuffer(base, buf)
	require.NoError(t, err)
	assert.Equal(t, w, w2)
	assert.Equal(t, h, h2)
	assert.Len(t, out2, need)
	// Confirm same backing array (reuse) by growing and checking capacity
	assert.Equal(t, cap(buf), cap(out2))
}

func TestResizeImage_CustomMaxConstraints(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 4000, 3000))
	cons := ImageConstraints{MaxWidth: 960, MaxHeight: 1024, MinWidth: 32, MinHeight: 32}
	out, err := ResizeImage(img, cons)
	require.NoError(t, err)
	b := out.Bounds()
	// Within max constraints and multiples of 32
	assert.LessOrEqual(t, b.Dx(), 960)
	assert.LessOrEqual(t, b.Dy(), 1024)
	assert.Equal(t, 0, b.Dx()%32)
	assert.Equal(t, 0, b.Dy()%32)
}
