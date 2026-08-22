package utils

import (
	"image"
	"image/color"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/mempool"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// solidImage builds an opaque RGBA image filled with a single color.
func solidImage(w, h int, c color.RGBA) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.Set(x, y, c)
		}
	}
	return img
}

func imagenetParams() NormalizeParams {
	return NormalizeParams{
		Channels: 3,
		Scale:    1.0 / 255.0,
		Mean:     [3]float32{0.485, 0.456, 0.406},
		Std:      [3]float32{0.229, 0.224, 0.225},
	}
}

func minusOneToOneParams() NormalizeParams {
	return NormalizeParams{
		Channels: 3,
		Scale:    1.0 / 255.0,
		Mean:     [3]float32{0.5, 0.5, 0.5},
		Std:      [3]float32{0.5, 0.5, 0.5},
	}
}

// TestNormalizeImageWith_KnownPixel drives a known pixel through several
// parameterizations and checks the exact resulting tensor values.
func TestNormalizeImageWith_KnownPixel(t *testing.T) {
	const (
		rRaw = 128.0
		gRaw = 64.0
		bRaw = 192.0
	)
	src := solidImage(4, 3, color.RGBA{R: 128, G: 64, B: 192, A: 255})

	tests := []struct {
		name             string
		params           NormalizeParams
		wantR            float64
		wantG            float64
		wantB            float64
		wantNegativeSeen bool
	}{
		{
			name:   "identity 0..1",
			params: DefaultNormalizeParams(),
			wantR:  rRaw / 255.0,
			wantG:  gRaw / 255.0,
			wantB:  bRaw / 255.0,
		},
		{
			name:             "minus one to one",
			params:           minusOneToOneParams(),
			wantR:            (rRaw/255.0 - 0.5) / 0.5,
			wantG:            (gRaw/255.0 - 0.5) / 0.5,
			wantB:            (bRaw/255.0 - 0.5) / 0.5,
			wantNegativeSeen: true,
		},
		{
			name:             "imagenet",
			params:           imagenetParams(),
			wantR:            (rRaw/255.0 - 0.485) / 0.229,
			wantG:            (gRaw/255.0 - 0.456) / 0.224,
			wantB:            (bRaw/255.0 - 0.406) / 0.225,
			wantNegativeSeen: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, w, h, err := NormalizeImageWith(src, tt.params)
			require.NoError(t, err)
			assert.Equal(t, 4, w)
			assert.Equal(t, 3, h)
			require.Len(t, data, 3*w*h)

			plane := w * h
			for i := range plane {
				assert.InDelta(t, tt.wantR, data[i], 1e-6)
				assert.InDelta(t, tt.wantG, data[plane+i], 1e-6)
				assert.InDelta(t, tt.wantB, data[2*plane+i], 1e-6)
			}

			if tt.wantNegativeSeen {
				negative := false
				for _, v := range data {
					if v < 0 {
						negative = true
						break
					}
				}
				assert.True(t, negative, "expected centred parameters to produce negative values")
			}
		})
	}
}

// TestNormalizeEntryPoints_Identical proves that the three exported entry points
// differ only in allocation strategy and never in the values they compute.
func TestNormalizeEntryPoints_Identical(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 9, 7))
	for y := range 7 {
		for x := range 9 {
			src.Set(x, y, color.RGBA{R: uint8(x * 20), G: uint8(y * 30), B: uint8(x*7 + y*3), A: 255})
		}
	}

	paramSets := map[string]NormalizeParams{
		"identity": DefaultNormalizeParams(),
		"imagenet": imagenetParams(),
		"pm1":      minusOneToOneParams(),
	}

	for name, p := range paramSets {
		t.Run(name, func(t *testing.T) {
			plain, w1, h1, err := NormalizeImageWith(src, p)
			require.NoError(t, err)

			buffered, w2, h2, err := NormalizeImageIntoBufferWith(src, nil, p)
			require.NoError(t, err)

			pooled, w3, h3, err := NormalizeImagePooledWith(src, p)
			require.NoError(t, err)
			defer mempool.PutFloat32(pooled)

			assert.Equal(t, w1, w2)
			assert.Equal(t, w1, w3)
			assert.Equal(t, h1, h2)
			assert.Equal(t, h1, h3)
			require.Len(t, buffered, len(plain))
			require.Len(t, pooled, len(plain))

			// Exact equality: the entry points must differ only in allocation.
			for i := range plain {
				assert.InDelta(t, plain[i], buffered[i], 0, "buffered mismatch at %d", i)
				assert.InDelta(t, plain[i], pooled[i], 0, "pooled mismatch at %d", i)
			}
		})
	}
}

// TestNormalizeImagePooled covers the pooled entry point directly, including
// the case where the pool hands back a dirty (non-zeroed, oversized) buffer.
func TestNormalizeImagePooled(t *testing.T) {
	src := solidImage(8, 5, color.RGBA{R: 10, G: 200, B: 90, A: 255})
	needed := 3 * 8 * 5

	// Dirty the pool so a reused buffer carries stale values.
	dirty := mempool.GetFloat32(needed)
	for i := range dirty {
		dirty[i] = -12345
	}
	mempool.PutFloat32(dirty)

	data, w, h, err := NormalizeImagePooled(src)
	require.NoError(t, err)
	defer mempool.PutFloat32(data)

	assert.Equal(t, 8, w)
	assert.Equal(t, 5, h)
	require.Len(t, data, needed)

	plane := w * h
	for i := range plane {
		assert.InDelta(t, 10.0/255.0, data[i], 1e-6)
		assert.InDelta(t, 200.0/255.0, data[plane+i], 1e-6)
		assert.InDelta(t, 90.0/255.0, data[2*plane+i], 1e-6)
	}
}

func TestNormalizeImagePooled_NilImage(t *testing.T) {
	data, w, h, err := NormalizeImagePooled(nil)
	require.Error(t, err)
	assert.Nil(t, data)
	assert.Zero(t, w)
	assert.Zero(t, h)
	assert.Contains(t, err.Error(), "input image is nil")
}

// TestNormalize_DimensionGuard verifies that all three entry points reject
// zero-sized images (the guard used to be missing from NormalizeImage).
func TestNormalize_DimensionGuard(t *testing.T) {
	empty := image.NewRGBA(image.Rect(0, 0, 0, 0))

	const want = "invalid image dimensions"

	data, w, h, err := NormalizeImage(empty)
	requireNormalizeError(t, want, data, w, h, err)

	data, w, h, err = NormalizeImageIntoBuffer(empty, nil)
	requireNormalizeError(t, want, data, w, h, err)

	data, w, h, err = NormalizeImagePooled(empty)
	requireNormalizeError(t, want, data, w, h, err)
}

func TestNormalizeParams_Validation(t *testing.T) {
	src := solidImage(2, 2, color.RGBA{R: 1, G: 2, B: 3, A: 255})

	p := DefaultNormalizeParams()
	p.Channels = 1
	data, w, h, err := NormalizeImageWith(src, p)
	requireNormalizeError(t, "unsupported channel count", data, w, h, err)

	p = DefaultNormalizeParams()
	p.Std = [3]float32{1, 0, 1}
	data, w, h, err = NormalizeImageWith(src, p)
	requireNormalizeError(t, "must not be zero", data, w, h, err)
}

// requireNormalizeError asserts that a normalizer result is a failure whose
// message contains want and that no partial output was returned.
func requireNormalizeError(t *testing.T, want string, data []float32, w, h int, err error) {
	t.Helper()
	require.Error(t, err)
	assert.Nil(t, data)
	assert.Zero(t, w)
	assert.Zero(t, h)
	assert.Contains(t, err.Error(), want)
}
