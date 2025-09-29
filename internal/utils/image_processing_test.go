package utils

import (
	"errors"
	"image"
	"image/color"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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

func TestResizeImage(t *testing.T) {
	tests := []struct {
		name        string
		inputWidth  int
		inputHeight int
		constraints ImageConstraints
		expectError bool
		checkFunc   func(t *testing.T, result image.Image, originalWidth, originalHeight int)
	}{
		{
			name:        "normal resize down",
			inputWidth:  1000,
			inputHeight: 800,
			constraints: DefaultImageConstraints(),
			expectError: false,
			checkFunc: func(t *testing.T, result image.Image, origW, origH int) {
				t.Helper()
				bounds := result.Bounds()
				width := bounds.Dx()
				height := bounds.Dy()

				// Should be within constraints
				assert.LessOrEqual(t, width, 1024)
				assert.LessOrEqual(t, height, 1024)
				assert.GreaterOrEqual(t, width, 32)
				assert.GreaterOrEqual(t, height, 32)

				// Should be multiple of 32
				assert.Zero(t, width%32)
				assert.Zero(t, height%32)

				// Aspect ratio should be preserved approximately
				origRatio := float64(origW) / float64(origH)
				newRatio := float64(width) / float64(height)
				assert.InDelta(t, origRatio, newRatio, 0.1)
			},
		},
		{
			name:        "no resize needed",
			inputWidth:  512,
			inputHeight: 512,
			constraints: DefaultImageConstraints(),
			expectError: false,
			checkFunc: func(t *testing.T, result image.Image, origW, origH int) {
				t.Helper()
				bounds := result.Bounds()
				assert.Equal(t, 512, bounds.Dx())
				assert.Equal(t, 512, bounds.Dy())
			},
		},
		{
			name:        "very small image",
			inputWidth:  10,
			inputHeight: 10,
			constraints: DefaultImageConstraints(),
			expectError: true,
		},
		{
			name:        "nil image",
			inputWidth:  0,
			inputHeight: 0,
			constraints: DefaultImageConstraints(),
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var img image.Image
			if tt.inputWidth > 0 && tt.inputHeight > 0 {
				// Create a test image
				rgbaImg := image.NewRGBA(image.Rect(0, 0, tt.inputWidth, tt.inputHeight))
				// Fill with some color
				for y := range tt.inputHeight {
					for x := range tt.inputWidth {
						rgbaImg.Set(x, y, color.RGBA{255, 0, 0, 255})
					}
				}
				img = rgbaImg
			}

			result, err := ResizeImage(img, tt.constraints)

			if tt.expectError {
				require.Error(t, err)
				assert.Nil(t, result)
			} else {
				require.NoError(t, err)
				assert.NotNil(t, result)
				if tt.checkFunc != nil {
					tt.checkFunc(t, result, tt.inputWidth, tt.inputHeight)
				}
			}
		})
	}
}

func TestPadImage(t *testing.T) {
	tests := []struct {
		name         string
		inputWidth   int
		inputHeight  int
		targetWidth  int
		targetHeight int
		expectError  bool
	}{
		{
			name:         "pad smaller image",
			inputWidth:   100,
			inputHeight:  100,
			targetWidth:  200,
			targetHeight: 200,
			expectError:  false,
		},
		{
			name:         "no padding needed",
			inputWidth:   200,
			inputHeight:  200,
			targetWidth:  200,
			targetHeight: 200,
			expectError:  false,
		},
		{
			name:         "larger image no padding",
			inputWidth:   300,
			inputHeight:  300,
			targetWidth:  200,
			targetHeight: 200,
			expectError:  false,
		},
		{
			name:         "invalid target dimensions",
			inputWidth:   100,
			inputHeight:  100,
			targetWidth:  0,
			targetHeight: 200,
			expectError:  true,
		},
		{
			name:         "nil image",
			inputWidth:   0,
			inputHeight:  0,
			targetWidth:  200,
			targetHeight: 200,
			expectError:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var img image.Image
			if tt.inputWidth > 0 && tt.inputHeight > 0 {
				rgbaImg := image.NewRGBA(image.Rect(0, 0, tt.inputWidth, tt.inputHeight))
				// Fill with white
				for y := range tt.inputHeight {
					for x := range tt.inputWidth {
						rgbaImg.Set(x, y, color.RGBA{255, 255, 255, 255})
					}
				}
				img = rgbaImg
			}

			result, err := PadImage(img, tt.targetWidth, tt.targetHeight)

			if tt.expectError {
				require.Error(t, err)
				assert.Nil(t, result)
			} else {
				require.NoError(t, err)
				assert.NotNil(t, result)

				bounds := result.Bounds()
				assert.Equal(t, tt.targetWidth, bounds.Dx())
				assert.Equal(t, tt.targetHeight, bounds.Dy())

				// Check that the original image is centered (only when strictly smaller)
				if tt.inputWidth < tt.targetWidth && tt.inputHeight < tt.targetHeight {
					// Check corners are black
					corner1 := result.At(0, 0)
					corner2 := result.At(tt.targetWidth-1, 0)
					corner3 := result.At(0, tt.targetHeight-1)
					corner4 := result.At(tt.targetWidth-1, tt.targetHeight-1)

					r1, g1, b1, _ := corner1.RGBA()
					r2, g2, b2, _ := corner2.RGBA()
					r3, g3, b3, _ := corner3.RGBA()
					r4, g4, b4, _ := corner4.RGBA()

					assert.Equal(t, uint32(0), r1)
					assert.Equal(t, uint32(0), g1)
					assert.Equal(t, uint32(0), b1)
					assert.Equal(t, uint32(0), r2)
					assert.Equal(t, uint32(0), g2)
					assert.Equal(t, uint32(0), b2)
					assert.Equal(t, uint32(0), r3)
					assert.Equal(t, uint32(0), g3)
					assert.Equal(t, uint32(0), b3)
					assert.Equal(t, uint32(0), r4)
					assert.Equal(t, uint32(0), g4)
					assert.Equal(t, uint32(0), b4)
				}
			}
		})
	}
}

func TestNormalizeImage(t *testing.T) {
	tests := []struct {
		name        string
		width       int
		height      int
		expectError bool
	}{
		{
			name:        "normal image",
			width:       100,
			height:      100,
			expectError: false,
		},
		{
			name:        "small image",
			width:       32,
			height:      32,
			expectError: false,
		},
		{
			name:        "nil image",
			width:       0,
			height:      0,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var img image.Image
			if tt.width > 0 && tt.height > 0 {
				rgbaImg := image.NewRGBA(image.Rect(0, 0, tt.width, tt.height))
				// Fill with known color
				for y := range tt.height {
					for x := range tt.width {
						rgbaImg.Set(x, y, color.RGBA{128, 64, 192, 255})
					}
				}
				img = rgbaImg
			}

			tensor, width, height, err := NormalizeImage(img)

			if tt.expectError {
				require.Error(t, err)
				assert.Nil(t, tensor)
				assert.Zero(t, width)
				assert.Zero(t, height)
			} else {
				require.NoError(t, err)
				assert.NotNil(t, tensor)
				assert.Equal(t, tt.width, width)
				assert.Equal(t, tt.height, height)

				// Check tensor size: 3 channels * height * width
				expectedSize := 3 * tt.height * tt.width
				assert.Len(t, tensor, expectedSize)

				// Check that values are in [0, 1] range
				for _, val := range tensor {
					assert.GreaterOrEqual(t, val, float32(0.0))
					assert.LessOrEqual(t, val, float32(1.0))
				}

				// Check specific values (128/255 ≈ 0.502, 64/255 ≈ 0.251, 192/255 ≈ 0.753)
				// Due to RGBA conversion, values might be slightly different
				rVal := tensor[0]                    // First red value
				gVal := tensor[tt.height*tt.width]   // First green value
				bVal := tensor[2*tt.height*tt.width] // First blue value

				assert.InDelta(t, 128.0/255.0, rVal, 0.01)
				assert.InDelta(t, 64.0/255.0, gVal, 0.01)
				assert.InDelta(t, 192.0/255.0, bVal, 0.01)
			}
		})
	}
}

func TestAssessImageQuality(t *testing.T) {
	tests := []struct {
		name      string
		createImg func() image.Image
		expected  ImageQuality
	}{
		{
			name: "RGB image",
			createImg: func() image.Image {
				rgbaImg := image.NewRGBA(image.Rect(0, 0, 100, 100))
				for y := range 100 {
					for x := range 100 {
						rgbaImg.Set(x, y, color.RGBA{255, 0, 0, 255})
					}
				}
				return rgbaImg
			},
			expected: ImageQuality{
				Width:       100,
				Height:      100,
				AspectRatio: 1.0,
				IsGrayscale: false,
				HasAlpha:    false,
			},
		},
		{
			name: "grayscale image",
			createImg: func() image.Image {
				grayImg := image.NewGray(image.Rect(0, 0, 50, 50))
				for y := range 50 {
					for x := range 50 {
						grayImg.SetGray(x, y, color.Gray{128})
					}
				}
				return grayImg
			},
			expected: ImageQuality{
				Width:       50,
				Height:      50,
				AspectRatio: 1.0,
				IsGrayscale: true,
				HasAlpha:    false,
			},
		},
		{
			name: "image with alpha",
			createImg: func() image.Image {
				rgbaImg := image.NewRGBA(image.Rect(0, 0, 32, 32))
				for y := range 32 {
					for x := range 32 {
						rgbaImg.Set(x, y, color.RGBA{255, 255, 255, 128})
					}
				}
				return rgbaImg
			},
			expected: ImageQuality{
				Width:       32,
				Height:      32,
				AspectRatio: 1.0,
				IsGrayscale: true,
				HasAlpha:    true,
			},
		},
		{
			name: "nil image",
			createImg: func() image.Image {
				return nil
			},
			expected: ImageQuality{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			img := tt.createImg()
			result := AssessImageQuality(img)

			assert.Equal(t, tt.expected.Width, result.Width)
			assert.Equal(t, tt.expected.Height, result.Height)
			assert.Equal(t, tt.expected.IsGrayscale, result.IsGrayscale)
			assert.Equal(t, tt.expected.HasAlpha, result.HasAlpha)
			if tt.expected.Width > 0 && tt.expected.Height > 0 {
				assert.InDelta(t, tt.expected.AspectRatio, result.AspectRatio, 0.001)
			}
		})
	}
}

func TestImageProcessingError(t *testing.T) {
	err := &ImageProcessingError{
		Operation: "test",
		Err:       errors.New("test error"),
	}

	expectedMsg := "image processing error in test: test error"
	assert.Equal(t, expectedMsg, err.Error())
}

func TestDefaultImageConstraints(t *testing.T) {
	constraints := DefaultImageConstraints()

	assert.Equal(t, 1024, constraints.MaxWidth)
	assert.Equal(t, 1024, constraints.MaxHeight)
	assert.Equal(t, 32, constraints.MinWidth)
	assert.Equal(t, 32, constraints.MinHeight)
}

// Benchmark tests.
func BenchmarkResizeImage(b *testing.B) {
	img := image.NewRGBA(image.Rect(0, 0, 1000, 1000))
	constraints := DefaultImageConstraints()

	b.ResetTimer()
	for range b.N {
		_, _ = ResizeImage(img, constraints)
	}
}

func BenchmarkPadImage(b *testing.B) {
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))

	b.ResetTimer()
	for range b.N {
		_, _ = PadImage(img, 200, 200)
	}
}

func BenchmarkNormalizeImage(b *testing.B) {
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))

	b.ResetTimer()
	for range b.N {
		_, _, _, _ = NormalizeImage(img)
	}
}

func BenchmarkAssessImageQuality(b *testing.B) {
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))

	b.ResetTimer()
	for range b.N {
		_ = AssessImageQuality(img)
	}
}
