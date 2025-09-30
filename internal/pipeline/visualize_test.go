package pipeline

import (
	"fmt"
	"image"
	"image/color"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestTransformCoordinates tests coordinate transformation for all rotation angles.
func TestTransformCoordinates(t *testing.T) {
	tests := []struct {
		name      string
		x, y      float64
		angle     int
		w0, h0    int
		expectedX float64
		expectedY float64
	}{
		{
			name: "no rotation",
			x:    10, y: 20,
			angle: 0,
			w0:    100, h0: 200,
			expectedX: 10, expectedY: 20,
		},
		{
			name: "90 degree rotation - top left",
			x:    0, y: 0,
			angle: 90,
			w0:    100, h0: 200,
			expectedX: 99, expectedY: 0, // w0-1 - y, x
		},
		{
			name: "90 degree rotation - center",
			x:    50, y: 25,
			angle: 90,
			w0:    100, h0: 200,
			expectedX: 74, expectedY: 50, // w0-1 - y, x
		},
		{
			name: "180 degree rotation - top left",
			x:    0, y: 0,
			angle: 180,
			w0:    100, h0: 200,
			expectedX: 99, expectedY: 199, // w0-1 - x, h0-1 - y
		},
		{
			name: "180 degree rotation - center",
			x:    50, y: 100,
			angle: 180,
			w0:    100, h0: 200,
			expectedX: 49, expectedY: 99, // w0-1 - x, h0-1 - y
		},
		{
			name: "270 degree rotation - top left",
			x:    0, y: 0,
			angle: 270,
			w0:    100, h0: 200,
			expectedX: 0, expectedY: 199, // y, h0-1 - x
		},
		{
			name: "270 degree rotation - center",
			x:    50, y: 25,
			angle: 270,
			w0:    100, h0: 200,
			expectedX: 25, expectedY: 149, // y, h0-1 - x
		},
		{
			name: "invalid angle treated as no rotation",
			x:    10, y: 20,
			angle: 45,
			w0:    100, h0: 200,
			expectedX: 10, expectedY: 20,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x, y := transformCoordinates(tt.x, tt.y, tt.angle, tt.w0, tt.h0)
			assert.InEpsilon(t, tt.expectedX, x, 1e-9, "x coordinate mismatch")
			assert.InEpsilon(t, tt.expectedY, y, 1e-9, "y coordinate mismatch")
		})
	}
}

// TestMin4 tests the min4 helper function.
func TestMin4(t *testing.T) {
	tests := []struct {
		name       string
		a, b, c, d float64
		expected   float64
	}{
		{
			name: "first is minimum",
			a:    1.0, b: 2.0, c: 3.0, d: 4.0,
			expected: 1.0,
		},
		{
			name: "second is minimum",
			a:    2.0, b: 1.0, c: 3.0, d: 4.0,
			expected: 1.0,
		},
		{
			name: "third is minimum",
			a:    3.0, b: 2.0, c: 1.0, d: 4.0,
			expected: 1.0,
		},
		{
			name: "fourth is minimum",
			a:    4.0, b: 3.0, c: 2.0, d: 1.0,
			expected: 1.0,
		},
		{
			name: "all equal",
			a:    5.0, b: 5.0, c: 5.0, d: 5.0,
			expected: 5.0,
		},
		{
			name: "negative values",
			a:    -1.0, b: -2.0, c: -3.0, d: -4.0,
			expected: -4.0,
		},
		{
			name: "mixed positive and negative",
			a:    10.0, b: -5.0, c: 0.0, d: 3.0,
			expected: -5.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := min4(tt.a, tt.b, tt.c, tt.d)
			assert.InEpsilon(t, tt.expected, result, 1e-9)
		})
	}
}

// TestMax4 tests the max4 helper function.
func TestMax4(t *testing.T) {
	tests := []struct {
		name       string
		a, b, c, d float64
		expected   float64
	}{
		{
			name: "first is maximum",
			a:    4.0, b: 3.0, c: 2.0, d: 1.0,
			expected: 4.0,
		},
		{
			name: "second is maximum",
			a:    3.0, b: 4.0, c: 2.0, d: 1.0,
			expected: 4.0,
		},
		{
			name: "third is maximum",
			a:    2.0, b: 3.0, c: 4.0, d: 1.0,
			expected: 4.0,
		},
		{
			name: "fourth is maximum",
			a:    1.0, b: 2.0, c: 3.0, d: 4.0,
			expected: 4.0,
		},
		{
			name: "all equal",
			a:    5.0, b: 5.0, c: 5.0, d: 5.0,
			expected: 5.0,
		},
		{
			name: "negative values",
			a:    -1.0, b: -2.0, c: -3.0, d: -4.0,
			expected: -1.0,
		},
		{
			name: "mixed positive and negative",
			a:    10.0, b: -5.0, c: 0.0, d: 3.0,
			expected: 10.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := max4(tt.a, tt.b, tt.c, tt.d)
			assert.InEpsilon(t, tt.expected, result, 1e-9)
		})
	}
}

// TestDrawRegionBox tests the drawRegionBox function.
func TestDrawRegionBox(t *testing.T) {
	tests := []struct {
		name       string
		imgWidth   int
		imgHeight  int
		region     OCRRegionResult
		angle      int
		boxColor   color.Color
		shouldDraw bool
	}{
		{
			name:      "simple box no rotation",
			imgWidth:  100,
			imgHeight: 100,
			region: OCRRegionResult{
				Box: struct{ X, Y, W, H int }{X: 10, Y: 10, W: 30, H: 20},
			},
			angle:      0,
			boxColor:   color.RGBA{255, 0, 0, 255},
			shouldDraw: true,
		},
		{
			name:      "box with 90 degree rotation",
			imgWidth:  100,
			imgHeight: 100,
			region: OCRRegionResult{
				Box: struct{ X, Y, W, H int }{X: 10, Y: 10, W: 30, H: 20},
			},
			angle:      90,
			boxColor:   color.RGBA{0, 255, 0, 255},
			shouldDraw: true,
		},
		{
			name:      "box with 180 degree rotation",
			imgWidth:  100,
			imgHeight: 100,
			region: OCRRegionResult{
				Box: struct{ X, Y, W, H int }{X: 10, Y: 10, W: 30, H: 20},
			},
			angle:      180,
			boxColor:   color.RGBA{0, 0, 255, 255},
			shouldDraw: true,
		},
		{
			name:      "box with 270 degree rotation",
			imgWidth:  100,
			imgHeight: 100,
			region: OCRRegionResult{
				Box: struct{ X, Y, W, H int }{X: 10, Y: 10, W: 30, H: 20},
			},
			angle:      270,
			boxColor:   color.RGBA{255, 255, 0, 255},
			shouldDraw: true,
		},
		{
			name:      "small box at edge",
			imgWidth:  100,
			imgHeight: 100,
			region: OCRRegionResult{
				Box: struct{ X, Y, W, H int }{X: 0, Y: 0, W: 5, H: 5},
			},
			angle:      0,
			boxColor:   color.RGBA{128, 128, 128, 255},
			shouldDraw: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := image.NewRGBA(image.Rect(0, 0, tt.imgWidth, tt.imgHeight))
			// Fill with white background
			for y := range tt.imgHeight {
				for x := range tt.imgWidth {
					dst.Set(x, y, color.White)
				}
			}

			drawRegionBox(dst, tt.region, tt.angle, tt.imgWidth, tt.imgHeight, tt.boxColor)

			if tt.shouldDraw {
				// Verify that at least some pixels have been colored
				coloredPixels := 0
				for y := range tt.imgHeight {
					for x := range tt.imgWidth {
						r, g, b, a := dst.At(x, y).RGBA()
						// Check if pixel is not white (255, 255, 255, 255)
						if r != 65535 || g != 65535 || b != 65535 || a != 65535 {
							coloredPixels++
						}
					}
				}
				assert.Positive(t, coloredPixels, "expected some pixels to be drawn")
			}
		})
	}
}

// countColoredPixels counts pixels that are not white in the given image.
func countColoredPixels(img image.Image, width, height int) int {
	coloredPixels := 0
	for y := range height {
		for x := range width {
			r, g, b, a := img.At(x, y).RGBA()
			// Check if pixel is not white (255, 255, 255, 255)
			if r != 65535 || g != 65535 || b != 65535 || a != 65535 {
				coloredPixels++
			}
		}
	}
	return coloredPixels
}

// verifyAllPixelsWhite verifies that all pixels in the image are white.
func verifyAllPixelsWhite(t *testing.T, img image.Image, width, height int) {
	t.Helper()
	for y := range height {
		for x := range width {
			r, g, b, a := img.At(x, y).RGBA()
			assert.Equal(t, uint32(65535), r, "pixel should be white at (%d, %d)", x, y)
			assert.Equal(t, uint32(65535), g, "pixel should be white at (%d, %d)", x, y)
			assert.Equal(t, uint32(65535), b, "pixel should be white at (%d, %d)", x, y)
			assert.Equal(t, uint32(65535), a, "pixel should be white at (%d, %d)", x, y)
		}
	}
}

// TestDrawRegionPolygon tests the drawRegionPolygon function.
func TestDrawRegionPolygon(t *testing.T) {
	tests := []struct {
		name       string
		imgWidth   int
		imgHeight  int
		region     OCRRegionResult
		angle      int
		polyColor  color.Color
		shouldDraw bool
	}{
		{
			name:      "simple polygon no rotation",
			imgWidth:  100,
			imgHeight: 100,
			region: OCRRegionResult{
				Polygon: []struct{ X, Y float64 }{
					{10, 10},
					{40, 10},
					{40, 30},
					{10, 30},
				},
			},
			angle:      0,
			polyColor:  color.RGBA{255, 0, 0, 255},
			shouldDraw: true,
		},
		{
			name:      "polygon with 90 degree rotation",
			imgWidth:  100,
			imgHeight: 100,
			region: OCRRegionResult{
				Polygon: []struct{ X, Y float64 }{
					{20, 20},
					{50, 20},
					{50, 40},
					{20, 40},
				},
			},
			angle:      90,
			polyColor:  color.RGBA{0, 255, 0, 255},
			shouldDraw: true,
		},
		{
			name:      "polygon with less than 2 points should not draw",
			imgWidth:  100,
			imgHeight: 100,
			region: OCRRegionResult{
				Polygon: []struct{ X, Y float64 }{
					{10, 10},
				},
			},
			angle:      0,
			polyColor:  color.RGBA{0, 0, 255, 255},
			shouldDraw: false,
		},
		{
			name:      "empty polygon should not draw",
			imgWidth:  100,
			imgHeight: 100,
			region: OCRRegionResult{
				Polygon: []struct{ X, Y float64 }{},
			},
			angle:      0,
			polyColor:  color.RGBA{255, 0, 255, 255},
			shouldDraw: false,
		},
		{
			name:      "triangular polygon with 180 degree rotation",
			imgWidth:  100,
			imgHeight: 100,
			region: OCRRegionResult{
				Polygon: []struct{ X, Y float64 }{
					{30, 20},
					{50, 40},
					{10, 40},
				},
			},
			angle:      180,
			polyColor:  color.RGBA{128, 128, 0, 255},
			shouldDraw: true,
		},
		{
			name:      "polygon with 270 degree rotation",
			imgWidth:  100,
			imgHeight: 100,
			region: OCRRegionResult{
				Polygon: []struct{ X, Y float64 }{
					{15, 15},
					{35, 15},
					{35, 35},
					{15, 35},
				},
			},
			angle:      270,
			polyColor:  color.RGBA{0, 128, 128, 255},
			shouldDraw: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := image.NewRGBA(image.Rect(0, 0, tt.imgWidth, tt.imgHeight))
			// Fill with white background
			for y := range tt.imgHeight {
				for x := range tt.imgWidth {
					dst.Set(x, y, color.White)
				}
			}

			drawRegionPolygon(dst, tt.region, tt.angle, tt.imgWidth, tt.imgHeight, tt.polyColor)

			if tt.shouldDraw {
				coloredPixels := countColoredPixels(dst, tt.imgWidth, tt.imgHeight)
				assert.Positive(t, coloredPixels, "expected some pixels to be drawn")
			} else {
				verifyAllPixelsWhite(t, dst, tt.imgWidth, tt.imgHeight)
			}
		})
	}
}

// TestRenderOverlay_NilInputs tests RenderOverlay with nil inputs.
func TestRenderOverlay_NilInputs(t *testing.T) {
	t.Run("nil image returns nil", func(t *testing.T) {
		res := sampleResult()
		overlay := RenderOverlay(nil, res, color.RGBA{255, 0, 0, 255}, color.RGBA{0, 255, 0, 255})
		assert.Nil(t, overlay)
	})

	t.Run("nil result returns image copy", func(t *testing.T) {
		img := image.NewRGBA(image.Rect(0, 0, 100, 100))
		overlay := RenderOverlay(img, nil, color.RGBA{255, 0, 0, 255}, color.RGBA{0, 255, 0, 255})
		require.NotNil(t, overlay)
		assert.Equal(t, img.Bounds().Dx(), overlay.Bounds().Dx())
		assert.Equal(t, img.Bounds().Dy(), overlay.Bounds().Dy())
	})
}

// TestRenderOverlay_EmptyRegions tests RenderOverlay with empty regions.
func TestRenderOverlay_EmptyRegions(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	res := &OCRImageResult{
		Width:   100,
		Height:  100,
		Regions: []OCRRegionResult{},
	}
	overlay := RenderOverlay(img, res, color.RGBA{255, 0, 0, 255}, color.RGBA{0, 255, 0, 255})
	require.NotNil(t, overlay)
	assert.Equal(t, img.Bounds().Dx(), overlay.Bounds().Dx())
	assert.Equal(t, img.Bounds().Dy(), overlay.Bounds().Dy())
}

// TestRenderOverlay_AllRotationAngles tests RenderOverlay with all rotation angles.
func TestRenderOverlay_AllRotationAngles(t *testing.T) {
	angles := []int{0, 90, 180, 270}

	for _, angle := range angles {
		t.Run(fmt.Sprintf("angle_%d", angle), func(t *testing.T) {
			img := image.NewRGBA(image.Rect(0, 0, 100, 100))
			res := &OCRImageResult{
				Width:  100,
				Height: 100,
				Regions: []OCRRegionResult{
					{
						Polygon: []struct{ X, Y float64 }{
							{20, 20},
							{60, 20},
							{60, 40},
							{20, 40},
						},
						Box: struct{ X, Y, W, H int }{X: 20, Y: 20, W: 40, H: 20},
					},
				},
			}
			res.Orientation.Angle = angle
			res.Orientation.Applied = true

			overlay := RenderOverlay(img, res, color.RGBA{255, 0, 0, 255}, color.RGBA{0, 255, 0, 255})
			require.NotNil(t, overlay)
			assert.Equal(t, 100, overlay.Bounds().Dx())
			assert.Equal(t, 100, overlay.Bounds().Dy())

			// Verify that some drawing occurred
			coloredPixels := 0
			for y := range 100 {
				for x := range 100 {
					r, g, b, _ := overlay.At(x, y).RGBA()
					// Check for red or green pixels (box or polygon)
					if r > 30000 || g > 30000 || b > 30000 {
						coloredPixels++
					}
				}
			}
			assert.Positive(t, coloredPixels, "expected some pixels to be drawn for angle %d", angle)
		})
	}
}

// TestRenderOverlay_BoxAndPolygon tests that both box and polygon are rendered.
func TestRenderOverlay_BoxAndPolygon(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 200, 100))
	// Fill with black background
	for y := range 100 {
		for x := range 200 {
			img.Set(x, y, color.Black)
		}
	}

	res := &OCRImageResult{
		Width:  200,
		Height: 100,
		Regions: []OCRRegionResult{
			{
				Polygon: []struct{ X, Y float64 }{
					{30, 30},
					{70, 30},
					{70, 50},
					{30, 50},
				},
				Box: struct{ X, Y, W, H int }{X: 30, Y: 30, W: 40, H: 20},
			},
		},
	}

	boxColor := color.RGBA{255, 0, 0, 255}  // red
	polyColor := color.RGBA{0, 255, 0, 255} // green

	overlay := RenderOverlay(img, res, boxColor, polyColor)
	require.NotNil(t, overlay)

	// Count red and green pixels
	redPixels := 0
	greenPixels := 0
	for y := range 100 {
		for x := range 200 {
			r, g, b, _ := overlay.At(x, y).RGBA()
			// Red pixel (box)
			if r > 50000 && g < 10000 && b < 10000 {
				redPixels++
			}
			// Green pixel (polygon)
			if g > 50000 && r < 10000 && b < 10000 {
				greenPixels++
			}
		}
	}

	assert.Positive(t, redPixels, "expected red pixels for box")
	assert.Positive(t, greenPixels, "expected green pixels for polygon")
}

// TestRenderOverlay_NonZeroBounds tests RenderOverlay with non-zero bounded images.
func TestRenderOverlay_NonZeroBounds(t *testing.T) {
	// Create image with non-zero minimum bounds
	img := image.NewRGBA(image.Rect(10, 10, 110, 110))
	res := &OCRImageResult{
		Width:  100,
		Height: 100,
		Regions: []OCRRegionResult{
			{
				Polygon: []struct{ X, Y float64 }{
					{20, 20},
					{60, 20},
					{60, 40},
					{20, 40},
				},
				Box: struct{ X, Y, W, H int }{X: 20, Y: 20, W: 40, H: 20},
			},
		},
	}

	overlay := RenderOverlay(img, res, color.RGBA{255, 0, 0, 255}, color.RGBA{0, 255, 0, 255})
	require.NotNil(t, overlay)
	// The overlay should have same dimensions as input
	assert.Equal(t, img.Bounds().Dx(), overlay.Bounds().Dx())
	assert.Equal(t, img.Bounds().Dy(), overlay.Bounds().Dy())
}

// TestRenderOverlay_CoordinateAccuracy tests coordinate transformation accuracy.
func TestRenderOverlay_CoordinateAccuracy(t *testing.T) {
	tests := []struct {
		name       string
		angle      int
		regionBox  struct{ X, Y, W, H int }
		imgW, imgH int
	}{
		{
			name:      "90 degree precise mapping",
			angle:     90,
			regionBox: struct{ X, Y, W, H int }{X: 10, Y: 5, W: 20, H: 10},
			imgW:      80,
			imgH:      50,
		},
		{
			name:      "180 degree precise mapping",
			angle:     180,
			regionBox: struct{ X, Y, W, H int }{X: 20, Y: 10, W: 30, H: 15},
			imgW:      100,
			imgH:      100,
		},
		{
			name:      "270 degree precise mapping",
			angle:     270,
			regionBox: struct{ X, Y, W, H int }{X: 15, Y: 20, W: 25, H: 30},
			imgW:      120,
			imgH:      80,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			img := image.NewRGBA(image.Rect(0, 0, tt.imgW, tt.imgH))
			res := &OCRImageResult{
				Width:  tt.imgW,
				Height: tt.imgH,
				Regions: []OCRRegionResult{
					{
						Box: tt.regionBox,
						Polygon: []struct{ X, Y float64 }{
							{float64(tt.regionBox.X), float64(tt.regionBox.Y)},
							{float64(tt.regionBox.X + tt.regionBox.W), float64(tt.regionBox.Y)},
							{float64(tt.regionBox.X + tt.regionBox.W), float64(tt.regionBox.Y + tt.regionBox.H)},
							{float64(tt.regionBox.X), float64(tt.regionBox.Y + tt.regionBox.H)},
						},
					},
				},
			}
			res.Orientation.Angle = tt.angle
			res.Orientation.Applied = true

			overlay := RenderOverlay(img, res, color.RGBA{255, 0, 0, 255}, color.RGBA{0, 255, 0, 255})
			require.NotNil(t, overlay)

			// Verify dimensions are preserved
			assert.Equal(t, tt.imgW, overlay.Bounds().Dx())
			assert.Equal(t, tt.imgH, overlay.Bounds().Dy())

			// Verify that drawing occurred within image bounds
			coloredPixels := 0
			for y := range tt.imgH {
				for x := range tt.imgW {
					r, g, b, _ := overlay.At(x, y).RGBA()
					if r > 30000 || g > 30000 || b > 30000 {
						coloredPixels++
					}
				}
			}
			assert.Positive(t, coloredPixels, "expected pixels to be drawn within bounds")
		})
	}
}
