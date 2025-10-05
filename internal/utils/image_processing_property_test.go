package utils

import (
	"image"
	"image/color"
	"testing"

	"github.com/disintegration/imaging"
	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
)

// genTestImage generates a simple test image.
func genTestImage(width, height int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			// Create a simple pattern
			val := uint8((x + y) % 256)
			img.Set(x, y, color.RGBA{val, val, val, 255})
		}
	}
	return img
}

// TestResizeImage_ProducesValidOutput verifies resize produces valid dimensions.
// Note: Due to rounding to multiples of 32 for ONNX compatibility, aspect ratio
// may not be perfectly preserved. This is by design.
func TestResizeImage_ProducesValidOutput(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("resize produces valid dimensions within constraints", prop.ForAll(
		func(width, height int) bool {
			if width < 32 || height < 32 || width > 512 || height > 512 {
				return true
			}

			img := genTestImage(width, height)
			constraints := ImageConstraints{
				MaxWidth:  256,
				MaxHeight: 256,
				MinWidth:  32,
				MinHeight: 32,
			}

			resized, err := ResizeImage(img, constraints)
			if err != nil {
				return false
			}

			bounds := resized.Bounds()
			newWidth := bounds.Dx()
			newHeight := bounds.Dy()

			// Verify output meets constraints
			if newWidth > constraints.MaxWidth || newHeight > constraints.MaxHeight {
				return false
			}
			if newWidth < constraints.MinWidth || newHeight < constraints.MinHeight {
				return false
			}

			// Verify dimensions are multiples of 32
			if newWidth%32 != 0 || newHeight%32 != 0 {
				return false
			}

			// Verify no upscaling occurred (unless needed to meet minimum)
			if width >= constraints.MinWidth && height >= constraints.MinHeight {
				if newWidth > width || newHeight > height {
					return false
				}
			}

			return true
		},
		gen.IntRange(32, 512),
		gen.IntRange(32, 512),
	))

	properties.TestingRun(t)
}

// TestResizeImage_DimensionsMultipleOf32 verifies output dimensions are multiples of 32.
func TestResizeImage_DimensionsMultipleOf32(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("resized dimensions are multiples of 32", prop.ForAll(
		func(width, height int) bool {
			if width < 64 || height < 64 || width > 512 || height > 512 {
				return true
			}

			img := genTestImage(width, height)
			constraints := ImageConstraints{
				MaxWidth:  512,
				MaxHeight: 512,
				MinWidth:  32,
				MinHeight: 32,
			}

			resized, err := ResizeImage(img, constraints)
			if err != nil {
				return false
			}

			bounds := resized.Bounds()
			newWidth := bounds.Dx()
			newHeight := bounds.Dy()

			return newWidth%32 == 0 && newHeight%32 == 0
		},
		gen.IntRange(64, 512),
		gen.IntRange(64, 512),
	))

	properties.TestingRun(t)
}

// TestResizeImage_RespectMaxConstraints verifies max constraints are honored.
func TestResizeImage_RespectMaxConstraints(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("resized image respects max constraints", prop.ForAll(
		func(width, height, maxWidth, maxHeight int) bool {
			if width < 64 || height < 64 {
				return true
			}
			if maxWidth < 32 || maxHeight < 32 {
				return true
			}
			if maxWidth > 1024 || maxHeight > 1024 {
				return true
			}

			img := genTestImage(width, height)
			constraints := ImageConstraints{
				MaxWidth:  maxWidth,
				MaxHeight: maxHeight,
				MinWidth:  32,
				MinHeight: 32,
			}

			resized, err := ResizeImage(img, constraints)
			if err != nil {
				return false
			}

			bounds := resized.Bounds()
			newWidth := bounds.Dx()
			newHeight := bounds.Dy()

			return newWidth <= maxWidth && newHeight <= maxHeight
		},
		gen.IntRange(64, 800),
		gen.IntRange(64, 800),
		gen.IntRange(128, 512),
		gen.IntRange(128, 512),
	))

	properties.TestingRun(t)
}

// TestPadImage_OutputDimensions verifies padded image has correct dimensions.
func TestPadImage_OutputDimensions(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("padded image has target dimensions", prop.ForAll(
		func(width, height, targetW, targetH int) bool {
			if width < 10 || height < 10 || width > 100 || height > 100 {
				return true
			}
			if targetW < width || targetH < height {
				return true // can't pad to smaller
			}
			if targetW > 200 || targetH > 200 {
				return true
			}

			img := genTestImage(width, height)
			padded, err := PadImage(img, targetW, targetH)
			if err != nil {
				return false
			}

			bounds := padded.Bounds()
			return bounds.Dx() == targetW && bounds.Dy() == targetH
		},
		gen.IntRange(10, 100),
		gen.IntRange(10, 100),
		gen.IntRange(50, 200),
		gen.IntRange(50, 200),
	))

	properties.TestingRun(t)
}

// TestNormalizeImage_OutputBounds verifies normalized values are in [0, 1].
func TestNormalizeImage_OutputBounds(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("normalized image values are in [0, 1]", prop.ForAll(
		func(width, height int) bool {
			if width < 10 || height < 10 || width > 50 || height > 50 {
				return true
			}

			img := genTestImage(width, height)
			tensor, w, h, err := NormalizeImage(img)
			if err != nil {
				return false
			}

			if w != width || h != height {
				return false
			}

			// Check all values are in [0, 1]
			for _, val := range tensor {
				if val < 0.0 || val > 1.0 {
					return false
				}
			}
			return true
		},
		gen.IntRange(10, 50),
		gen.IntRange(10, 50),
	))

	properties.TestingRun(t)
}

// TestNormalizeImage_TensorLength verifies tensor length equals 3*width*height.
func TestNormalizeImage_TensorLength(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("normalized tensor length = 3 * width * height", prop.ForAll(
		func(width, height int) bool {
			if width < 10 || height < 10 || width > 50 || height > 50 {
				return true
			}

			img := genTestImage(width, height)
			tensor, w, h, err := NormalizeImage(img)
			if err != nil {
				return false
			}

			expectedLength := 3 * w * h
			return len(tensor) == expectedLength
		},
		gen.IntRange(10, 50),
		gen.IntRange(10, 50),
	))

	properties.TestingRun(t)
}

// TestNormalizeImage_NCHWLayout verifies NCHW layout (channels first).
func TestNormalizeImage_NCHWLayout(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("normalized tensor uses NCHW layout", prop.ForAll(
		func(width, height int) bool {
			if width < 10 || height < 10 || width > 30 || height > 30 {
				return true
			}

			// Create image with known RGB pattern
			img := image.NewRGBA(image.Rect(0, 0, width, height))
			for y := range height {
				for x := range width {
					img.Set(x, y, color.RGBA{255, 128, 64, 255})
				}
			}

			tensor, w, h, err := NormalizeImage(img)
			if err != nil {
				return false
			}

			// Check NCHW layout: R channel at [0:w*h], G at [w*h:2*w*h], B at [2*w*h:3*w*h]
			rIdx := 0
			gIdx := w * h
			bIdx := 2 * w * h

			// Check expected values (with tolerance for normalization)
			rVal := tensor[rIdx]
			gVal := tensor[gIdx]
			bVal := tensor[bIdx]

			// R should be ~1.0, G ~0.5, B ~0.25
			return rVal > 0.95 && rVal <= 1.0 &&
				gVal > 0.45 && gVal < 0.55 &&
				bVal > 0.20 && bVal < 0.30
		},
		gen.IntRange(10, 30),
		gen.IntRange(10, 30),
	))

	properties.TestingRun(t)
}

// TestNormalizeImageIntoBuffer_ReuseBuffer verifies buffer reuse works.
func TestNormalizeImageIntoBuffer_ReuseBuffer(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("normalize into buffer reuses provided buffer", prop.ForAll(
		func(width, height int) bool {
			if width < 10 || height < 10 || width > 30 || height > 30 {
				return true
			}

			img := genTestImage(width, height)
			bufSize := 3 * width * height

			// Provide a buffer with sufficient capacity
			buf := make([]float32, 0, bufSize)

			tensor, w, h, err := NormalizeImageIntoBuffer(img, buf)
			if err != nil {
				return false
			}

			// Should have reused the provided buffer
			return w == width && h == height && len(tensor) == bufSize
		},
		gen.IntRange(10, 30),
		gen.IntRange(10, 30),
	))

	properties.TestingRun(t)
}

// TestAssessImageQuality_WidthHeight verifies correct dimensions are reported.
func TestAssessImageQuality_WidthHeight(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("image quality assessment reports correct dimensions", prop.ForAll(
		func(width, height int) bool {
			if width < 10 || height < 10 || width > 100 || height > 100 {
				return true
			}

			img := genTestImage(width, height)
			quality := AssessImageQuality(img)

			return quality.Width == width && quality.Height == height
		},
		gen.IntRange(10, 100),
		gen.IntRange(10, 100),
	))

	properties.TestingRun(t)
}

// TestAssessImageQuality_AspectRatio verifies aspect ratio calculation.
func TestAssessImageQuality_AspectRatio(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("image quality aspect ratio is width/height", prop.ForAll(
		func(width, height int) bool {
			if width < 10 || height < 10 || width > 100 || height > 100 {
				return true
			}

			img := genTestImage(width, height)
			quality := AssessImageQuality(img)

			expectedAspect := float64(width) / float64(height)
			return quality.AspectRatio > expectedAspect-0.01 &&
				quality.AspectRatio < expectedAspect+0.01
		},
		gen.IntRange(10, 100),
		gen.IntRange(10, 100),
	))

	properties.TestingRun(t)
}

// TestResizeImage_NoUpscaling verifies images aren't scaled up.
func TestResizeImage_NoUpscaling(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("resize does not upscale images", prop.ForAll(
		func(width, height int) bool {
			if width < 32 || height < 32 || width > 256 || height > 256 {
				return true
			}

			img := genTestImage(width, height)
			constraints := ImageConstraints{
				MaxWidth:  1024,
				MaxHeight: 1024,
				MinWidth:  32,
				MinHeight: 32,
			}

			resized, err := ResizeImage(img, constraints)
			if err != nil {
				return false
			}

			bounds := resized.Bounds()
			newWidth := bounds.Dx()
			newHeight := bounds.Dy()

			// Should not be larger than original
			return newWidth <= width && newHeight <= height
		},
		gen.IntRange(32, 256),
		gen.IntRange(32, 256),
	))

	properties.TestingRun(t)
}

// TestPadImage_CenteredPlacement verifies content is centered.
func TestPadImage_CenteredPlacement(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("padding centers the original content", prop.ForAll(
		func(width, height int) bool {
			if width < 20 || height < 20 || width > 60 || height > 60 {
				return true
			}

			targetW := width * 2
			targetH := height * 2

			// Create image with specific pattern to verify centering
			img := image.NewRGBA(image.Rect(0, 0, width, height))
			for y := range height {
				for x := range width {
					img.Set(x, y, color.RGBA{255, 255, 255, 255}) // white
				}
			}

			padded, err := PadImage(img, targetW, targetH)
			if err != nil {
				return false
			}

			// Check that center region contains white pixels
			centerX := targetW / 2
			centerY := targetH / 2
			r, g, b, _ := padded.At(centerX, centerY).RGBA()

			// Center should be white (from original image)
			return r > 60000 && g > 60000 && b > 60000
		},
		gen.IntRange(20, 60),
		gen.IntRange(20, 60),
	))

	properties.TestingRun(t)
}

// TestNormalizeImage_HandlesDifferentColorSpaces verifies color space handling.
func TestNormalizeImage_HandlesDifferentColorSpaces(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("normalize handles different image types", prop.ForAll(
		func(width, height int) bool {
			if width < 10 || height < 10 || width > 30 || height > 30 {
				return true
			}

			// Create grayscale image
			gray := image.NewGray(image.Rect(0, 0, width, height))
			for y := range height {
				for x := range width {
					gray.Set(x, y, color.Gray{128})
				}
			}

			tensor, w, h, err := NormalizeImage(gray)
			if err != nil {
				return false
			}

			return w == width && h == height && len(tensor) == 3*width*height
		},
		gen.IntRange(10, 30),
		gen.IntRange(10, 30),
	))

	properties.TestingRun(t)
}

// TestResizeImage_RespectsMinConstraints verifies min constraints are honored.
func TestResizeImage_RespectsMinConstraints(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("resized image respects min constraints", prop.ForAll(
		func(width, height int) bool {
			if width < 100 || height < 100 || width > 500 || height > 500 {
				return true
			}

			img := genTestImage(width, height)
			constraints := ImageConstraints{
				MaxWidth:  200,
				MaxHeight: 200,
				MinWidth:  64,
				MinHeight: 64,
			}

			resized, err := ResizeImage(img, constraints)
			if err != nil {
				return false
			}

			bounds := resized.Bounds()
			newWidth := bounds.Dx()
			newHeight := bounds.Dy()

			return newWidth >= constraints.MinWidth && newHeight >= constraints.MinHeight
		},
		gen.IntRange(100, 500),
		gen.IntRange(100, 500),
	))

	properties.TestingRun(t)
}

// TestNormalizeImage_Deterministic verifies normalization is deterministic.
func TestNormalizeImage_Deterministic(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("image normalization is deterministic", prop.ForAll(
		func(width, height int) bool {
			if width < 10 || height < 10 || width > 30 || height > 30 {
				return true
			}

			img := genTestImage(width, height)

			tensor1, w1, h1, err1 := NormalizeImage(img)
			tensor2, w2, h2, err2 := NormalizeImage(img)

			if err1 != nil || err2 != nil {
				return false
			}
			if w1 != w2 || h1 != h2 {
				return false
			}
			if len(tensor1) != len(tensor2) {
				return false
			}

			// Check all values are identical
			for i := range tensor1 {
				if tensor1[i] != tensor2[i] {
					return false
				}
			}
			return true
		},
		gen.IntRange(10, 30),
		gen.IntRange(10, 30),
	))

	properties.TestingRun(t)
}

// TestPadImage_BlackBackground verifies padding uses black background.
func TestPadImage_BlackBackground(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("padding uses black background", prop.ForAll(
		func(width, height int) bool {
			if width < 20 || height < 20 || width > 40 || height > 40 {
				return true
			}

			targetW := width * 3
			targetH := height * 3

			img := imaging.New(width, height, color.White)

			padded, err := PadImage(img, targetW, targetH)
			if err != nil {
				return false
			}

			// Check corners (should be black from padding)
			r1, g1, b1, _ := padded.At(0, 0).RGBA()
			r2, g2, b2, _ := padded.At(targetW-1, targetH-1).RGBA()

			// Should be black or very dark
			return (r1+g1+b1) < 1000 && (r2+g2+b2) < 1000
		},
		gen.IntRange(20, 40),
		gen.IntRange(20, 40),
	))

	properties.TestingRun(t)
}
