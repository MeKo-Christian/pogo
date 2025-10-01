package utils

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"

	"github.com/MeKo-Tech/pogo/internal/mempool"
	"github.com/disintegration/imaging"
)

// ImageProcessingError represents errors that can occur during image processing.
type ImageProcessingError struct {
	Operation string
	Err       error
}

func (e *ImageProcessingError) Error() string {
	return fmt.Sprintf("image processing error in %s: %v", e.Operation, e.Err)
}

// ImageConstraints defines the constraints for image processing.
type ImageConstraints struct {
	MaxWidth  int
	MaxHeight int
	MinWidth  int
	MinHeight int
}

// DefaultImageConstraints returns the default constraints for OCR processing.
func DefaultImageConstraints() ImageConstraints {
	return ImageConstraints{
		MaxWidth:  1024,
		MaxHeight: 1024,
		MinWidth:  32,
		MinHeight: 32,
	}
}

// ResizeImage resizes an image while preserving aspect ratio and ensuring dimensions are multiples of 32
// Uses Lanczos resampling for high quality.
func ResizeImage(img image.Image, constraints ImageConstraints) (image.Image, error) {
	if img == nil {
		return nil, &ImageProcessingError{Operation: "resize", Err: errors.New("input image is nil")}
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// Validate input dimensions
	if width < constraints.MinWidth || height < constraints.MinHeight {
		return nil, &ImageProcessingError{
			Operation: "resize",
			Err: fmt.Errorf("image dimensions %dx%d below minimum %dx%d",
				width, height, constraints.MinWidth, constraints.MinHeight),
		}
	}

	// Calculate scaling factor to fit within max dimensions while preserving aspect ratio
	scaleX := float64(constraints.MaxWidth) / float64(width)
	scaleY := float64(constraints.MaxHeight) / float64(height)
	scale := math.Min(scaleX, scaleY)

	// Only scale down, never up
	if scale >= 1.0 {
		scale = 1.0
	}

	// Calculate new dimensions
	newWidth := int(float64(width) * scale)
	newHeight := int(float64(height) * scale)

	// Ensure dimensions are multiples of 32 for ONNX model compatibility
	newWidth = (newWidth / 32) * 32
	newHeight = (newHeight / 32) * 32

	// Ensure we don't go below minimum dimensions
	if newWidth < constraints.MinWidth {
		newWidth = constraints.MinWidth
	}
	if newHeight < constraints.MinHeight {
		newHeight = constraints.MinHeight
	}

	// Resize using Lanczos filter for high quality
	resized := imaging.Resize(img, newWidth, newHeight, imaging.Lanczos)

	return resized, nil
}

// PadImage pads an image to target dimensions with centered placement
// Uses black background as required for OCR processing.
func PadImage(img image.Image, targetWidth, targetHeight int) (image.Image, error) {
	if img == nil {
		return nil, &ImageProcessingError{Operation: "pad", Err: errors.New("input image is nil")}
	}

	if targetWidth <= 0 || targetHeight <= 0 {
		return nil, &ImageProcessingError{
			Operation: "pad",
			Err:       fmt.Errorf("invalid target dimensions: %dx%d", targetWidth, targetHeight),
		}
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// If image is larger than target, crop center to target dimensions
	if width > targetWidth || height > targetHeight {
		// Determine crop rectangle centered
		cropW := targetWidth
		cropH := targetHeight
		if cropW > width {
			cropW = width
		}
		if cropH > height {
			cropH = height
		}
		x0 := (width - cropW) / 2
		y0 := (height - cropH) / 2
		rect := image.Rect(bounds.Min.X+x0, bounds.Min.Y+y0, bounds.Min.X+x0+cropW, bounds.Min.Y+y0+cropH)
		return imaging.Crop(img, rect), nil
	}

	// Create a new black background image
	background := imaging.New(targetWidth, targetHeight, color.Black)

	// Calculate position to center the original image
	x := (targetWidth - width) / 2
	y := (targetHeight - height) / 2

	// Ensure we don't go negative (for very large images)
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}

	// Paste the original image onto the background
	result := imaging.Paste(background, img, image.Pt(x, y))

	return result, nil
}

// NormalizeImage normalizes an image for OCR processing:
// - Converts to RGB (removes alpha channel)
// - Scales pixel values from 0-255 to 0-1
// - Reorders channels from RGB to NCHW format for ONNX.
func NormalizeImage(img image.Image) ([]float32, int, int, error) {
	if img == nil {
		return nil, 0, 0, &ImageProcessingError{Operation: "normalize", Err: errors.New("input image is nil")}
	}

	// Convert to NRGBA to ensure we have RGB channels
	nrgba := imaging.Clone(img)
	bounds := nrgba.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// Prepare NCHW tensor: [1, 3, height, width]
	// We use batch size 1 for single image processing
	tensor := make([]float32, 3*height*width)

	// Convert and normalize pixels
	for y := range height {
		for x := range width {
			r, g, b, _ := nrgba.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()

			// Convert from 0-65535 to 0-255, then to 0-1
			rFloat := float32(r>>8) / 255.0
			gFloat := float32(g>>8) / 255.0
			bFloat := float32(b>>8) / 255.0

			// Store in NCHW format: [batch=0, channel, y, x]
			// Channel 0: Red, Channel 1: Green, Channel 2: Blue
			rIdx := 0*height*width + y*width + x
			gIdx := 1*height*width + y*width + x
			bIdx := 2*height*width + y*width + x

			tensor[rIdx] = rFloat
			tensor[gIdx] = gFloat
			tensor[bIdx] = bFloat
		}
	}

	return tensor, width, height, nil
}

// NormalizeImageIntoBuffer normalizes an image into the provided buffer if it has
// sufficient capacity. If buf is nil or too small, a new buffer is allocated.
// Returns the slice used (length set appropriately) and image width/height.
func NormalizeImageIntoBuffer(img image.Image, buf []float32) ([]float32, int, int, error) {
	if img == nil {
		return nil, 0, 0, &ImageProcessingError{Operation: "normalize", Err: errors.New("input image is nil")}
	}

	nrgba := imaging.Clone(img)
	bounds := nrgba.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	if width <= 0 || height <= 0 {
		return nil, 0, 0, &ImageProcessingError{Operation: "normalize", Err: errors.New("invalid image dimensions")}
	}
	needed := 3 * width * height
	if buf == nil || cap(buf) < needed {
		buf = make([]float32, needed)
	}
	data := buf[:needed]
	for y := range height {
		for x := range width {
			r, g, b, _ := nrgba.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			rFloat := float32(r>>8) / 255.0
			gFloat := float32(g>>8) / 255.0
			bFloat := float32(b>>8) / 255.0
			idx := y*width + x
			data[idx] = rFloat
			data[width*height+idx] = gFloat
			data[2*width*height+idx] = bFloat
		}
	}
	return data, width, height, nil
}

// NormalizeImagePooled normalizes an image using memory pooling for the output buffer.
// The caller should return the buffer to the pool via mempool.PutFloat32 when done.
// Converts to RGB (removes alpha channel), scales pixel values from 0-255 to 0-1,
// and reorders channels from RGB to NCHW format for ONNX.
func NormalizeImagePooled(img image.Image) ([]float32, int, int, error) {
	if img == nil {
		return nil, 0, 0, &ImageProcessingError{Operation: "normalize", Err: errors.New("input image is nil")}
	}

	// Convert to NRGBA to ensure we have RGB channels
	nrgba := imaging.Clone(img)
	bounds := nrgba.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	if width <= 0 || height <= 0 {
		return nil, 0, 0, &ImageProcessingError{Operation: "normalize", Err: errors.New("invalid image dimensions")}
	}

	// Allocate buffer from pool
	needed := 3 * width * height
	tensor := mempool.GetFloat32(needed)

	// Convert and normalize pixels
	for y := range height {
		for x := range width {
			r, g, b, _ := nrgba.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()

			// Convert from 0-65535 to 0-255, then to 0-1
			rFloat := float32(r>>8) / 255.0
			gFloat := float32(g>>8) / 255.0
			bFloat := float32(b>>8) / 255.0

			// Store in NCHW format: [batch=0, channel, y, x]
			// Channel 0: Red, Channel 1: Green, Channel 2: Blue
			rIdx := 0*height*width + y*width + x
			gIdx := 1*height*width + y*width + x
			bIdx := 2*height*width + y*width + x

			tensor[rIdx] = rFloat
			tensor[gIdx] = gFloat
			tensor[bIdx] = bFloat
		}
	}

	return tensor, width, height, nil
}

// AssessImageQuality performs basic quality assessment of an image.
type ImageQuality struct {
	Width       int
	Height      int
	AspectRatio float64
	IsGrayscale bool
	HasAlpha    bool
	FileSize    int64 // if available
}

// AssessImageQuality analyzes basic image properties.
func AssessImageQuality(img image.Image) ImageQuality {
	if img == nil {
		return ImageQuality{}
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	aspectRatio := float64(width) / float64(height)
	isGrayscale, hasAlpha := analyzePixelProperties(img, bounds)

	return ImageQuality{
		Width:       width,
		Height:      height,
		AspectRatio: aspectRatio,
		IsGrayscale: isGrayscale,
		HasAlpha:    hasAlpha,
	}
}

// analyzePixelProperties checks if image is grayscale and has alpha channel.
func analyzePixelProperties(img image.Image, bounds image.Rectangle) (bool, bool) {
	isGrayscale := true
	hasAlpha := false

	for y := bounds.Min.Y; y < bounds.Max.Y && (isGrayscale || !hasAlpha); y++ {
		for x := bounds.Min.X; x < bounds.Max.X && (isGrayscale || !hasAlpha); x++ {
			r, g, b, a := img.At(x, y).RGBA()
			if a < 65535 {
				hasAlpha = true
			}
			if r != g || g != b {
				isGrayscale = false
			}
		}
	}

	return isGrayscale, hasAlpha
}
