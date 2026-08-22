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

// NormalizeParams describes how raw 8-bit pixel components are converted into
// tensor values. For every channel c the stored value is
//
//	(raw*Scale - Mean[c]) / Std[c]
//
// where raw is the 0..255 component of the pixel. This mirrors the PaddleOCR
// reference preprocessing, where Scale is typically 1/255.
//
// Identity parameters (Scale = 1/255, Mean = {0,0,0}, Std = {1,1,1}) yield the
// plain [0,1] scaling that all consumers used before normalization became
// configurable; see DefaultNormalizeParams.
type NormalizeParams struct {
	// Channels is the number of output channels of the NCHW tensor. Only 3
	// (RGB) is currently supported, because the source pixels are RGB.
	Channels int
	// Scale is applied to the raw 0..255 component before mean subtraction.
	Scale float32
	// Mean and Std are per-channel (R, G, B) offset and divisor.
	Mean, Std [3]float32
}

// DefaultNormalizeParams returns identity parameters that scale pixel values to
// the [0,1] range without any mean/std centring.
func DefaultNormalizeParams() NormalizeParams {
	return NormalizeParams{
		Channels: 3,
		Scale:    1.0 / 255.0,
		Mean:     [3]float32{0, 0, 0},
		Std:      [3]float32{1, 1, 1},
	}
}

// validate checks that the parameters can be applied to an RGB source image.
func (p NormalizeParams) validate() error {
	if p.Channels != normalizeChannels {
		return &ImageProcessingError{
			Operation: opNormalize,
			Err:       fmt.Errorf("unsupported channel count %d, only %d is supported", p.Channels, normalizeChannels),
		}
	}
	for i, s := range p.Std {
		if s == 0 {
			return &ImageProcessingError{
				Operation: opNormalize,
				Err:       fmt.Errorf("std for channel %d must not be zero", i),
			}
		}
	}
	return nil
}

const (
	// normalizeChannels is the number of channels produced by the normalizers.
	normalizeChannels = 3
	// opNormalize labels normalization errors.
	opNormalize = "normalize"
)

// normalizePrepare converts the input into an NRGBA copy and validates its
// dimensions. It is shared by all normalization entry points so that they all
// perform the same checks.
func normalizePrepare(img image.Image, p NormalizeParams) (*image.NRGBA, int, int, error) {
	if img == nil {
		return nil, 0, 0, &ImageProcessingError{Operation: opNormalize, Err: errors.New("input image is nil")}
	}
	if err := p.validate(); err != nil {
		return nil, 0, 0, err
	}

	// Convert to NRGBA to ensure we have RGB channels.
	nrgba := imaging.Clone(img)
	bounds := nrgba.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	if width <= 0 || height <= 0 {
		return nil, 0, 0, &ImageProcessingError{Operation: opNormalize, Err: errors.New("invalid image dimensions")}
	}
	return nrgba, width, height, nil
}

// normalizeInto writes the normalized pixels of nrgba into dst in NCHW order
// ([1, C, H, W], channel 0 = red, 1 = green, 2 = blue).
//
// Every element of dst is written unconditionally: pooled buffers are not
// zeroed and are rounded up to a size class, so skipping writes would leak
// stale values into the tensor.
//
// Note that At().RGBA() returns alpha-premultiplied 0..65535 components; the
// >>8 shift reduces them to the premultiplied 0..255 range, which is the
// behaviour every consumer has relied on so far.
func normalizeInto(nrgba *image.NRGBA, dst []float32, p NormalizeParams) {
	bounds := nrgba.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	plane := width * height

	// Precompute reciprocals so the inner loop avoids divisions.
	var invStd [normalizeChannels]float32
	for i := range invStd {
		invStd[i] = 1.0 / p.Std[i]
	}

	for y := range height {
		for x := range width {
			r, g, b, _ := nrgba.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			idx := y*width + x
			dst[idx] = (float32(r>>8)*p.Scale - p.Mean[0]) * invStd[0]
			dst[plane+idx] = (float32(g>>8)*p.Scale - p.Mean[1]) * invStd[1]
			dst[2*plane+idx] = (float32(b>>8)*p.Scale - p.Mean[2]) * invStd[2]
		}
	}
}

// NormalizeImage normalizes an image for OCR processing:
// - Converts to RGB (removes alpha channel)
// - Scales pixel values from 0-255 to 0-1
// - Reorders channels from RGB to NCHW format for ONNX.
func NormalizeImage(img image.Image) ([]float32, int, int, error) {
	return NormalizeImageWith(img, DefaultNormalizeParams())
}

// NormalizeImageWith normalizes an image into a freshly allocated NCHW tensor
// using the supplied parameters.
func NormalizeImageWith(img image.Image, p NormalizeParams) ([]float32, int, int, error) {
	nrgba, width, height, err := normalizePrepare(img, p)
	if err != nil {
		return nil, 0, 0, err
	}

	tensor := make([]float32, p.Channels*height*width)
	normalizeInto(nrgba, tensor, p)
	return tensor, width, height, nil
}

// NormalizeImageIntoBuffer normalizes an image into the provided buffer if it has
// sufficient capacity. If buf is nil or too small, a new buffer is allocated.
// Returns the slice used (length set appropriately) and image width/height.
func NormalizeImageIntoBuffer(img image.Image, buf []float32) ([]float32, int, int, error) {
	return NormalizeImageIntoBufferWith(img, buf, DefaultNormalizeParams())
}

// NormalizeImageIntoBufferWith normalizes an image into the provided buffer
// using the supplied parameters. If buf is nil or too small, a new buffer is
// allocated.
func NormalizeImageIntoBufferWith(
	img image.Image,
	buf []float32,
	p NormalizeParams,
) ([]float32, int, int, error) {
	nrgba, width, height, err := normalizePrepare(img, p)
	if err != nil {
		return nil, 0, 0, err
	}

	needed := p.Channels * width * height
	if buf == nil || cap(buf) < needed {
		buf = make([]float32, needed)
	}
	data := buf[:needed]
	normalizeInto(nrgba, data, p)
	return data, width, height, nil
}

// NormalizeImagePooled normalizes an image using memory pooling for the output buffer.
// The caller should return the buffer to the pool via mempool.PutFloat32 when done.
// Converts to RGB (removes alpha channel), scales pixel values from 0-255 to 0-1,
// and reorders channels from RGB to NCHW format for ONNX.
func NormalizeImagePooled(img image.Image) ([]float32, int, int, error) {
	return NormalizeImagePooledWith(img, DefaultNormalizeParams())
}

// NormalizeImagePooledWith normalizes an image into a pooled buffer using the
// supplied parameters. The caller should return the buffer to the pool via
// mempool.PutFloat32 when done.
func NormalizeImagePooledWith(img image.Image, p NormalizeParams) ([]float32, int, int, error) {
	nrgba, width, height, err := normalizePrepare(img, p)
	if err != nil {
		return nil, 0, 0, err
	}

	// Pooled buffers are neither zeroed nor exactly sized, so normalizeInto
	// must (and does) write every element of the returned slice.
	tensor := mempool.GetFloat32(p.Channels * width * height)
	normalizeInto(nrgba, tensor, p)
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
