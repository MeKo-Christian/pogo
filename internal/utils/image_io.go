package utils

import (
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"

	_ "golang.org/x/image/bmp"
)

// SupportedImageExtensions lists supported file extensions for loading.
var SupportedImageExtensions = []string{".jpg", ".jpeg", ".png", ".bmp"}

// IsSupportedImage reports whether the path has a supported image extension.
func IsSupportedImage(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	for _, s := range SupportedImageExtensions {
		if ext == s {
			return true
		}
	}
	return false
}

// ImageMetadata captures lightweight file and pixel information.
type ImageMetadata struct {
	Path        string
	Format      string
	SizeBytes   int64
	Width       int
	Height      int
	AspectRatio float64
}

// LoadImage opens and decodes an image file, returning the image and metadata.
func LoadImage(path string) (image.Image, ImageMetadata, error) {
	if path == "" {
		err := &ImageProcessingError{Operation: "load", Err: errors.New("empty path")}
		return nil, ImageMetadata{}, err
	}
	if !IsSupportedImage(path) {
		err := &ImageProcessingError{Operation: "load", Err: fmt.Errorf("unsupported format: %s", filepath.Ext(path))}
		return nil, ImageMetadata{}, err
	}

	f, err := os.Open(path) //nolint:gosec // G304: Reading user-provided image file path is expected
	if err != nil {
		err = &ImageProcessingError{Operation: "load", Err: err}
		return nil, ImageMetadata{}, err
	}
	defer func() {
		if err := f.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "Error closing image file: %v\n", err)
		}
	}()

	fi, statErr := f.Stat()
	if statErr != nil {
		return nil, ImageMetadata{}, &ImageProcessingError{Operation: "load", Err: statErr}
	}

	img, format, decErr := image.Decode(f)
	if decErr != nil {
		return nil, ImageMetadata{}, &ImageProcessingError{Operation: "decode", Err: decErr}
	}

	b := img.Bounds()
	meta := ImageMetadata{
		Path:        path,
		Format:      format,
		SizeBytes:   fi.Size(),
		Width:       b.Dx(),
		Height:      b.Dy(),
		AspectRatio: float64(b.Dx()) / float64(b.Dy()),
	}

	return img, meta, nil
}

// BatchLoadImages loads multiple images and returns results in-order.
// Any failed load returns a non-nil error in the corresponding entry.
type BatchImageResult struct {
	Path string
	Img  image.Image
	Meta ImageMetadata
	Err  error
}

func BatchLoadImages(paths []string) []BatchImageResult {
	results := make([]BatchImageResult, 0, len(paths))
	for _, p := range paths {
		img, meta, err := LoadImage(p)
		results = append(results, BatchImageResult{Path: p, Img: img, Meta: meta, Err: err})
	}
	return results
}

// ValidateImageConstraints checks dimensions against the provided constraints.
func ValidateImageConstraints(img image.Image, constraints ImageConstraints) error {
	if img == nil {
		return &ImageProcessingError{Operation: "validate", Err: errors.New("input image is nil")}
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w < constraints.MinWidth || h < constraints.MinHeight {
		return &ImageProcessingError{
			Operation: "validate",
			Err: fmt.Errorf(
				"image too small: %dx%d < %dx%d",
				w, h, constraints.MinWidth, constraints.MinHeight,
			),
		}
	}
	// No error if exceeding max — resizing pipeline will handle scale-down.
	return nil
}
