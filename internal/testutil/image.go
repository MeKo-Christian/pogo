package testutil

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/disintegration/imaging"
	"github.com/stretchr/testify/require"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

// ImageSize represents common image dimensions.
type ImageSize struct {
	Width  int
	Height int
}

var (
	// Common test image sizes.
	SmallSize  = ImageSize{320, 240}
	MediumSize = ImageSize{640, 480}
	LargeSize  = ImageSize{1024, 768}
)

// TestImageConfig holds configuration for generating test images.
type TestImageConfig struct {
	Text       string
	Size       ImageSize
	Background color.Color
	Foreground color.Color
	FontFace   font.Face
	Rotation   float64 // rotation in degrees
	Multiline  bool
}

// DefaultTestImageConfig returns a default configuration for test images.
func DefaultTestImageConfig() TestImageConfig {
	return TestImageConfig{
		Text:       "Sample Text",
		Size:       MediumSize,
		Background: color.White,
		Foreground: color.Black,
		FontFace:   basicfont.Face7x13,
		Rotation:   0,
		Multiline:  false,
	}
}

// GenerateTextImage creates a synthetic text image with the given configuration.
func GenerateTextImage(config TestImageConfig) (*image.RGBA, error) {
	// Create base image
	img := image.NewRGBA(image.Rect(0, 0, config.Size.Width, config.Size.Height))

	// Fill background
	draw.Draw(img, img.Bounds(), &image.Uniform{config.Background}, image.Point{}, draw.Src)

	// Draw text
	drawer := &font.Drawer{
		Dst:  img,
		Src:  &image.Uniform{config.Foreground},
		Face: config.FontFace,
	}

	if config.Multiline {
		// Build lines with a fixed number of words per line to avoid deep nesting.
		words := []string{"This", "is", "a", "multiline", "text", "sample", "for", "OCR", "testing", "purposes"}
		wordsPerLine := 3
		var lines []string
		for i := 0; i < len(words); i += wordsPerLine {
			end := i + wordsPerLine
			if end > len(words) {
				end = len(words)
			}
			lines = append(lines, strings.Join(words[i:end], " "))
		}

		// Draw each line
		lineHeight := config.FontFace.Metrics().Height.Ceil()
		startY := (config.Size.Height - len(lines)*lineHeight) / 2
		for i, line := range lines {
			y := startY + (i+1)*lineHeight
			textWidth := font.MeasureString(config.FontFace, line).Ceil()
			x := (config.Size.Width - textWidth) / 2
			drawer.Dot = fixed.P(x, y)
			drawer.DrawString(line)
		}
	} else {
		// Center the text
		textWidth := font.MeasureString(config.FontFace, config.Text).Ceil()
		textHeight := config.FontFace.Metrics().Height.Ceil()
		x := (config.Size.Width - textWidth) / 2
		y := (config.Size.Height + textHeight) / 2
		drawer.Dot = fixed.P(x, y)
		drawer.DrawString(config.Text)
	}

	// Apply rotation if specified
	if config.Rotation != 0 {
		rotated := imaging.Rotate(img, config.Rotation, color.White)
		// Convert to RGBA
		rgba := image.NewRGBA(rotated.Bounds())
		draw.Draw(rgba, rgba.Bounds(), rotated, rotated.Bounds().Min, draw.Src)
		return rgba, nil
	}

	return img, nil
}

// SaveImage saves an image to the specified path.
func SaveImage(t *testing.T, img image.Image, path string) {
	t.Helper()

	// Ensure directory exists
	dir := filepath.Dir(path)
	require.NoError(t, EnsureDir(dir), "Failed to create directory %s", dir)

	file, err := os.Create(path) //nolint:gosec // G304: Test file creation with controlled path
	require.NoError(t, err, "Failed to create file %s", path)
	defer func() {
		require.NoError(t, file.Close())
	}()

	err = png.Encode(file, img)
	require.NoError(t, err, "Failed to encode PNG image")
}

// LoadImage loads an image from the specified path.
func LoadImage(t *testing.T, path string) image.Image {
	t.Helper()

	file, err := os.Open(path) //nolint:gosec // G304: Test file reading with controlled path
	require.NoError(t, err, "Failed to open image file %s", path)
	defer func() { _ = file.Close() }()

	img, _, err := image.Decode(file)
	require.NoError(t, err, "Failed to decode image")

	return img
}

// CompareImages compares two images and returns true if they are similar.
func CompareImages(img1, img2 image.Image, tolerance float64) bool {
	bounds1 := img1.Bounds()
	bounds2 := img2.Bounds()

	if bounds1 != bounds2 {
		return false
	}

	var totalDiff float64
	var pixelCount float64

	for y := bounds1.Min.Y; y < bounds1.Max.Y; y++ {
		for x := bounds1.Min.X; x < bounds1.Max.X; x++ {
			r1, g1, b1, a1 := img1.At(x, y).RGBA()
			r2, g2, b2, a2 := img2.At(x, y).RGBA()

			// Calculate color difference
			dr := float64(r1) - float64(r2)
			dg := float64(g1) - float64(g2)
			db := float64(b1) - float64(b2)
			da := float64(a1) - float64(a2)

			diff := math.Sqrt(dr*dr + dg*dg + db*db + da*da)
			totalDiff += diff
			pixelCount++
		}
	}

	avgDiff := totalDiff / pixelCount
	maxDiff := math.Sqrt(4 * 65535 * 65535) // Maximum possible difference

	return (avgDiff / maxDiff) <= tolerance
}

// GenerateTestImages creates a set of standard test images in the testdata directory.
func GenerateTestImages(t *testing.T) {
	t.Helper()

	// Simple single-word images
	simpleDir := GetTestImageDir(t, "simple")
	require.NoError(t, EnsureDir(simpleDir))

	words := []string{"Hello", "World", "OCR", "Test", "123", "Sample"}
	for i, word := range words {
		config := DefaultTestImageConfig()
		config.Text = word
		config.Size = SmallSize

		img, err := GenerateTextImage(config)
		require.NoError(t, err, "Failed to generate simple image for word: %s", word)

		SaveImage(t, img, filepath.Join(simpleDir, fmt.Sprintf("simple_%d_%s.png", i+1, word)))
	}

	// Multiline text documents
	multilineDir := GetTestImageDir(t, "multiline")
	require.NoError(t, EnsureDir(multilineDir))

	config := DefaultTestImageConfig()
	config.Size = LargeSize
	config.Multiline = true

	img, err := GenerateTextImage(config)
	require.NoError(t, err, "Failed to generate multiline image")

	SaveImage(t, img, filepath.Join(multilineDir, "multiline_document.png"))

	// Rotated text samples
	rotatedDir := GetTestImageDir(t, "rotated")
	require.NoError(t, EnsureDir(rotatedDir))

	rotations := []float64{0, 90, 180, 270, 45, -45}
	for _, rotation := range rotations {
		config := DefaultTestImageConfig()
		config.Text = "Rotated Text"
		config.Size = MediumSize
		config.Rotation = rotation

		img, err := GenerateTextImage(config)
		require.NoError(t, err, "Failed to generate rotated image for angle: %.1f", rotation)

		SaveImage(t, img, filepath.Join(rotatedDir, fmt.Sprintf("rotated_%.0f.png", rotation)))
	}

	// Simulated scanned documents (with noise and artifacts)
	scannedDir := GetTestImageDir(t, "scanned")
	require.NoError(t, EnsureDir(scannedDir))

	config = DefaultTestImageConfig()
	config.Text = "Scanned Document Sample"
	config.Size = LargeSize
	config.Background = color.RGBA{248, 248, 248, 255} // Slightly off-white
	config.Foreground = color.RGBA{32, 32, 32, 255}    // Slightly off-black

	img, err = GenerateTextImage(config)
	require.NoError(t, err, "Failed to generate scanned document image")

	// Add some noise to simulate scanning artifacts
	noisyImg := addNoise(img, 0.02) // 2% noise
	SaveImage(t, noisyImg, filepath.Join(scannedDir, "scanned_document.png"))
}

// addNoise adds random noise to an image to simulate scanning artifacts.
func addNoise(img *image.RGBA, noiseLevel float64) *image.RGBA {
	bounds := img.Bounds()
	noisy := image.NewRGBA(bounds)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, a := img.At(x, y).RGBA()

			// Add random noise
			if math.Mod(float64(x*y), 1.0/noiseLevel) < 1.0 {
				// Flip random pixels
				if (x+y)%2 == 0 {
					r = 65535 - r
					g = 65535 - g
					b = 65535 - b
				}
			}

			//nolint:gosec // G115: Safe conversion for image noise generation
			noisy.Set(x, y, color.RGBA64{uint16(r), uint16(g), uint16(b), uint16(a)})
		}
	}

	return noisy
}

// CreateTestImage creates a simple test image with the specified dimensions and color.
func CreateTestImage(width, height int, backgroundColor color.Color) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.Draw(img, img.Bounds(), &image.Uniform{backgroundColor}, image.Point{}, draw.Src)
	return img
}

// CreateTestImageWithText creates a test image with text rendered on it.
func CreateTestImageWithText(text string, width, height int) image.Image {
	config := DefaultTestImageConfig()
	config.Text = text
	config.Size = ImageSize{Width: width, Height: height}

	img, err := GenerateTextImage(config)
	if err != nil {
		// Fallback to simple colored image if text generation fails
		return CreateTestImage(width, height, color.White)
	}

	return img
}

// LoadImageFile loads an image from the specified path (non-testing version).
func LoadImageFile(path string) (image.Image, error) {
	file, err := os.Open(path) //nolint:gosec // G304: Opening user-provided image file is expected
	if err != nil {
		return nil, fmt.Errorf("failed to open image file %s: %w", path, err)
	}
	defer func() { _ = file.Close() }()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}

	return img, nil
}
