package testutil

import (
	"image/color"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultTestImageConfig(t *testing.T) {
	config := DefaultTestImageConfig()
	assert.Equal(t, "Sample Text", config.Text)
	assert.Equal(t, MediumSize, config.Size)
	assert.Equal(t, color.White, config.Background)
	assert.Equal(t, color.Black, config.Foreground)
	assert.InDelta(t, 0.0, config.Rotation, 0.0001)
	assert.False(t, config.Multiline)
}

func TestGenerateTextImage(t *testing.T) {
	config := DefaultTestImageConfig()
	config.Text = "Test"
	config.Size = SmallSize

	img, err := GenerateTextImage(config)
	require.NoError(t, err)
	assert.NotNil(t, img)

	bounds := img.Bounds()
	assert.Equal(t, SmallSize.Width, bounds.Dx())
	assert.Equal(t, SmallSize.Height, bounds.Dy())
}

func TestGenerateMultilineTextImage(t *testing.T) {
	config := DefaultTestImageConfig()
	config.Multiline = true
	config.Size = LargeSize

	img, err := GenerateTextImage(config)
	require.NoError(t, err)
	assert.NotNil(t, img)

	bounds := img.Bounds()
	assert.Equal(t, LargeSize.Width, bounds.Dx())
	assert.Equal(t, LargeSize.Height, bounds.Dy())
}

func TestGenerateRotatedTextImage(t *testing.T) {
	config := DefaultTestImageConfig()
	config.Text = "Rotated"
	config.Rotation = 45.0

	img, err := GenerateTextImage(config)
	require.NoError(t, err)
	assert.NotNil(t, img)
}

func TestSaveAndLoadImage(t *testing.T) {
	// Generate a test image
	config := DefaultTestImageConfig()
	config.Text = "Save Test"
	img, err := GenerateTextImage(config)
	require.NoError(t, err)

	// Save to temporary file
	tempDir := CreateTempDir(t)
	imagePath := tempDir + "/test_image.png"
	SaveImage(t, img, imagePath)

	// Verify file exists
	assert.True(t, FileExists(imagePath))

	// Load the image back
	loadedImg := LoadImage(t, imagePath)
	assert.NotNil(t, loadedImg)

	// Compare bounds
	assert.Equal(t, img.Bounds(), loadedImg.Bounds())
}

func TestCompareImages(t *testing.T) {
	config := DefaultTestImageConfig()
	config.Text = "Compare Test"

	// Generate two identical images
	img1, err := GenerateTextImage(config)
	require.NoError(t, err)

	img2, err := GenerateTextImage(config)
	require.NoError(t, err)

	// Should be identical (or very similar)
	assert.True(t, CompareImages(img1, img2, 0.01))

	// Generate a very different image (different background color)
	config.Text = "Completely Different"
	config.Background = color.Black
	config.Foreground = color.White
	img3, err := GenerateTextImage(config)
	require.NoError(t, err)

	// Should be different (use a lower tolerance for this test)
	assert.False(t, CompareImages(img1, img3, 0.8))
}

// TestGenerateTestImages tests the main image generation function
// This also serves as a way to actually generate the test images.
func TestGenerateTestImages(t *testing.T) {
	// This will generate all the standard test images
	GenerateTestImages(t)

	// Verify that images were created
	simpleDir := GetTestImageDir(t, "simple")
	assert.True(t, DirExists(simpleDir))

	multilineDir := GetTestImageDir(t, "multiline")
	assert.True(t, DirExists(multilineDir))

	rotatedDir := GetTestImageDir(t, "rotated")
	assert.True(t, DirExists(rotatedDir))

	scannedDir := GetTestImageDir(t, "scanned")
	assert.True(t, DirExists(scannedDir))

	// Check that some specific files exist
	assert.True(t, FileExists(simpleDir+"/simple_1_Hello.png"))
	assert.True(t, FileExists(multilineDir+"/multiline_document.png"))
	assert.True(t, FileExists(rotatedDir+"/rotated_90.png"))
	assert.True(t, FileExists(scannedDir+"/scanned_document.png"))
}
