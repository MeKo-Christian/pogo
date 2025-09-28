package utils

import (
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestIsSupportedImage(t *testing.T) {
	cases := []struct {
		path string
		ok   bool
	}{
		{"a.jpg", true},
		{"b.jpeg", true},
		{"c.png", true},
		{"d.bmp", true},
		{"e.tiff", false},
		{"f.gif", false},
	}
	for _, c := range cases {
		if IsSupportedImage(c.path) != c.ok {
			t.Fatalf("IsSupportedImage(%s) expected %v", c.path, c.ok)
		}
	}
}

func writeTempPNG(t *testing.T, dir string, w, h int, col color.Color) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.Set(x, y, col)
		}
	}
	path := filepath.Join(dir, "test.png")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer func() {
		require.NoError(t, f.Close())
	}()
	if err := png.Encode(f, img); err != nil {
		t.Fatalf("encode: %v", err)
	}
	return path
}

func TestLoadImageAndMetadata(t *testing.T) {
	dir := t.TempDir()
	p := writeTempPNG(t, dir, 10, 20, color.RGBA{R: 10, G: 20, B: 30, A: 255})

	img, meta, err := LoadImage(p)
	if err != nil {
		t.Fatalf("LoadImage error: %v", err)
	}
	if img == nil {
		t.Fatalf("nil image")
	}
	if meta.Format != "png" {
		t.Fatalf("expected format png, got %s", meta.Format)
	}
	if meta.Width != 10 || meta.Height != 20 {
		t.Fatalf("unexpected dims: %dx%d", meta.Width, meta.Height)
	}
	if meta.SizeBytes <= 0 {
		t.Fatalf("expected SizeBytes > 0")
	}
}

func TestValidateImageConstraints(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	cons := ImageConstraints{MaxWidth: 1024, MaxHeight: 1024, MinWidth: 32, MinHeight: 32}
	if err := ValidateImageConstraints(img, cons); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	cons.MinWidth = 128
	if err := ValidateImageConstraints(img, cons); err == nil {
		t.Fatalf("expected error for too small image")
	}
}

func TestResizeImageProperties(t *testing.T) {
	// 300x200 -> constrained by 100x100, expect multiples of 32
	base := image.NewRGBA(image.Rect(0, 0, 300, 200))
	cons := ImageConstraints{MaxWidth: 100, MaxHeight: 100, MinWidth: 32, MinHeight: 32}
	resized, err := ResizeImage(base, cons)
	if err != nil {
		t.Fatalf("ResizeImage error: %v", err)
	}
	b := resized.Bounds()
	if b.Dx()%32 != 0 || b.Dy()%32 != 0 {
		t.Fatalf("expected multiples of 32, got %dx%d", b.Dx(), b.Dy())
	}
	if b.Dx() > 100 || b.Dy() > 100 {
		t.Fatalf("dimensions exceed constraints: %dx%d", b.Dx(), b.Dy())
	}
}

func TestPadImageCentering(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 50, 50))
	padded, err := PadImage(img, 64, 64)
	if err != nil {
		t.Fatalf("PadImage error: %v", err)
	}
	b := padded.Bounds()
	if b.Dx() != 64 || b.Dy() != 64 {
		t.Fatalf("unexpected padded dims: %dx%d", b.Dx(), b.Dy())
	}
}

// NormalizeImage and AssessImageQuality tests exist in image_processing_test.go

func TestBoundingBoxAndScale(t *testing.T) {
	pts := []Point{{0, 0}, {10, 5}, {3, 7}}
	box := BoundingBox(pts)
	if box.MinX != 0 || box.MinY != 0 || box.MaxX != 10 || box.MaxY != 7 {
		t.Fatalf("unexpected box: %+v", box)
	}
	s := ScalePoints(pts, 2, 3)
	if s[1].X != 20 || s[1].Y != 15 {
		t.Fatalf("scale mismatch: %+v", s[1])
	}
}

func TestCropAndRotate(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 8, 4))
	// fill with red
	for y := range 4 {
		for x := range 8 {
			img.Set(x, y, color.RGBA{255, 0, 0, 255})
		}
	}
	// crop 2..6 x 1..3 => width=4, height=2
	cropped := CropImageRect(img, image.Rect(2, 1, 6, 3))
	if cropped.Bounds().Dx() != 4 || cropped.Bounds().Dy() != 2 {
		t.Fatalf("unexpected crop dims: %dx%d", cropped.Bounds().Dx(), cropped.Bounds().Dy())
	}
	r90 := Rotate90(cropped)
	if r90.Bounds().Dx() != 2 || r90.Bounds().Dy() != 4 {
		t.Fatalf("unexpected rotate dims: %dx%d", r90.Bounds().Dx(), r90.Bounds().Dy())
	}
}

func TestDrawRectAndPolygon(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 20, 10))
	rect := image.Rect(2, 2, 10, 8)
	DrawRect(img, rect, color.RGBA{0, 255, 0, 255}, 1)
	// Expect corners painted
	if img.RGBAAt(2, 2) == (color.RGBA{}) {
		t.Fatalf("expected top-left pixel colored")
	}
	poly := []Point{{12, 2}, {18, 2}, {18, 8}, {12, 8}}
	DrawPolygon(img, poly, color.RGBA{0, 0, 255, 255}, 1)
	if img.RGBAAt(12, 2) == (color.RGBA{}) {
		t.Fatalf("expected polygon pixel colored")
	}
}

// Test BatchLoadImages function.
func TestBatchLoadImages(t *testing.T) {
	dir := t.TempDir()

	// Create test images with unique names
	path1 := filepath.Join(dir, "test1.png")
	path2 := filepath.Join(dir, "test2.png")
	invalidPath := filepath.Join(dir, "nonexistent.png")

	// Write first image
	img1 := image.NewRGBA(image.Rect(0, 0, 10, 15))
	f1, err := os.Create(path1)
	require.NoError(t, err)
	require.NoError(t, png.Encode(f1, img1))
	require.NoError(t, f1.Close())

	// Write second image
	img2 := image.NewRGBA(image.Rect(0, 0, 20, 25))
	f2, err := os.Create(path2)
	require.NoError(t, err)
	require.NoError(t, png.Encode(f2, img2))
	require.NoError(t, f2.Close())

	paths := []string{path1, path2, invalidPath}
	results := BatchLoadImages(paths)

	require.Len(t, results, 3)

	// Check first valid image
	require.Equal(t, path1, results[0].Path)
	require.NotNil(t, results[0].Img)
	require.NoError(t, results[0].Err)
	require.Equal(t, 10, results[0].Meta.Width)
	require.Equal(t, 15, results[0].Meta.Height)

	// Check second valid image
	require.Equal(t, path2, results[1].Path)
	require.NotNil(t, results[1].Img)
	require.NoError(t, results[1].Err)
	require.Equal(t, 20, results[1].Meta.Width)
	require.Equal(t, 25, results[1].Meta.Height)

	// Check invalid image
	require.Equal(t, invalidPath, results[2].Path)
	require.Nil(t, results[2].Img)
	require.Error(t, results[2].Err)
}

func TestBatchLoadImages_EmptySlice(t *testing.T) {
	results := BatchLoadImages([]string{})
	require.Empty(t, results)
}

// Test Box geometric operations.
func TestNewBox(t *testing.T) {
	tests := []struct {
		name           string
		x1, y1, x2, y2 float64
		expectedBox    Box
	}{
		{
			name: "normal coordinates",
			x1:   1, y1: 2, x2: 5, y2: 8,
			expectedBox: Box{MinX: 1, MinY: 2, MaxX: 5, MaxY: 8},
		},
		{
			name: "swapped x coordinates",
			x1:   5, y1: 2, x2: 1, y2: 8,
			expectedBox: Box{MinX: 1, MinY: 2, MaxX: 5, MaxY: 8},
		},
		{
			name: "swapped y coordinates",
			x1:   1, y1: 8, x2: 5, y2: 2,
			expectedBox: Box{MinX: 1, MinY: 2, MaxX: 5, MaxY: 8},
		},
		{
			name: "all swapped",
			x1:   5, y1: 8, x2: 1, y2: 2,
			expectedBox: Box{MinX: 1, MinY: 2, MaxX: 5, MaxY: 8},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			box := NewBox(tt.x1, tt.y1, tt.x2, tt.y2)
			require.Equal(t, tt.expectedBox, box)
		})
	}
}

func TestBox_Width(t *testing.T) {
	box := Box{MinX: 2, MinY: 3, MaxX: 7, MaxY: 10}
	require.Equal(t, 5.0, box.Width())
}

func TestBox_Height(t *testing.T) {
	box := Box{MinX: 2, MinY: 3, MaxX: 7, MaxY: 10}
	require.Equal(t, 7.0, box.Height())
}

func TestBox_ToRect(t *testing.T) {
	bounds := image.Rect(0, 0, 100, 100)

	tests := []struct {
		name     string
		box      Box
		expected image.Rectangle
	}{
		{
			name:     "normal box within bounds",
			box:      Box{MinX: 10.3, MinY: 20.7, MaxX: 30.1, MaxY: 40.9},
			expected: image.Rect(10, 20, 31, 41),
		},
		{
			name:     "box exceeding bounds",
			box:      Box{MinX: -5, MinY: -10, MaxX: 150, MaxY: 200},
			expected: image.Rect(0, 0, 100, 100),
		},
		{
			name:     "valid box coordinates",
			box:      Box{MinX: 20, MinY: 30, MaxX: 50, MaxY: 60},
			expected: image.Rect(20, 30, 50, 60),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.box.ToRect(bounds)
			require.Equal(t, tt.expected, result)
		})
	}
}

func TestClampInt(t *testing.T) {
	tests := []struct {
		name      string
		v, lo, hi int
		expected  int
	}{
		{"within range", 5, 1, 10, 5},
		{"below range", -2, 1, 10, 1},
		{"above range", 15, 1, 10, 10},
		{"at lower bound", 1, 1, 10, 1},
		{"at upper bound", 10, 1, 10, 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := clampInt(tt.v, tt.lo, tt.hi)
			require.Equal(t, tt.expected, result)
		})
	}
}

func TestOffsetPoint(t *testing.T) {
	point := Point{X: 10, Y: 20}
	result := OffsetPoint(point, 5, -3)
	expected := Point{X: 15, Y: 17}
	require.Equal(t, expected, result)
}

func TestOffsetPoints(t *testing.T) {
	points := []Point{{X: 1, Y: 2}, {X: 3, Y: 4}, {X: 5, Y: 6}}
	result := OffsetPoints(points, 10, -5)
	expected := []Point{{X: 11, Y: -3}, {X: 13, Y: -1}, {X: 15, Y: 1}}
	require.Equal(t, expected, result)
}

func TestCropImageBox(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 10, 8))

	// Fill with test pattern
	for y := range 8 {
		for x := range 10 {
			img.Set(x, y, color.RGBA{R: uint8(x * 25), G: uint8(y * 30), A: 255})
		}
	}

	box := Box{MinX: 2, MinY: 1, MaxX: 6, MaxY: 5}
	cropped := CropImageBox(img, box)

	bounds := cropped.Bounds()
	require.Equal(t, 4, bounds.Dx()) // 6-2 = 4
	require.Equal(t, 4, bounds.Dy()) // 5-1 = 4

	// Verify that cropping worked by checking bounds
	require.Equal(t, 0, bounds.Min.X)
	require.Equal(t, 0, bounds.Min.Y)
	require.NotNil(t, cropped)
}

func TestRotate180(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 3, 2))
	img.Set(0, 0, color.RGBA{R: 255, A: 255}) // top-left red
	img.Set(2, 1, color.RGBA{B: 255, A: 255}) // bottom-right blue

	rotated := Rotate180(img)

	// Verify dimensions remain the same
	bounds := rotated.Bounds()
	require.Equal(t, 3, bounds.Dx())
	require.Equal(t, 2, bounds.Dy())
	require.NotNil(t, rotated)
}

func TestRotate270(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 4, 2))
	img.Set(0, 0, color.RGBA{R: 255, A: 255}) // top-left red

	rotated := Rotate270(img)

	// Original 4x2 becomes 2x4 after 270° rotation
	bounds := rotated.Bounds()
	require.Equal(t, 2, bounds.Dx())
	require.Equal(t, 4, bounds.Dy())
	require.NotNil(t, rotated)
}
