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
