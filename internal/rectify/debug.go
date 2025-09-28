package rectify

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"time"

	"github.com/MeKo-Tech/pogo/internal/utils"
)

func dumpMaskPNG(dir string, mask []float32, w, h int, thr float64) error {
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return err
	}
	ts := time.Now().UnixNano()
	path := filepath.Join(dir, fmt.Sprintf("rect_mask_%d.png", ts))
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		row := y * w
		for x := range w {
			v := float64(mask[row+x])
			// visualize as grayscale, emphasize threshold
			g := uint8(clamp01(v) * 255)
			if v >= thr {
				img.Set(x, y, color.RGBA{R: g, G: 0, B: 0, A: 255}) // red-ish for positive
			} else {
				img.Set(x, y, color.RGBA{R: g, G: g, B: g, A: 255})
			}
		}
	}
	f, err := os.Create(path) //nolint:gosec // G304: path is constructed from timestamp in debug directory
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	return png.Encode(f, img)
}

func dumpOverlayPNG(dir string, src image.Image, quad []utils.Point) error {
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return err
	}
	ts := time.Now().UnixNano()
	path := filepath.Join(dir, fmt.Sprintf("rect_overlay_%d.png", ts))
	// Clone to RGBA and draw polygon
	b := src.Bounds()
	canvas := image.NewRGBA(b)
	// draw original
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			canvas.Set(x, y, src.At(x, y))
		}
	}
	utils.DrawPolygon(canvas, quad, color.RGBA{255, 0, 0, 255}, 2)
	f, err := os.Create(path) //nolint:gosec // G304: path is constructed from timestamp in debug directory
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	return png.Encode(f, canvas)
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func dumpComparePNG(dir string, src image.Image, srcQuad []utils.Point, dst image.Image) error {
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return err
	}
	ts := time.Now().UnixNano()
	path := filepath.Join(dir, fmt.Sprintf("rect_compare_%d.png", ts))
	sb := src.Bounds()
	db := dst.Bounds()
	gap := 10
	outW := sb.Dx() + gap + db.Dx()
	outH := sb.Dy()
	if db.Dy() > outH {
		outH = db.Dy()
	}
	canvas := image.NewRGBA(image.Rect(0, 0, outW, outH))
	// draw source on left
	for y := range sb.Dy() {
		for x := range sb.Dx() {
			canvas.Set(x, y, src.At(sb.Min.X+x, sb.Min.Y+y))
		}
	}
	// draw destination on right
	xoff := sb.Dx() + gap
	for y := range db.Dy() {
		for x := range db.Dx() {
			canvas.Set(xoff+x, y, dst.At(db.Min.X+x, db.Min.Y+y))
		}
	}
	// overlay quad on left
	utils.DrawPolygon(canvas, srcQuad, color.RGBA{255, 0, 0, 255}, 2)
	// overlay rectangle border on right
	utils.DrawRect(canvas, image.Rect(xoff, 0, xoff+db.Dx(), db.Dy()), color.RGBA{0, 255, 0, 255}, 2)
	f, err := os.Create(path) //nolint:gosec // G304: path is constructed from timestamp in debug directory
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	return png.Encode(f, canvas)
}
