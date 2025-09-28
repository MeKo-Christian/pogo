package rectify

import (
	"image"
	"image/color"

	"github.com/MeKo-Tech/pogo/internal/utils"
)

// transformAndWarpImage transforms coordinates and warps the image.
func (r *Rectifier) transformAndWarpImage(img, resized image.Image, rect []utils.Point) (image.Image, error) {
	// Scale rect points back to original image coordinates
	rb := resized.Bounds()
	ib := img.Bounds()
	sx := float64(ib.Dx()) / float64(rb.Dx())
	sy := float64(ib.Dy()) / float64(rb.Dy())
	srcQuad := make([]utils.Point, 4)
	for i := range 4 {
		srcQuad[i] = utils.Point{X: rect[i].X * sx, Y: rect[i].Y * sy}
	}

	if r.cfg.DebugDir != "" {
		_ = dumpOverlayPNG(r.cfg.DebugDir, img, srcQuad)
	}

	// Determine output dimensions based on quad edges
	w0 := hypot(srcQuad[1], srcQuad[0])
	w1 := hypot(srcQuad[2], srcQuad[3])
	h0 := hypot(srcQuad[3], srcQuad[0])
	h1 := hypot(srcQuad[2], srcQuad[1])
	avgW := (w0 + w1) * 0.5
	avgH := (h0 + h1) * 0.5

	if avgW <= 1 || avgH <= 1 {
		return img, nil
	}

	targetH := r.cfg.OutputHeight
	if targetH <= 0 {
		targetH = 1024
	}
	targetW := int((avgW / avgH) * float64(targetH))

	// Round to multiples of 32 to be detector-friendly
	targetW = (targetW / 32) * 32
	targetH = (targetH / 32) * 32
	if targetW < 32 {
		targetW = 32
	}
	if targetH < 32 {
		targetH = 32
	}

	dst := warpPerspective(img, srcQuad, targetW, targetH)
	if dst == nil {
		return img, nil
	}

	if r.cfg.DebugDir != "" {
		_ = dumpComparePNG(r.cfg.DebugDir, img, srcQuad, dst)
	}

	return dst, nil
}

// warpPerspective warps the quadrilateral region srcQuad from src into a
// target rectangle of size dstW x dstH using inverse homography + bilinear sampling.
func warpPerspective(src image.Image, srcQuad []utils.Point, dstW, dstH int) image.Image {
	if len(srcQuad) != 4 || dstW <= 0 || dstH <= 0 {
		return nil
	}

	// Build homography from dst rect to src quad. dst corners in CCW: (0,0),(W-1,0),(W-1,H-1),(0,H-1)
	d0 := utils.Point{X: 0, Y: 0}
	d1 := utils.Point{X: float64(dstW - 1), Y: 0}
	d2 := utils.Point{X: float64(dstW - 1), Y: float64(dstH - 1)}
	d3 := utils.Point{X: 0, Y: float64(dstH - 1)}
	H, ok := computeHomography(
		[4]utils.Point{d0, d1, d2, d3},
		[4]utils.Point{srcQuad[0], srcQuad[1], srcQuad[2], srcQuad[3]},
	)
	if !ok {
		return nil
	}

	// Generate destination image
	out := image.NewRGBA(image.Rect(0, 0, dstW, dstH))

	// Precompute bounds
	sb := src.Bounds()
	for y := range dstH {
		for x := range dstW {
			// Map (x,y,1) via H to source coords
			sx, sy := applyHomography(H, float64(x), float64(y))
			// Bilinear sample
			cr := bilinearSample(src, sx+float64(sb.Min.X), sy+float64(sb.Min.Y))
			out.Set(x, y, cr)
		}
	}

	return out
}

func bilinearSample(src image.Image, x, y float64) color.Color {
	// Clamp sampling outside bounds to black
	b := src.Bounds()
	if x < float64(b.Min.X) || y < float64(b.Min.Y) || x > float64(b.Max.X-1) || y > float64(b.Max.Y-1) {
		return color.RGBA{0, 0, 0, 255}
	}
	x0 := int(x)
	y0 := int(y)
	x1 := x0 + 1
	y1 := y0 + 1
	if x1 >= b.Max.X {
		x1 = b.Max.X - 1
	}
	if y1 >= b.Max.Y {
		y1 = b.Max.Y - 1
	}
	fx := x - float64(x0)
	fy := y - float64(y0)
	c00 := toRGBA(src.At(x0, y0))
	c10 := toRGBA(src.At(x1, y0))
	c01 := toRGBA(src.At(x0, y1))
	c11 := toRGBA(src.At(x1, y1))
	r := lerp(lerp(c00.R, c10.R, fx), lerp(c01.R, c11.R, fx), fy)
	g := lerp(lerp(c00.G, c10.G, fx), lerp(c01.G, c11.G, fx), fy)
	bl := lerp(lerp(c00.B, c10.B, fx), lerp(c01.B, c11.B, fx), fy)
	a := lerp(lerp(c00.A, c10.A, fx), lerp(c01.A, c11.A, fx), fy)
	return color.RGBA{uint8(r + 0.5), uint8(g + 0.5), uint8(bl + 0.5), uint8(a + 0.5)}
}

type rgba struct{ R, G, B, A float64 }

func toRGBA(c color.Color) rgba {
	r, g, b, a := c.RGBA()
	return rgba{R: float64(r >> 8), G: float64(g >> 8), B: float64(b >> 8), A: float64(a >> 8)}
}

func lerp(a, b, t float64) float64 { return a + (b-a)*t }
func abs(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}
