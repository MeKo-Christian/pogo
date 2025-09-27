package pipeline

import (
	"image"
	"image/color"

	"github.com/MeKo-Tech/pogo/internal/utils"
)

// RenderOverlay draws region boxes and polygons over the image and returns an RGBA copy.
func RenderOverlay(img image.Image, res *OCRImageResult, boxColor color.Color, polyColor color.Color) *image.RGBA {
	if img == nil {
		return nil
	}
	b := img.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	// copy background
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.Set(x-b.Min.X, y-b.Min.Y, img.At(x, y))
		}
	}
	if res == nil {
		return dst
	}
	// Coordinate transform from possibly rotated working image back to original
	angle := 0
	if res.Orientation.Applied {
		angle = res.Orientation.Angle
	}
	W0, H0 := b.Dx(), b.Dy()
	toOriginal := func(x, y float64) (float64, float64) {
		switch angle {
		case 90: // working was 90 CCW relative to original => inverse mapping
			// original(x0,y0) -> rotated(x1,y1): x1=y0, y1=W0-1-x0
			// inverse: x0 = W0-1-y1; y0 = x1
			return float64(W0-1) - y, x
		case 180:
			return float64(W0-1) - x, float64(H0-1) - y
		case 270:
			// CCW 270 (CW 90): original->rotated: x1=H0-1-y0, y1=x0; inverse: x0=y1, y0=H0-1-x1
			return y, float64(H0-1) - x
		default:
			return x, y
		}
	}
	// draw regions
	for _, r := range res.Regions {
		// Transform AABB by transforming its four corners and re-AABB
		x1, y1 := toOriginal(float64(r.Box.X), float64(r.Box.Y))
		x2, y2 := toOriginal(float64(r.Box.X+r.Box.W), float64(r.Box.Y))
		x3, y3 := toOriginal(float64(r.Box.X+r.Box.W), float64(r.Box.Y+r.Box.H))
		x4, y4 := toOriginal(float64(r.Box.X), float64(r.Box.Y+r.Box.H))
		minX, maxX := min4(x1, x2, x3, x4), max4(x1, x2, x3, x4)
		minY, maxY := min4(y1, y2, y3, y4), max4(y1, y2, y3, y4)
		rect := image.Rect(int(minX+0.5), int(minY+0.5), int(maxX+0.5), int(maxY+0.5))
		utils.DrawRect(dst, rect, boxColor, 1)
		// polygon (if any)
		if len(r.Polygon) >= 2 {
			pts := make([]utils.Point, len(r.Polygon))
			for i, p := range r.Polygon {
				ox, oy := toOriginal(p.X, p.Y)
				pts[i] = utils.Point{X: ox, Y: oy}
			}
			utils.DrawPolygon(dst, pts, polyColor, 1)
		}
	}
	return dst
}

func min4(a, b, c, d float64) float64 {
	m := a
	if b < m {
		m = b
	}
	if c < m {
		m = c
	}
	if d < m {
		m = d
	}
	return m
}

func max4(a, b, c, d float64) float64 {
	m := a
	if b > m {
		m = b
	}
	if c > m {
		m = c
	}
	if d > m {
		m = d
	}
	return m
}
