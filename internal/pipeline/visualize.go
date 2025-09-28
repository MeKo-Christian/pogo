package pipeline

import (
	"image"
	"image/color"

	"github.com/MeKo-Tech/pogo/internal/utils"
)

// transformCoordinates applies the inverse rotation transformation from working image to original image.
func transformCoordinates(x, y float64, angle int, w0, h0 int) (float64, float64) {
	switch angle {
	case 90: // working was 90 CCW relative to original => inverse mapping
		return float64(w0-1) - y, x
	case 180:
		return float64(w0-1) - x, float64(h0-1) - y
	case 270:
		return y, float64(h0-1) - x
	default:
		return x, y
	}
}

// drawRegionBox draws a transformed bounding box on the destination image.
func drawRegionBox(dst *image.RGBA, r OCRRegionResult, angle int, w0, h0 int, boxColor color.Color) {
	// Transform AABB by transforming its four corners and re-AABB
	x1, y1 := transformCoordinates(float64(r.Box.X), float64(r.Box.Y), angle, w0, h0)
	x2, y2 := transformCoordinates(float64(r.Box.X+r.Box.W), float64(r.Box.Y), angle, w0, h0)
	x3, y3 := transformCoordinates(float64(r.Box.X+r.Box.W), float64(r.Box.Y+r.Box.H), angle, w0, h0)
	x4, y4 := transformCoordinates(float64(r.Box.X), float64(r.Box.Y+r.Box.H), angle, w0, h0)
	minX, maxX := min4(x1, x2, x3, x4), max4(x1, x2, x3, x4)
	minY, maxY := min4(y1, y2, y3, y4), max4(y1, y2, y3, y4)
	rect := image.Rect(int(minX+0.5), int(minY+0.5), int(maxX+0.5), int(maxY+0.5))
	utils.DrawRect(dst, rect, boxColor, 1)
}

// drawRegionPolygon draws a transformed polygon on the destination image.
func drawRegionPolygon(dst *image.RGBA, r OCRRegionResult, angle int, w0, h0 int, polyColor color.Color) {
	if len(r.Polygon) < 2 {
		return
	}
	pts := make([]utils.Point, len(r.Polygon))
	for i, p := range r.Polygon {
		ox, oy := transformCoordinates(p.X, p.Y, angle, w0, h0)
		pts[i] = utils.Point{X: ox, Y: oy}
	}
	utils.DrawPolygon(dst, pts, polyColor, 1)
}

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
	w0, h0 := b.Dx(), b.Dy()
	// draw regions
	for _, r := range res.Regions {
		drawRegionBox(dst, r, angle, w0, h0, boxColor)
		drawRegionPolygon(dst, r, angle, w0, h0, polyColor)
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
