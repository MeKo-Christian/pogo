package utils

import (
	"image"
	"image/color"
	"math"

	"github.com/disintegration/imaging"
)

// Point represents a 2D coordinate in float space.
type Point struct {
	X float64
	Y float64
}

// Box represents an axis-aligned bounding box in float coordinates.
type Box struct {
	MinX float64
	MinY float64
	MaxX float64
	MaxY float64
}

// NewBox constructs a Box from min/max coordinates ensuring ordering.
func NewBox(x1, y1, x2, y2 float64) Box {
	if x1 > x2 {
		x1, x2 = x2, x1
	}
	if y1 > y2 {
		y1, y2 = y2, y1
	}
	return Box{MinX: x1, MinY: y1, MaxX: x2, MaxY: y2}
}

// Width returns the box width.
func (b Box) Width() float64 { return b.MaxX - b.MinX }

// Height returns the box height.
func (b Box) Height() float64 { return b.MaxY - b.MinY }

// ToRect converts a Box to an image.Rectangle, clamped to image bounds.
func (b Box) ToRect(bounds image.Rectangle) image.Rectangle {
	x1 := clampInt(int(math.Floor(b.MinX)), bounds.Min.X, bounds.Max.X)
	y1 := clampInt(int(math.Floor(b.MinY)), bounds.Min.Y, bounds.Max.Y)
	x2 := clampInt(int(math.Ceil(b.MaxX)), bounds.Min.X, bounds.Max.X)
	y2 := clampInt(int(math.Ceil(b.MaxY)), bounds.Min.Y, bounds.Max.Y)
	if x2 < x1 {
		x2 = x1
	}
	if y2 < y1 {
		y2 = y1
	}
	return image.Rect(x1, y1, x2, y2)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// ScalePoint scales a point by sx, sy.
func ScalePoint(p Point, sx, sy float64) Point {
	return Point{X: p.X * sx, Y: p.Y * sy}
}

// OffsetPoint offsets a point by dx, dy.
func OffsetPoint(p Point, dx, dy float64) Point {
	return Point{X: p.X + dx, Y: p.Y + dy}
}

// ScalePoints returns a scaled copy of points.
func ScalePoints(pts []Point, sx, sy float64) []Point {
	out := make([]Point, len(pts))
	for i, p := range pts {
		out[i] = ScalePoint(p, sx, sy)
	}
	return out
}

// OffsetPoints returns an offset copy of points.
func OffsetPoints(pts []Point, dx, dy float64) []Point {
	out := make([]Point, len(pts))
	for i, p := range pts {
		out[i] = OffsetPoint(p, dx, dy)
	}
	return out
}

// BoundingBox returns the axis-aligned bounding box for a set of points.
func BoundingBox(pts []Point) Box {
	if len(pts) == 0 {
		return Box{}
	}
	minX, minY := pts[0].X, pts[0].Y
	maxX, maxY := pts[0].X, pts[0].Y
	for _, p := range pts[1:] {
		if p.X < minX {
			minX = p.X
		}
		if p.Y < minY {
			minY = p.Y
		}
		if p.X > maxX {
			maxX = p.X
		}
		if p.Y > maxY {
			maxY = p.Y
		}
	}
	return Box{MinX: minX, MinY: minY, MaxX: maxX, MaxY: maxY}
}

// CropImageRect crops an image to the given rectangle.
func CropImageRect(img image.Image, rect image.Rectangle) image.Image {
	rect = rect.Intersect(img.Bounds())
	if rect.Empty() {
		return imaging.New(0, 0, color.Transparent)
	}
	return imaging.Crop(img, rect)
}

// CropImageBox crops an image using a float Box.
func CropImageBox(img image.Image, box Box) image.Image {
	return CropImageRect(img, box.ToRect(img.Bounds()))
}

// Rotate90 rotates the image 90 degrees counter-clockwise.
func Rotate90(img image.Image) image.Image { return imaging.Rotate90(img) }

// Rotate180 rotates the image 180 degrees.
func Rotate180(img image.Image) image.Image { return imaging.Rotate180(img) }

// Rotate270 rotates the image 270 degrees counter-clockwise.
func Rotate270(img image.Image) image.Image { return imaging.Rotate270(img) }

// DrawRect draws an axis-aligned rectangle outline into dst.
func DrawRect(dst *image.RGBA, rect image.Rectangle, col color.Color, thickness int) {
	if thickness < 1 {
		thickness = 1
	}
	rect = rect.Intersect(dst.Bounds())
	if rect.Empty() {
		return
	}
	// Top and bottom edges
	for t := range thickness {
		yTop := rect.Min.Y + t
		yBot := rect.Max.Y - 1 - t
		for x := rect.Min.X; x < rect.Max.X; x++ {
			dst.Set(x, yTop, col)
			dst.Set(x, yBot, col)
		}
	}
	// Left and right edges
	for t := range thickness {
		xLeft := rect.Min.X + t
		xRight := rect.Max.X - 1 - t
		for y := rect.Min.Y; y < rect.Max.Y; y++ {
			dst.Set(xLeft, y, col)
			dst.Set(xRight, y, col)
		}
	}
}

// DrawPolygon draws connected line segments and closes the polygon.
func DrawPolygon(dst *image.RGBA, pts []Point, col color.Color, thickness int) {
	if len(pts) < 2 {
		return
	}
	// Convert to int points
	ip := make([]image.Point, len(pts))
	for i, p := range pts {
		ip[i] = image.Pt(int(math.Round(p.X)), int(math.Round(p.Y)))
	}
	for i := range ip {
		a := ip[i]
		b := ip[(i+1)%len(ip)]
		drawLine(dst, a, b, col, thickness)
	}
}

// drawLine draws a line between two points using a simple Bresenham variant.
func drawLine(dst *image.RGBA, a, b image.Point, col color.Color, thickness int) {
	x0, y0 := a.X, a.Y
	x1, y1 := b.X, b.Y
	dx := int(math.Abs(float64(x1 - x0)))
	sx := -1
	if x0 < x1 {
		sx = 1
	}
	dy := -int(math.Abs(float64(y1 - y0)))
	sy := -1
	if y0 < y1 {
		sy = 1
	}
	err := dx + dy
	for {
		drawThickPoint(dst, x0, y0, col, thickness)
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

func drawThickPoint(dst *image.RGBA, x, y int, col color.Color, thickness int) {
	if thickness < 1 {
		thickness = 1
	}
	r := (thickness - 1) / 2
	for yy := y - r; yy <= y+r; yy++ {
		for xx := x - r; xx <= x+r; xx++ {
			if image.Pt(xx, yy).In(dst.Bounds()) {
				dst.Set(xx, yy, col)
			}
		}
	}
}
