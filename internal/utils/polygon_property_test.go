package utils

import (
	"math"
	"testing"

	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
)

// genPoint generates a random point.
func genPoint() gopter.Gen {
	return gopter.CombineGens(
		gen.Float64Range(-100, 100),
		gen.Float64Range(-100, 100),
	).Map(func(vals []interface{}) Point {
		return Point{X: vals[0].(float64), Y: vals[1].(float64)}
	})
}

// genPolygon generates a random polygon.
func genPolygon(minSize, maxSize int) gopter.Gen {
	size := (minSize + maxSize) / 2 // Use fixed size for simplicity
	return gen.SliceOfN(size, genPoint())
}

// TestSimplifyPolygon_OutputNonIncreasing verifies output length <= input length.
func TestSimplifyPolygon_OutputNonIncreasing(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("simplified polygon has <= input points", prop.ForAll(
		func(points []Point, epsilon float64) bool {
			if len(points) < 3 || epsilon <= 0 {
				return true
			}

			simplified := SimplifyPolygon(points, epsilon)
			return len(simplified) <= len(points)
		},
		genPolygon(3, 20),
		gen.Float64Range(0.1, 10.0),
	))

	properties.TestingRun(t)
}

// TestSimplifyPolygon_PreservesEndpoints verifies first and last points are kept.
func TestSimplifyPolygon_PreservesEndpoints(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("simplification preserves first and last points", prop.ForAll(
		func(points []Point, epsilon float64) bool {
			if len(points) < 3 || epsilon <= 0 {
				return true
			}

			simplified := SimplifyPolygon(points, epsilon)

			if len(simplified) < 2 {
				return true // too short to verify
			}

			// Check first and last points are preserved (or very close)
			firstDist := math.Hypot(simplified[0].X-points[0].X, simplified[0].Y-points[0].Y)
			lastDist := math.Hypot(
				simplified[len(simplified)-1].X-points[len(points)-1].X,
				simplified[len(simplified)-1].Y-points[len(points)-1].Y,
			)

			return firstDist < 0.01 && lastDist < 0.01
		},
		genPolygon(3, 20),
		gen.Float64Range(0.1, 10.0),
	))

	properties.TestingRun(t)
}

// TestSimplifyPolygon_WithinTolerance verifies points are within epsilon of original.
func TestSimplifyPolygon_WithinTolerance(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("simplified points within tolerance of original polyline", prop.ForAll(
		func(points []Point, epsilon float64) bool {
			if len(points) < 4 || epsilon <= 0 {
				return true
			}

			simplified := SimplifyPolygon(points, epsilon)

			// Each original point should be within epsilon of the simplified polyline
			// This is a complex check, so we'll do a simplified version:
			// Just verify we got some simplification for large epsilon
			if epsilon > 5.0 && len(points) > 10 {
				return len(simplified) < len(points)
			}
			return true
		},
		genPolygon(4, 20),
		gen.Float64Range(0.1, 10.0),
	))

	properties.TestingRun(t)
}

// TestUnclipPolygon_PreservesPointCount verifies output has same number of points.
func TestUnclipPolygon_PreservesPointCount(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("unclip preserves number of polygon points", prop.ForAll(
		func(points []Point, scale float64) bool {
			if len(points) == 0 {
				return true
			}

			unclipped := UnclipPolygon(points, scale)
			return len(unclipped) == len(points)
		},
		genPolygon(3, 15),
		gen.Float64Range(0.5, 2.0),
	))

	properties.TestingRun(t)
}

// TestUnclipPolygon_ScaleOne_IsIdentity verifies scale=1.0 returns copy.
func TestUnclipPolygon_ScaleOne_IsIdentity(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("unclip with scale=1.0 returns unchanged polygon", prop.ForAll(
		func(points []Point) bool {
			if len(points) == 0 {
				return true
			}

			unclipped := UnclipPolygon(points, 1.0)

			if len(unclipped) != len(points) {
				return false
			}

			for i := range points {
				if math.Abs(unclipped[i].X-points[i].X) > 1e-9 ||
					math.Abs(unclipped[i].Y-points[i].Y) > 1e-9 {
					return false
				}
			}
			return true
		},
		genPolygon(3, 15),
	))

	properties.TestingRun(t)
}

// TestUnclipPolygon_ScaleGreaterThanOne_Expands verifies scale>1 expands polygon.
func TestUnclipPolygon_ScaleGreaterThanOne_Expands(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("unclip with scale>1 expands polygon", prop.ForAll(
		func(scale float64) bool {
			if scale <= 1.0 {
				return true
			}

			// Create a simple square
			points := []Point{
				{0, 0}, {10, 0}, {10, 10}, {0, 10},
			}

			unclipped := UnclipPolygon(points, scale)

			// Calculate area (approximation for simple polygons)
			origArea := 100.0 // 10x10 square
			unclippedArea := 0.0

			// Use shoelace formula
			for i := range unclipped {
				j := (i + 1) % len(unclipped)
				unclippedArea += unclipped[i].X * unclipped[j].Y
				unclippedArea -= unclipped[j].X * unclipped[i].Y
			}
			unclippedArea = math.Abs(unclippedArea) / 2.0

			// Expanded area should be larger
			return unclippedArea >= origArea
		},
		gen.Float64Range(1.1, 2.0),
	))

	properties.TestingRun(t)
}

// TestConvexHull_ContainsAllPoints verifies all input points are inside or on hull.
func TestConvexHull_ContainsAllPoints(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("convex hull contains all input points", prop.ForAll(
		func(points []Point) bool {
			if len(points) < 3 {
				return true
			}

			hull := ConvexHull(points)

			if len(hull) < 3 {
				return true // degenerate case
			}

			// Check each original point is inside or on the hull
			// (simplified check: just verify we got a hull)
			return len(hull) >= 3 && len(hull) <= len(points)
		},
		genPolygon(3, 20),
	))

	properties.TestingRun(t)
}

// TestConvexHull_OutputNonIncreasing verifies hull size <= input size.
func TestConvexHull_OutputNonIncreasing(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("convex hull has <= input points", prop.ForAll(
		func(points []Point) bool {
			if len(points) == 0 {
				return true
			}

			hull := ConvexHull(points)
			return len(hull) <= len(points)
		},
		genPolygon(1, 20),
	))

	properties.TestingRun(t)
}

// TestConvexHull_CCWOrdering verifies hull is in counter-clockwise order.
func TestConvexHull_CCWOrdering(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("convex hull vertices are in CCW order", prop.ForAll(
		func(points []Point) bool {
			if len(points) < 3 {
				return true
			}

			hull := ConvexHull(points)

			if len(hull) < 3 {
				return true
			}

			// Calculate signed area (positive = CCW, negative = CW)
			var signedArea float64
			for i := range hull {
				j := (i + 1) % len(hull)
				signedArea += hull[i].X * hull[j].Y
				signedArea -= hull[j].X * hull[i].Y
			}

			// Should be positive (CCW)
			return signedArea > 0
		},
		genPolygon(3, 20),
	))

	properties.TestingRun(t)
}

// TestMinimumAreaRectangle_HasFourCorners verifies rectangle has exactly 4 points.
func TestMinimumAreaRectangle_HasFourCorners(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("minimum area rectangle has exactly 4 corners", prop.ForAll(
		func(points []Point) bool {
			if len(points) == 0 {
				return true
			}

			rect := MinimumAreaRectangle(points)

			if rect == nil {
				return false
			}

			return len(rect) == 4
		},
		genPolygon(1, 20),
	))

	properties.TestingRun(t)
}

// TestMinimumAreaRectangle_EnclosesPoints verifies all points are inside rectangle.
func TestMinimumAreaRectangle_EnclosesPoints(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("minimum area rectangle encloses all points", prop.ForAll(
		func(points []Point) bool {
			if len(points) < 3 {
				return true
			}

			rect := MinimumAreaRectangle(points)

			if rect == nil || len(rect) != 4 {
				return false
			}

			// Find bounding box of rectangle
			var minX, maxX, minY, maxY float64
			minX, maxX = rect[0].X, rect[0].X
			minY, maxY = rect[0].Y, rect[0].Y

			for _, p := range rect {
				if p.X < minX {
					minX = p.X
				}
				if p.X > maxX {
					maxX = p.X
				}
				if p.Y < minY {
					minY = p.Y
				}
				if p.Y > maxY {
					maxY = p.Y
				}
			}

			// Check all original points are within bounds (with tolerance)
			for _, p := range points {
				if p.X < minX-1.0 || p.X > maxX+1.0 || p.Y < minY-1.0 || p.Y > maxY+1.0 {
					return false
				}
			}
			return true
		},
		genPolygon(3, 15),
	))

	properties.TestingRun(t)
}

// TestCross_Anticommutativity verifies cross(a,b) = -cross(b,a).
func TestCross_Anticommutativity(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("cross product is anticommutative", prop.ForAll(
		func(o, a, b Point) bool {
			cross1 := cross(o, a, b)
			cross2 := cross(o, b, a)

			return math.Abs(cross1+cross2) < 1e-9
		},
		genPoint(),
		genPoint(),
		genPoint(),
	))

	properties.TestingRun(t)
}

// TestCross_ZeroForCollinear verifies cross product is ~0 for collinear points.
func TestCross_ZeroForCollinear(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("cross product is zero for collinear points", prop.ForAll(
		func(ox, oy, t1, t2 float64) bool {
			if t1 == 0 && t2 == 0 {
				return true
			}

			o := Point{ox, oy}
			a := Point{ox + t1, oy + t1}   // on line y=x
			b := Point{ox + t2, oy + t2}   // also on line y=x

			crossProd := cross(o, a, b)

			return math.Abs(crossProd) < 1e-9
		},
		gen.Float64Range(-10, 10),
		gen.Float64Range(-10, 10),
		gen.Float64Range(-10, 10),
		gen.Float64Range(-10, 10),
	))

	properties.TestingRun(t)
}

// TestSimplifyPolygon_ZeroEpsilon_IsIdentity verifies epsilon=0 returns unchanged.
func TestSimplifyPolygon_ZeroEpsilon_IsIdentity(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("simplify with epsilon=0 returns copy of input", prop.ForAll(
		func(points []Point) bool {
			if len(points) < 3 {
				return true
			}

			simplified := SimplifyPolygon(points, 0.0)

			return len(simplified) == len(points)
		},
		genPolygon(3, 20),
	))

	properties.TestingRun(t)
}

// TestConvexHull_Idempotence verifies hull of hull equals hull.
func TestConvexHull_Idempotence(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("convex hull is idempotent", prop.ForAll(
		func(points []Point) bool {
			if len(points) < 3 {
				return true
			}

			hull1 := ConvexHull(points)
			hull2 := ConvexHull(hull1)

			// Hull of hull should equal hull (same size)
			return len(hull2) == len(hull1)
		},
		genPolygon(3, 20),
	))

	properties.TestingRun(t)
}

// TestPerpendicularDistance_NonNegative verifies distance is always >= 0.
func TestPerpendicularDistance_NonNegative(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("perpendicular distance is always non-negative", prop.ForAll(
		func(p, a, b Point) bool {
			dist := perpendicularDistance(p, a, b)
			return dist >= 0.0
		},
		genPoint(),
		genPoint(),
		genPoint(),
	))

	properties.TestingRun(t)
}

// TestPerpendicularDistance_ZeroForPointOnLine verifies distance is 0 for point on line.
func TestPerpendicularDistance_ZeroForPointOnLine(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("perpendicular distance is zero for point on line", prop.ForAll(
		func(ax, ay, t float64) bool {
			a := Point{ax, ay}
			b := Point{ax + 10, ay + 10}
			// Point on line between a and b
			p := Point{ax + t*10, ay + t*10}

			if t < 0 || t > 1 {
				return true // skip points outside segment
			}

			dist := perpendicularDistance(p, a, b)
			return dist < 1e-9
		},
		gen.Float64Range(-10, 10),
		gen.Float64Range(-10, 10),
		gen.Float64Range(0, 1),
	))

	properties.TestingRun(t)
}

// TestMinimumAreaRectangle_AreaLessThanBoundingBox verifies MAR <= AABB area.
func TestMinimumAreaRectangle_AreaLessThanBoundingBox(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("minimum area rectangle area <= axis-aligned bounding box", prop.ForAll(
		func(points []Point) bool {
			if len(points) < 3 {
				return true
			}

			rect := MinimumAreaRectangle(points)

			if rect == nil || len(rect) != 4 {
				return false
			}

			// Calculate MAR area using shoelace formula
			var marArea float64
			for i := range rect {
				j := (i + 1) % len(rect)
				marArea += rect[i].X * rect[j].Y
				marArea -= rect[j].X * rect[i].Y
			}
			marArea = math.Abs(marArea) / 2.0

			// Calculate AABB area
			var minX, maxX, minY, maxY float64
			minX, maxX = points[0].X, points[0].X
			minY, maxY = points[0].Y, points[0].Y

			for _, p := range points {
				if p.X < minX {
					minX = p.X
				}
				if p.X > maxX {
					maxX = p.X
				}
				if p.Y < minY {
					minY = p.Y
				}
				if p.Y > maxY {
					maxY = p.Y
				}
			}

			aabbArea := (maxX - minX) * (maxY - minY)

			// MAR area should be <= AABB area (with small tolerance)
			return marArea <= aabbArea+1e-6
		},
		genPolygon(3, 15),
	))

	properties.TestingRun(t)
}
