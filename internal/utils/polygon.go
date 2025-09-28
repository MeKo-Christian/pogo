package utils

import "math"

// SimplifyPolygon reduces the number of points in a polygon using the
// Douglas–Peucker algorithm with the given tolerance epsilon.
// The polygon is treated as closed for simplification continuity.
func SimplifyPolygon(pts []Point, epsilon float64) []Point {
	if len(pts) <= 3 || epsilon <= 0 {
		return append([]Point(nil), pts...)
	}
	// Work on an open polyline by duplicating first at end; we'll close implicitly
	open := append([]Point(nil), pts...)
	// Run DP on the open sequence
	keep := make([]bool, len(open))
	for i := range keep {
		keep[i] = false
	}
	dpSimplify(open, 0, len(open)-1, epsilon, keep)
	// Always keep endpoints to ensure closure continuity
	keep[0] = true
	keep[len(open)-1] = true
	out := make([]Point, 0, len(open))
	for i, k := range keep {
		if k {
			out = append(out, open[i])
		}
	}
	return out
}

func dpSimplify(pts []Point, start, end int, eps float64, keep []bool) {
	if end <= start+1 {
		return
	}
	maxDist := -1.0
	index := -1
	a := pts[start]
	b := pts[end]
	for i := start + 1; i < end; i++ {
		d := perpendicularDistance(pts[i], a, b)
		if d > maxDist {
			maxDist = d
			index = i
		}
	}
	if maxDist > eps {
		// Keep the farthest point and recurse
		dpSimplify(pts, start, index, eps, keep)
		keep[index] = true
		dpSimplify(pts, index, end, eps, keep)
	}
}

func perpendicularDistance(p, a, b Point) float64 {
	// Distance from point p to segment ab
	vx, vy := b.X-a.X, b.Y-a.Y
	if vx == 0 && vy == 0 {
		dx, dy := p.X-a.X, p.Y-a.Y
		return math.Hypot(dx, dy)
	}
	// Area of parallelogram / base length
	num := math.Abs((p.X-a.X)*vy - (p.Y-a.Y)*vx)
	den := math.Hypot(vx, vy)
	return num / den
}

// UnclipPolygon scales polygon points outward from the centroid by scale (>1 grows).
// For scale <= 0, returns a copy of the input. Degenerate polygons are returned as-is.
func UnclipPolygon(pts []Point, scale float64) []Point {
	if len(pts) == 0 || scale == 1.0 {
		return append([]Point(nil), pts...)
	}
	if scale <= 0 {
		return append([]Point(nil), pts...)
	}
	// Centroid as average of points (sufficient for unclip heuristic)
	cx, cy := 0.0, 0.0
	for _, p := range pts {
		cx += p.X
		cy += p.Y
	}
	cx /= float64(len(pts))
	cy /= float64(len(pts))
	out := make([]Point, len(pts))
	for i, p := range pts {
		dx := p.X - cx
		dy := p.Y - cy
		out[i] = Point{X: cx + dx*scale, Y: cy + dy*scale}
	}
	return out
}

// ConvexHull computes the convex hull of a set of points using the
// monotone chain algorithm. Returns the hull in CCW order without
// duplicating the first point at the end.
func ConvexHull(pts []Point) []Point {
	n := len(pts)
	if n <= 1 {
		return append([]Point(nil), pts...)
	}
	// Copy and sort by X then Y
	p := make([]Point, n)
	copy(p, pts)
	sortPoints(p)
	// Remove duplicates
	p = removeDuplicatePoints(p)
	n = len(p)
	if n <= 1 {
		return append([]Point(nil), p...)
	}
	lower := buildLowerHull(p)
	upper := buildUpperHull(p)
	// Concatenate lower and upper to get full hull, excluding last point of each (duplicate)
	hull := make([]Point, 0, len(lower)+len(upper)-2)
	hull = append(hull, lower[:len(lower)-1]...)
	hull = append(hull, upper[:len(upper)-1]...)
	return hull
}

func removeDuplicatePoints(p []Point) []Point {
	q := p[:0]
	var last Point
	hasLast := false
	for _, pt := range p {
		if !hasLast || pt.X != last.X || pt.Y != last.Y {
			q = append(q, pt)
			last = pt
			hasLast = true
		}
	}
	return q
}

func buildLowerHull(p []Point) []Point {
	lower := make([]Point, 0, len(p))
	for _, pt := range p {
		for len(lower) >= 2 && cross(lower[len(lower)-2], lower[len(lower)-1], pt) <= 0 {
			lower = lower[:len(lower)-1]
		}
		lower = append(lower, pt)
	}
	return lower
}

func buildUpperHull(p []Point) []Point {
	upper := make([]Point, 0, len(p))
	for i := len(p) - 1; i >= 0; i-- {
		pt := p[i]
		for len(upper) >= 2 && cross(upper[len(upper)-2], upper[len(upper)-1], pt) <= 0 {
			upper = upper[:len(upper)-1]
		}
		upper = append(upper, pt)
	}
	return upper
}

func sortPoints(p []Point) {
	// simple insertion sort since n is usually small
	for i := 1; i < len(p); i++ {
		v := p[i]
		j := i - 1
		for j >= 0 && (p[j].X > v.X || (p[j].X == v.X && p[j].Y > v.Y)) {
			p[j+1] = p[j]
			j--
		}
		p[j+1] = v
	}
}

func cross(o, a, b Point) float64 {
	return (a.X-o.X)*(b.Y-o.Y) - (a.Y-o.Y)*(b.X-o.X)
}

// MinimumAreaRectangle computes the minimum-area enclosing rectangle using a
// rotating calipers approach over the convex hull. Returns 4 points in CCW order.
// Falls back to axis-aligned bounding box for degenerate cases.
func MinimumAreaRectangle(pts []Point) []Point {
	if len(pts) == 0 {
		return nil
	}
	hull := ConvexHull(pts)
	if len(hull) == 0 {
		return nil
	}
	if len(hull) == 1 {
		return rectangleForSinglePoint(hull[0])
	}
	if len(hull) == 2 {
		return rectangleForTwoPoints(hull[0], hull[1])
	}
	return findMinimumAreaRectangle(hull)
}

func rectangleForSinglePoint(p Point) []Point {
	return []Point{{p.X, p.Y}, {p.X + 1, p.Y}, {p.X + 1, p.Y + 1}, {p.X, p.Y + 1}}
}

func rectangleForTwoPoints(a, b Point) []Point {
	// Create a thin rectangle around the segment
	return []Point{a, b, {b.X, b.Y + 1}, {a.X, a.Y + 1}}
}

func findMinimumAreaRectangle(hull []Point) []Point {
	bestArea := math.Inf(1)
	var bestU, bestV Point
	var bestMinS, bestMaxS, bestMinT, bestMaxT float64
	// For each edge as orientation
	for i := range hull {
		a := hull[i]
		b := hull[(i+1)%len(hull)]
		dx := b.X - a.X
		dy := b.Y - a.Y
		L := math.Hypot(dx, dy)
		if L == 0 {
			continue
		}
		ux, uy := dx/L, dy/L
		vx, vy := -uy, ux // perpendicular
		// Project all points
		minS, maxS := math.Inf(1), math.Inf(-1)
		minT, maxT := math.Inf(1), math.Inf(-1)
		for _, p := range hull {
			s := p.X*ux + p.Y*uy
			t := p.X*vx + p.Y*vy
			if s < minS {
				minS = s
			}
			if s > maxS {
				maxS = s
			}
			if t < minT {
				minT = t
			}
			if t > maxT {
				maxT = t
			}
		}
		area := (maxS - minS) * (maxT - minT)
		if area < bestArea {
			bestArea = area
			bestU = Point{ux, uy}
			bestV = Point{vx, vy}
			bestMinS, bestMaxS, bestMinT, bestMaxT = minS, maxS, minT, maxT
		}
	}
	// Reconstruct rectangle corners c0..c3 in world coordinates
	c0 := Point{X: bestU.X*bestMinS + bestV.X*bestMinT, Y: bestU.Y*bestMinS + bestV.Y*bestMinT}
	c1 := Point{X: bestU.X*bestMaxS + bestV.X*bestMinT, Y: bestU.Y*bestMaxS + bestV.Y*bestMinT}
	c2 := Point{X: bestU.X*bestMaxS + bestV.X*bestMaxT, Y: bestU.Y*bestMaxS + bestV.Y*bestMaxT}
	c3 := Point{X: bestU.X*bestMinS + bestV.X*bestMaxT, Y: bestU.Y*bestMinS + bestV.Y*bestMaxT}
	return []Point{c0, c1, c2, c3}
}
