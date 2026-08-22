package detector

import "github.com/MeKo-Tech/pogo/internal/utils"

// traceContourMoore extracts a boundary polygon for the given labeled component
// using Moore-Neighbor tracing. It restricts the search to the component's AABB
// from comp statistics for efficiency. Returned points are pixel-center coordinates.
func traceContourMoore(labels []int, w, h, label int, st compStats) []utils.Point {
	if label <= 0 || len(labels) != w*h {
		return nil
	}

	// Find starting boundary pixel
	sx, sy := findStartingBoundaryPixel(labels, w, h, label, st)
	if sx == -1 {
		return nil
	}

	// Initialize contour tracing
	pts := make([]utils.Point, 0, 64)
	cx, cy := sx, sy
	bx, by := sx-1, sy // backtrack to the left of start

	addPoint := func(x, y int) {
		p := utils.Point{X: float64(x), Y: float64(y)}
		n := len(pts)
		if n >= 2 {
			a := pts[n-2]
			b := pts[n-1]
			// Check collinearity: (b-a) x (p-b) == 0
			v1x, v1y := b.X-a.X, b.Y-a.Y
			v2x, v2y := p.X-b.X, p.Y-b.Y
			cross := v1x*v2y - v1y*v2x
			// Drop b only when it lies *between* a and p. A 180 degree
			// reversal is also cross == 0, but there b is the tip of a
			// one-pixel-wide spur and dropping it erases the spur.
			if cross == 0 && v1x*v2x+v1y*v2y > 0 {
				// remove middle point b
				pts = pts[:n-1]
			}
		}
		pts = append(pts, p)
	}

	addPoint(cx, cy)

	// Moore-Neighbor tracing
	startCx, startCy := cx, cy
	startBx, startBy := bx, by
	// A contour cannot be longer than four times the component's own bounding
	// box area, so bound the walk by that rather than by the whole probability
	// map: on a large canvas holding a few small components, w*h*4 lets a single
	// 100x7 component spin for millions of steps and emit a polygon with
	// hundreds of thousands of points.
	bw := st.maxX - st.minX + 1
	bh := st.maxY - st.minY + 1
	maxSteps := (bw+2)*(bh+2)*4 + 8

	return traceContourLoop(labels, w, h, label, &cx, &cy, &bx, &by,
		startCx, startCy, startBx, startBy, maxSteps, &pts, addPoint)
}

// traceContourLoop performs the main contour tracing loop.
//
// The walk is a deterministic function of the state (current pixel, backtrack
// cell), so the state sequence is eventually periodic. Jacob's criterion —
// "stop on re-entering the start pixel from the same direction" — is stated
// against the *initial* state, but that state is synthetic: the caller invents
// the backtrack as the cell west of the raster-scan start, and for shapes such
// as a one-pixel-wide run the walk re-enters the start pixel from the south
// instead and never reproduces it. The loop then only ends by exhausting
// maxSteps. Keying the criterion on the state after the first step instead
// makes it a state the walk demonstrably reaches, so the cycle always closes.
func traceContourLoop(labels []int, w, h, label int, cx, cy, bx, by *int,
	startCx, startCy, startBx, startBy, maxSteps int,
	pts *[]utils.Point, addPoint func(int, int),
) []utils.Point {
	var sentinelCx, sentinelCy, sentinelBx, sentinelBy int
	haveSentinel := false

	for range maxSteps {
		nx, ny, nbx, nby, found := findNextBoundaryPixel(labels, w, h, label, *cx, *cy, *bx, *by)
		if !found {
			break
		}

		if haveSentinel && hasReturnedToStart(nx, ny, nbx, nby,
			sentinelCx, sentinelCy, sentinelBx, sentinelBy) {
			// The cycle has closed; the point was already recorded on the
			// first pass, so stop before appending it a second time.
			break
		}

		*bx, *by = nbx, nby
		*cx, *cy = nx, ny

		if !haveSentinel {
			sentinelCx, sentinelCy, sentinelBx, sentinelBy = nx, ny, nbx, nby
			haveSentinel = true
		}

		// Append point if different from last
		if shouldAddPoint(*pts, *cx, *cy) {
			addPoint(*cx, *cy)
		}

		// The synthetic start state can still be reached on well-formed
		// contours, and it closes the loop one step earlier when it is.
		if hasReturnedToStart(*cx, *cy, *bx, *by, startCx, startCy, startBx, startBy) {
			break
		}
	}

	// Remove duplicated closing point if present
	removeDuplicateClosingPoint(pts)
	return *pts
}

// hasReturnedToStart checks if we've returned to the starting position.
func hasReturnedToStart(cx, cy, bx, by, startCx, startCy, startBx, startBy int) bool {
	return cx == startCx && cy == startCy && bx == startBx && by == startBy
}

// removeDuplicateClosingPoint removes the duplicate closing point if present.
func removeDuplicateClosingPoint(pts *[]utils.Point) {
	if len(*pts) >= 2 && (*pts)[0].X == (*pts)[len(*pts)-1].X && (*pts)[0].Y == (*pts)[len(*pts)-1].Y {
		*pts = (*pts)[:len(*pts)-1]
	}
}

// shouldAddPoint checks if a point should be added to avoid duplicates.
func shouldAddPoint(pts []utils.Point, x, y int) bool {
	if len(pts) == 0 {
		return true
	}
	last := pts[len(pts)-1]
	return last.X != float64(x) || last.Y != float64(y)
}

// findStartingBoundaryPixel finds the first boundary pixel within the component's AABB.
func findStartingBoundaryPixel(labels []int, w, h, label int, st compStats) (int, int) {
	// Find a starting boundary pixel within component bbox
	for y := st.minY; y <= st.maxY; y++ {
		for x := st.minX; x <= st.maxX; x++ {
			if isBoundaryPixel(labels, w, h, label, x, y) {
				return x, y
			}
		}
	}

	// Fallback: try any pixel of the label
	for y := st.minY; y <= st.maxY; y++ {
		for x := st.minX; x <= st.maxX; x++ {
			if isLabelPixel(labels, w, h, label, x, y) {
				return x, y
			}
		}
	}

	return -1, -1
}

// isBoundaryPixel checks if a pixel is a boundary pixel of the given label.
func isBoundaryPixel(labels []int, w, h, label, x, y int) bool {
	if !isLabelPixel(labels, w, h, label, x, y) {
		return false
	}
	return !isLabelPixel(labels, w, h, label, x+1, y) ||
		!isLabelPixel(labels, w, h, label, x-1, y) ||
		!isLabelPixel(labels, w, h, label, x, y+1) ||
		!isLabelPixel(labels, w, h, label, x, y-1)
}

// isLabelPixel checks if a pixel belongs to the given label.
func isLabelPixel(labels []int, w, h, label, x, y int) bool {
	if x < 0 || y < 0 || x >= w || y >= h {
		return false
	}
	return labels[y*w+x] == label
}

// findNextBoundaryPixel finds the next boundary pixel in the Moore neighborhood.
func findNextBoundaryPixel(labels []int, w, h, label int, cx, cy, bx, by int) (int, int, int, int, bool) {
	idx := func(x, y int) int { return y*w + x }
	inBounds := func(x, y int) bool { return x >= 0 && y >= 0 && x < w && y < h }
	isLabel := func(x, y int) bool {
		if !inBounds(x, y) {
			return false
		}
		return labels[idx(x, y)] == label
	}

	// 8-neighborhood clockwise order: E, SE, S, SW, W, NW, N, NE
	ndx := [8]int{1, 1, 0, -1, -1, -1, 0, 1}
	ndy := [8]int{0, 1, 1, 1, 0, -1, -1, -1}

	dirIndex := func(dx, dy int) int {
		for i := range 8 {
			if ndx[i] == dx && ndy[i] == dy {
				return i
			}
		}
		return 0
	}

	// Determine neighbor order start index relative to backtrack b
	dx, dy := bx-cx, by-cy
	start := (dirIndex(dx, dy) + 1) % 8

	// Search for next boundary pixel. The new backtrack is the last background
	// cell examined before the hit, which is what Jacob's stopping criterion in
	// traceContourLoop compares against: returning the previous *current* pixel
	// instead makes the backtrack always a labelled pixel, so it can never equal
	// the background cell the walk started from and the loop only ever ends by
	// exhausting maxSteps.
	for k := range 8 {
		i := (start + k) % 8
		tx, ty := cx+ndx[i], cy+ndy[i]
		if isLabel(tx, ty) {
			return tx, ty, bx, by, true
		}
		// advance b to this neighbor for clockwise scanning
		bx, by = tx, ty
	}

	return 0, 0, bx, by, false
}
