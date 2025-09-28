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
			if cross == 0 {
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
	maxSteps := w*h*4 + 8

	return traceContourLoop(labels, w, h, label, &cx, &cy, &bx, &by,
		startCx, startCy, startBx, startBy, maxSteps, &pts, addPoint)
}

// traceContourLoop performs the main contour tracing loop.
func traceContourLoop(labels []int, w, h, label int, cx, cy, bx, by *int,
	startCx, startCy, startBx, startBy, maxSteps int,
	pts *[]utils.Point, addPoint func(int, int)) []utils.Point {
	steps := 0

	for steps < maxSteps {
		steps++

		nx, ny, nbx, nby, found := findNextBoundaryPixel(labels, w, h, label, *cx, *cy, *bx, *by)
		if !found {
			break
		}

		// Set new backtrack as previous current
		*bx, *by = nbx, nby
		*cx, *cy = nx, ny

		// Append point if different from last
		if shouldAddPoint(*pts, *cx, *cy) {
			addPoint(*cx, *cy)
		}

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

	// Search for next boundary pixel
	for k := range 8 {
		i := (start + k) % 8
		tx, ty := cx+ndx[i], cy+ndy[i]
		if isLabel(tx, ty) {
			return tx, ty, cx, cy, true
		}
		// advance b to this neighbor for clockwise scanning
		bx, by = tx, ty
	}

	return 0, 0, bx, by, false
}
