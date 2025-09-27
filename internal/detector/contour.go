package detector

import "github.com/MeKo-Tech/pogo/internal/utils"

// traceContourMoore extracts a boundary polygon for the given labeled component
// using Moore-Neighbor tracing. It restricts the search to the component's AABB
// from comp statistics for efficiency. Returned points are pixel-center coordinates.
func traceContourMoore(labels []int, w, h, label int, st compStats) []utils.Point {
	if label <= 0 || len(labels) != w*h {
		return nil
	}
	idx := func(x, y int) int { return y*w + x }
	inBounds := func(x, y int) bool { return x >= 0 && y >= 0 && x < w && y < h }
	isLabel := func(x, y int) bool {
		if !inBounds(x, y) {
			return false
		}
		return labels[idx(x, y)] == label
	}
	// Boundary if foreground and any 4-neighbor is background
	isBoundary := func(x, y int) bool {
		if !isLabel(x, y) {
			return false
		}
		if !isLabel(x+1, y) || !isLabel(x-1, y) || !isLabel(x, y+1) || !isLabel(x, y-1) {
			return true
		}
		return false
	}

	// Find a starting boundary pixel within component bbox
	sx, sy := -1, -1
	for y := st.minY; y <= st.maxY; y++ {
		for x := st.minX; x <= st.maxX; x++ {
			if isBoundary(x, y) {
				sx, sy = x, y
				break
			}
		}
		if sx != -1 {
			break
		}
	}
	if sx == -1 {
		// Fallback: try any pixel of the label
		for y := st.minY; y <= st.maxY && sy == -1; y++ {
			for x := st.minX; x <= st.maxX; x++ {
				if isLabel(x, y) {
					sx, sy = x, y
					break
				}
			}
		}
		if sx == -1 {
			return nil
		}
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

	// Initialize current c and backtrack b (to the left of start)
	cx, cy := sx, sy
	bx, by := sx-1, sy

	// Helper to append points (float coords at centers), removing collinear tails
	pts := make([]utils.Point, 0, 64)
	push := func(x, y int) {
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

	push(cx, cy)

	// Moore-Neighbor tracing
	startCx, startCy := cx, cy
	startBx, startBy := bx, by
	maxSteps := w*h*4 + 8
	steps := 0
	for {
		steps++
		if steps > maxSteps {
			break // safety
		}
		// Determine neighbor order start index relative to backtrack b
		dx, dy := bx-cx, by-cy
		start := (dirIndex(dx, dy) + 1) % 8
		found := false
		var nx, ny int
		for k := range 8 {
			i := (start + k) % 8
			tx, ty := cx+ndx[i], cy+ndy[i]
			if isLabel(tx, ty) {
				nx, ny = tx, ty
				found = true
				// Set new backtrack as previous current
				bx, by = cx, cy
				cx, cy = nx, ny
				// Append point
				if len(pts) == 0 || pts[len(pts)-1].X != float64(cx) || pts[len(pts)-1].Y != float64(cy) {
					push(cx, cy)
				}
				break
			}
			// advance b to this neighbor for clockwise scanning
			bx, by = tx, ty
		}
		if !found {
			break
		}
		if cx == startCx && cy == startCy && bx == startBx && by == startBy {
			break
		}
	}

	// Remove duplicated closing point if present
	if len(pts) >= 2 {
		if pts[0].X == pts[len(pts)-1].X && pts[0].Y == pts[len(pts)-1].Y {
			pts = pts[:len(pts)-1]
		}
	}
	return pts
}
