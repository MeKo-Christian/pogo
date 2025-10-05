package detector

import (
	"testing"

	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
)

// genLabeledImage generates a labeled connected component image for testing.
func genLabeledImage(width, height int) ([]int, compStats) {
	labels := make([]int, width*height)

	// Create a simple rectangular region
	minX, minY := width/4, height/4
	maxX, maxY := 3*width/4, 3*height/4

	pixelCount := 0
	for y := minY; y < maxY; y++ {
		for x := minX; x < maxX; x++ {
			labels[y*width+x] = 1
			pixelCount++
		}
	}

	stats := compStats{
		count: pixelCount,
		sum:   float64(pixelCount) * 0.9, // Average probability
		minX:  minX,
		maxX:  maxX,
		minY:  minY,
		maxY:  maxY,
	}

	return labels, stats
}

// TestTraceContourMoore_ClosedContourProperty verifies contours form valid polygons.
func TestTraceContourMoore_ClosedContourProperty(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("contour forms a valid polygon boundary", prop.ForAll(
		func(width, height int) bool {
			if width < 30 || height < 30 || width > 100 || height > 100 {
				return true // skip invalid dimensions
			}

			labels, stats := genLabeledImage(width, height)
			contour := traceContourMoore(labels, width, height, 1, stats)

			if len(contour) < 3 {
				return true // too few points to form a meaningful contour
			}

			// The Moore algorithm with collinearity removal may produce a simplified
			// polygon (e.g., just the 4 corners of a rectangle). The key property is
			// that the contour should trace the component boundary, which means:
			// 1. All points should be on or near the boundary
			// 2. The polygon should be non-degenerate (has area)
			// 3. Points should be ordered to form a valid traversal

			// Check that polygon is non-degenerate by verifying it has reasonable perimeter
			perimeter := 0.0
			for i := range contour {
				next := (i + 1) % len(contour)
				dx := contour[next].X - contour[i].X
				dy := contour[next].Y - contour[i].Y
				perimeter += dx*dx + dy*dy // squared distance is fine for comparison
			}

			// For a rectangular region of size (maxX-minX) x (maxY-minY),
			// the minimum perimeter would be 4 * side (if it's a line),
			// and we should have something reasonable
			regionSize := float64((stats.maxX - stats.minX) + (stats.maxY - stats.minY))

			// Perimeter should be at least proportional to region size
			return perimeter > regionSize
		},
		gen.IntRange(30, 80),
		gen.IntRange(30, 80),
	))

	properties.TestingRun(t)
}

// TestTraceContourMoore_WithinBounds verifies all contour points are within image bounds.
func TestTraceContourMoore_WithinBounds(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("all contour points are within image bounds", prop.ForAll(
		func(width, height int) bool {
			if width < 20 || height < 20 || width > 100 || height > 100 {
				return true
			}

			labels, stats := genLabeledImage(width, height)
			contour := traceContourMoore(labels, width, height, 1, stats)

			for _, pt := range contour {
				if pt.X < 0 || pt.Y < 0 {
					return false
				}
				if pt.X >= float64(width) || pt.Y >= float64(height) {
					return false
				}
			}
			return true
		},
		gen.IntRange(20, 80),
		gen.IntRange(20, 80),
	))

	properties.TestingRun(t)
}

// TestTraceContourMoore_NonEmpty verifies non-empty regions produce non-empty contours.
func TestTraceContourMoore_NonEmpty(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("non-empty labeled regions produce non-empty contours", prop.ForAll(
		func(width, height int) bool {
			if width < 20 || height < 20 || width > 100 || height > 100 {
				return true
			}

			labels, stats := genLabeledImage(width, height)
			if stats.count == 0 {
				return true // skip empty regions
			}

			contour := traceContourMoore(labels, width, height, 1, stats)

			// Non-empty region should have non-empty contour
			return len(contour) > 0
		},
		gen.IntRange(20, 80),
		gen.IntRange(20, 80),
	))

	properties.TestingRun(t)
}

// TestTraceContourMoore_NoDuplicateConsecutivePoints verifies no consecutive duplicates.
func TestTraceContourMoore_NoDuplicateConsecutivePoints(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("contour has no consecutive duplicate points", prop.ForAll(
		func(width, height int) bool {
			if width < 20 || height < 20 || width > 100 || height > 100 {
				return true
			}

			labels, stats := genLabeledImage(width, height)
			contour := traceContourMoore(labels, width, height, 1, stats)

			if len(contour) < 2 {
				return true
			}

			// Check for consecutive duplicates
			for i := 1; i < len(contour); i++ {
				if contour[i].X == contour[i-1].X && contour[i].Y == contour[i-1].Y {
					return false
				}
			}
			return true
		},
		gen.IntRange(20, 80),
		gen.IntRange(20, 80),
	))

	properties.TestingRun(t)
}

// TestTraceContourMoore_ReasonableLength verifies contour length is reasonable.
func TestTraceContourMoore_ReasonableLength(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("contour length is reasonable relative to region", prop.ForAll(
		func(width, height int) bool {
			if width < 20 || height < 20 || width > 100 || height > 100 {
				return true
			}

			labels, stats := genLabeledImage(width, height)
			contour := traceContourMoore(labels, width, height, 1, stats)

			// Contour should have at least 4 points for a rectangular region
			// and at most perimeter + some tolerance
			regionWidth := stats.maxX - stats.minX
			regionHeight := stats.maxY - stats.minY
			expectedPerimeter := 2 * (regionWidth + regionHeight)

			// Allow generous tolerance for jagged edges from pixel tracing
			maxExpected := expectedPerimeter * 6

			return len(contour) >= 4 && len(contour) <= maxExpected
		},
		gen.IntRange(20, 80),
		gen.IntRange(20, 80),
	))

	properties.TestingRun(t)
}

// TestFindStartingBoundaryPixel_FindsValidPixel verifies starting pixel is valid.
func TestFindStartingBoundaryPixel_FindsValidPixel(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("starting boundary pixel is within region bounds", prop.ForAll(
		func(width, height int) bool {
			if width < 20 || height < 20 || width > 100 || height > 100 {
				return true
			}

			labels, stats := genLabeledImage(width, height)
			sx, sy := findStartingBoundaryPixel(labels, width, height, 1, stats)

			if sx == -1 || sy == -1 {
				return stats.count == 0 // should only fail for empty regions
			}

			// Starting pixel should be within region bounds
			if sx < stats.minX || sx > stats.maxX {
				return false
			}
			if sy < stats.minY || sy > stats.maxY {
				return false
			}

			// Starting pixel should be labeled
			return labels[sy*width+sx] == 1
		},
		gen.IntRange(20, 80),
		gen.IntRange(20, 80),
	))

	properties.TestingRun(t)
}

// validateBoundaryPixel checks if a boundary pixel has at least one non-labeled neighbor.
func validateBoundaryPixel(labels []int, width, height, label, x, y int) bool {
	hasNonLabeledNeighbor := false
	for _, dy := range []int{-1, 0, 1} {
		for _, dx := range []int{-1, 0, 1} {
			if dx == 0 && dy == 0 {
				continue
			}
			nx, ny := x+dx, y+dy
			if !isLabelPixel(labels, width, height, label, nx, ny) {
				hasNonLabeledNeighbor = true
				break
			}
		}
		if hasNonLabeledNeighbor {
			break
		}
	}
	return hasNonLabeledNeighbor
}

// TestIsBoundaryPixel_Correctness verifies boundary detection.
func TestIsBoundaryPixel_Correctness(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("boundary pixels have at least one non-labeled neighbor", prop.ForAll(
		func(width, height int) bool {
			if width < 20 || height < 20 || width > 100 || height > 100 {
				return true
			}

			labels, stats := genLabeledImage(width, height)

			// Check all pixels in the region
			for y := stats.minY; y <= stats.maxY; y++ {
				for x := stats.minX; x <= stats.maxX; x++ {
					if labels[y*width+x] != 1 {
						continue
					}

					isBoundary := isBoundaryPixel(labels, width, height, 1, x, y)

					// If it's marked as boundary, check it has non-labeled neighbor
					if isBoundary && !validateBoundaryPixel(labels, width, height, 1, x, y) {
						return false
					}
				}
			}
			return true
		},
		gen.IntRange(20, 80),
		gen.IntRange(20, 80),
	))

	properties.TestingRun(t)
}

// TestIsLabelPixel_BoundsChecking verifies bounds checking works correctly.
func TestIsLabelPixel_BoundsChecking(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("out-of-bounds pixels return false", prop.ForAll(
		func(width, height int) bool {
			if width < 20 || height < 20 || width > 100 || height > 100 {
				return true
			}

			labels, _ := genLabeledImage(width, height)

			// Test out-of-bounds coordinates
			testCases := []struct{ x, y int }{
				{-1, 0},
				{0, -1},
				{width, 0},
				{0, height},
				{width + 10, height + 10},
			}

			for _, tc := range testCases {
				if isLabelPixel(labels, width, height, 1, tc.x, tc.y) {
					return false
				}
			}
			return true
		},
		gen.IntRange(20, 80),
		gen.IntRange(20, 80),
	))

	properties.TestingRun(t)
}

// TestTraceContourMoore_Deterministic verifies same input produces same output.
func TestTraceContourMoore_Deterministic(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("contour tracing is deterministic", prop.ForAll(
		func(width, height int) bool {
			if width < 20 || height < 20 || width > 100 || height > 100 {
				return true
			}

			labels, stats := genLabeledImage(width, height)

			contour1 := traceContourMoore(labels, width, height, 1, stats)
			contour2 := traceContourMoore(labels, width, height, 1, stats)

			if len(contour1) != len(contour2) {
				return false
			}

			for i := range contour1 {
				if contour1[i].X != contour2[i].X || contour1[i].Y != contour2[i].Y {
					return false
				}
			}
			return true
		},
		gen.IntRange(20, 80),
		gen.IntRange(20, 80),
	))

	properties.TestingRun(t)
}

// TestFindNextBoundaryPixel_ReturnsValidNeighbor verifies next pixel is adjacent.
func TestFindNextBoundaryPixel_ReturnsValidNeighbor(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("next boundary pixel is 8-connected neighbor", prop.ForAll(
		func(width, height int) bool {
			if width < 20 || height < 20 || width > 100 || height > 100 {
				return true
			}

			labels, stats := genLabeledImage(width, height)
			sx, sy := findStartingBoundaryPixel(labels, width, height, 1, stats)

			if sx == -1 || sy == -1 {
				return true // no valid starting pixel
			}

			nx, ny, _, _, found := findNextBoundaryPixel(labels, width, height, 1, sx, sy, sx-1, sy)

			if !found {
				return true // no next pixel found
			}

			// Next pixel should be 8-connected (distance <= sqrt(2))
			dx := nx - sx
			dy := ny - sy
			distSq := dx*dx + dy*dy

			return distSq >= 1 && distSq <= 2 // 8-connected neighbor
		},
		gen.IntRange(20, 80),
		gen.IntRange(20, 80),
	))

	properties.TestingRun(t)
}
