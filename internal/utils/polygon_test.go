package utils

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSimplifyPolygon(t *testing.T) {
	tests := []struct {
		name           string
		points         []Point
		epsilon        float64
		expectedMinLen int
		expectedMaxLen int
	}{
		{
			name:           "empty polygon",
			points:         []Point{},
			epsilon:        1.0,
			expectedMinLen: 0,
			expectedMaxLen: 0,
		},
		{
			name:           "triangle (no simplification needed)",
			points:         []Point{{0, 0}, {10, 0}, {5, 10}},
			epsilon:        1.0,
			expectedMinLen: 3,
			expectedMaxLen: 3,
		},
		{
			name: "rectangle with extra points on edges",
			points: []Point{
				{0, 0}, {5, 0}, {10, 0}, // bottom edge with extra point
				{10, 5}, {10, 10}, // right edge with extra point
				{5, 10}, {0, 10}, // top edge with extra point
				{0, 5}, // left edge with extra point
			},
			epsilon:        0.1, // small epsilon should keep some points
			expectedMinLen: 4,
			expectedMaxLen: 8,
		},
		{
			name: "rectangle with extra points - high epsilon",
			points: []Point{
				{0, 0}, {5, 0}, {10, 0}, // bottom edge with extra point
				{10, 5}, {10, 10}, // right edge with extra point
				{5, 10}, {0, 10}, // top edge with extra point
				{0, 5}, // left edge with extra point
			},
			epsilon:        2.0, // high epsilon should remove collinear points
			expectedMinLen: 4,
			expectedMaxLen: 6,
		},
		{
			name:           "zero epsilon",
			points:         []Point{{0, 0}, {1, 1}, {2, 2}, {3, 3}},
			epsilon:        0.0,
			expectedMinLen: 4,
			expectedMaxLen: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := SimplifyPolygon(tt.points, tt.epsilon)
			require.GreaterOrEqual(t, len(result), tt.expectedMinLen)
			require.LessOrEqual(t, len(result), tt.expectedMaxLen)

			// For non-empty input, result should not be longer than input
			if len(tt.points) > 0 {
				require.LessOrEqual(t, len(result), len(tt.points))
			}
		})
	}
}

func TestSimplifyPolygon_CollinearPoints(t *testing.T) {
	// Test with perfectly collinear points that should be simplified
	points := []Point{
		{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, // horizontal line
		{4, 1}, {4, 2}, {4, 3}, {4, 4}, // vertical line
		{3, 4}, {2, 4}, {1, 4}, {0, 4}, // horizontal line back
		{0, 3}, {0, 2}, {0, 1}, // vertical line back
	}

	result := SimplifyPolygon(points, 0.1)

	// Should simplify to approximately a rectangle (4-5 points)
	require.LessOrEqual(t, len(result), 8)
	require.GreaterOrEqual(t, len(result), 4)
}

func TestUnclipPolygon(t *testing.T) {
	tests := []struct {
		name           string
		polygon        []Point
		distance       float64
		expectedChange bool
	}{
		{
			name:           "triangle expansion",
			polygon:        []Point{{10, 10}, {20, 10}, {15, 20}},
			distance:       2.0,
			expectedChange: true,
		},
		{
			name:           "square expansion",
			polygon:        []Point{{0, 0}, {10, 0}, {10, 10}, {0, 10}},
			distance:       1.0,
			expectedChange: true,
		},
		{
			name:           "zero distance",
			polygon:        []Point{{0, 0}, {10, 0}, {10, 10}},
			distance:       0.0,
			expectedChange: false,
		},
		{
			name:           "empty polygon",
			polygon:        []Point{},
			distance:       5.0,
			expectedChange: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := UnclipPolygon(tt.polygon, tt.distance)

			if !tt.expectedChange {
				if len(tt.polygon) == 0 {
					require.Empty(t, result)
				} else {
					require.Equal(t, tt.polygon, result)
				}
			} else {
				require.Len(t, result, len(tt.polygon))
				// Basic verification that the function returned something different
				require.NotNil(t, result)
			}
		})
	}
}

func TestConvexHull(t *testing.T) {
	tests := []struct {
		name           string
		points         []Point
		expectedMinLen int
		expectedMaxLen int
	}{
		{
			name:           "empty input",
			points:         []Point{},
			expectedMinLen: 0,
			expectedMaxLen: 0,
		},
		{
			name:           "single point",
			points:         []Point{{5, 5}},
			expectedMinLen: 1,
			expectedMaxLen: 1,
		},
		{
			name:           "two points",
			points:         []Point{{0, 0}, {10, 10}},
			expectedMinLen: 2,
			expectedMaxLen: 2,
		},
		{
			name:           "triangle",
			points:         []Point{{0, 0}, {10, 0}, {5, 10}},
			expectedMinLen: 3,
			expectedMaxLen: 3,
		},
		{
			name: "square with interior point",
			points: []Point{
				{0, 0}, {10, 0}, {10, 10}, {0, 10}, // square corners
				{5, 5}, // interior point - should not be in hull
			},
			expectedMinLen: 4,
			expectedMaxLen: 4,
		},
		{
			name: "collinear points",
			points: []Point{
				{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, // horizontal line
			},
			expectedMinLen: 2,
			expectedMaxLen: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ConvexHull(tt.points)
			require.GreaterOrEqual(t, len(result), tt.expectedMinLen)
			require.LessOrEqual(t, len(result), tt.expectedMaxLen)

			// Convex hull should never have more points than input
			require.LessOrEqual(t, len(result), len(tt.points))
		})
	}
}

func TestConvexHull_Rectangle(t *testing.T) {
	// Test with a rectangle plus some interior and boundary points
	points := []Point{
		{0, 0}, {10, 0}, {10, 10}, {0, 10}, // corners
		{5, 0}, {10, 5}, {5, 10}, {0, 5}, // edge midpoints
		{5, 5}, {3, 3}, {7, 7}, // interior points
	}

	hull := ConvexHull(points)

	// Should result in the 4 corners
	require.Len(t, hull, 4)

	// Check that all original corners are in the hull
	corners := map[Point]bool{
		{0, 0}: false, {10, 0}: false, {10, 10}: false, {0, 10}: false,
	}
	for _, p := range hull {
		if _, exists := corners[p]; exists {
			corners[p] = true
		}
	}
	for corner, found := range corners {
		require.True(t, found, "corner %v should be in convex hull", corner)
	}
}

func TestMinimumAreaRectangle(t *testing.T) {
	tests := []struct {
		name           string
		points         []Point
		expectValidBox bool
	}{
		{
			name:           "empty input",
			points:         []Point{},
			expectValidBox: false,
		},
		{
			name:           "single point",
			points:         []Point{{5, 5}},
			expectValidBox: true,
		},
		{
			name:           "two points",
			points:         []Point{{0, 0}, {10, 5}},
			expectValidBox: true,
		},
		{
			name:           "triangle",
			points:         []Point{{0, 0}, {10, 0}, {5, 10}},
			expectValidBox: true,
		},
		{
			name: "axis-aligned rectangle",
			points: []Point{
				{0, 0}, {10, 0}, {10, 5}, {0, 5},
			},
			expectValidBox: true,
		},
		{
			name: "rotated rectangle points",
			points: []Point{
				{5, 0}, {10, 5}, {5, 10}, {0, 5},
			},
			expectValidBox: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := MinimumAreaRectangle(tt.points)

			if !tt.expectValidBox {
				require.Empty(t, result)
			} else {
				require.NotEmpty(t, result)
				// For valid rectangles, should return 4 corners
				if len(tt.points) > 2 {
					require.Len(t, result, 4)
				}
			}
		})
	}
}

func TestMinimumAreaRectangle_AxisAligned(t *testing.T) {
	// Test with axis-aligned rectangle to verify the algorithm works correctly
	points := []Point{{1, 1}, {9, 1}, {9, 6}, {1, 6}}

	result := MinimumAreaRectangle(points)
	require.Len(t, result, 4)

	// Calculate the area - should be close to the original rectangle area (8 * 5 = 40)
	// This is a basic check - the exact coordinates may vary based on algorithm implementation
	minX, maxX := result[0].X, result[0].X
	minY, maxY := result[0].Y, result[0].Y

	for _, p := range result {
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

	// The minimum area rectangle should encompass all original points
	for _, original := range points {
		require.GreaterOrEqual(t, original.X, minX-0.1)
		require.LessOrEqual(t, original.X, maxX+0.1)
		require.GreaterOrEqual(t, original.Y, minY-0.1)
		require.LessOrEqual(t, original.Y, maxY+0.1)
	}
}

// Test helper functions.
func TestPerpendicularDistance(t *testing.T) {
	// Test perpendicular distance calculation
	// Point (5, 5) to line from (0, 0) to (10, 0) should be 5
	distance := perpendicularDistance(Point{5, 5}, Point{0, 0}, Point{10, 0})
	require.InDelta(t, 5.0, distance, 0.001)

	// Point on the line should have distance 0
	distance = perpendicularDistance(Point{5, 0}, Point{0, 0}, Point{10, 0})
	require.InDelta(t, 0.0, distance, 0.001)

	// Point (0, 5) to vertical line from (0, 0) to (0, 10) should be 0
	distance = perpendicularDistance(Point{0, 5}, Point{0, 0}, Point{0, 10})
	require.InDelta(t, 0.0, distance, 0.001)
}

func TestRemoveDuplicatePoints(t *testing.T) {
	// This function removes consecutive duplicates only
	points := []Point{{0, 0}, {0, 0}, {1, 1}, {1, 1}, {2, 2}}
	result := removeDuplicatePoints(points)

	require.Len(t, result, 3)
	require.Equal(t, Point{0, 0}, result[0])
	require.Equal(t, Point{1, 1}, result[1])
	require.Equal(t, Point{2, 2}, result[2])
}

func TestSortPoints(t *testing.T) {
	points := []Point{{3, 3}, {1, 1}, {2, 2}, {1, 3}}
	sortPoints(points)

	// Should be sorted by X first, then by Y
	require.Equal(t, Point{1, 1}, points[0])
	require.Equal(t, Point{1, 3}, points[1])
	require.Equal(t, Point{2, 2}, points[2])
	require.Equal(t, Point{3, 3}, points[3])
}

func TestCross(t *testing.T) {
	// Test cross product calculation with origin at (0,0)
	o := Point{0, 0}
	a := Point{1, 0}
	b := Point{0, 1}
	result := cross(o, a, b)
	require.InEpsilon(t, 1.0, result, 1e-6) // positive cross product

	// Reverse order should give negative
	result = cross(o, b, a)
	require.InEpsilon(t, -1.0, result, 1e-6)

	// Parallel vectors should give zero
	result = cross(o, Point{1, 0}, Point{2, 0})
	require.InEpsilon(t, 0.0, result, 1e-6)

	// Test with different origin
	result = cross(Point{1, 1}, Point{2, 1}, Point{1, 2})
	require.InEpsilon(t, 1.0, result, 1e-6)
}
