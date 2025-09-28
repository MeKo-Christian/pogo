package rectify

import (
	"testing"

	"github.com/MeKo-Tech/pogo/internal/utils"
)

// TestComputeHomography tests homography computation.
func TestComputeHomography(t *testing.T) {
	// Test identity transformation (points map to themselves)
	p := [4]utils.Point{
		{X: 0, Y: 0},
		{X: 100, Y: 0},
		{X: 100, Y: 100},
		{X: 0, Y: 100},
	}
	q := p // identity

	h, ok := computeHomography(p, q)
	if !ok {
		t.Error("Expected homography computation to succeed")
	}

	// Verify identity properties (approximately)
	if abs(h[0]-1) > 1e-6 || abs(h[4]-1) > 1e-6 || abs(h[8]-1) > 1e-6 {
		t.Errorf("Expected identity matrix, got %v", h)
	}
}

// TestApplyHomography tests homography application.
func TestApplyHomography(t *testing.T) {
	// Identity matrix
	h := [9]float64{1, 0, 0, 0, 1, 0, 0, 0, 1}

	x, y := applyHomography(h, 10, 20)
	if abs(x-10) > 1e-6 || abs(y-20) > 1e-6 {
		t.Errorf("Expected (10,20), got (%f,%f)", x, y)
	}

	// Test with zero denominator (should return large negative values)
	h[8] = 0 // set denominator to zero
	x, y = applyHomography(h, 0, 0)
	if x >= -1e8 || y >= -1e8 {
		t.Error("Expected large negative values for zero denominator")
	}
}

// TestSolve8x8 tests the 8x8 linear system solver.
func TestSolve8x8(t *testing.T) {
	// Test with identity matrix
	a := [8][8]float64{}
	b := [8]float64{}
	for i := range 8 {
		a[i][i] = 1.0
		b[i] = float64(i + 1)
	}

	x, ok := solve8x8(a, b)
	if !ok {
		t.Error("Expected solve8x8 to succeed with identity matrix")
	}

	for i, v := range x {
		expected := float64(i + 1)
		if abs(v-expected) > 1e-6 {
			t.Errorf("x[%d] = %f, expected %f", i, v, expected)
		}
	}

	// Test with singular matrix (should fail)
	singular := [8][8]float64{}
	for i := range 8 {
		for j := range 8 {
			singular[i][j] = 1.0 // all ones - singular
		}
	}
	_, ok = solve8x8(singular, b)
	if ok {
		t.Error("Expected solve8x8 to fail with singular matrix")
	}
}

// TestPivotAndNormalize tests pivot and normalization.
func TestPivotAndNormalize(t *testing.T) {
	matrix := [8][8]float64{
		{0, 1, 0},
		{1, 0, 0},
		{0, 0, 1},
	}
	vector := [8]float64{1, 2, 3}

	// This should work for a simple case
	ok := pivotAndNormalize(&matrix, &vector, 0)
	if !ok {
		t.Error("Expected pivotAndNormalize to succeed")
	}
}

// TestFindPivotRow tests pivot row finding.
func TestFindPivotRow(t *testing.T) {
	matrix := [8][8]float64{
		{0, 0, 0},
		{0, 1, 0},
		{0, 0, 2},
	}

	pivot := findPivotRow(matrix, 1)
	if pivot != 1 {
		t.Errorf("Expected pivot row 1, got %d", pivot)
	}

	pivot = findPivotRow(matrix, 0)
	if pivot != -1 {
		t.Error("Expected no pivot for column 0")
	}
}

// TestSwapRows tests row swapping.
func TestSwapRows(t *testing.T) {
	matrix := [8][8]float64{
		{1, 0},
		{0, 1},
	}
	vector := [8]float64{10, 20}

	swapRows(&matrix, &vector, 0, 1)

	if matrix[0][0] != 0 || matrix[0][1] != 1 || matrix[1][0] != 1 || matrix[1][1] != 0 {
		t.Error("Rows not swapped correctly")
	}
	if vector[0] != 20 || vector[1] != 10 {
		t.Error("Vector elements not swapped correctly")
	}
}
