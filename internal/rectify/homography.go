package rectify

import (
	"github.com/MeKo-Tech/pogo/internal/utils"
)

// computeHomography computes 3x3 matrix H mapping p[i] -> q[i]. Returns H as [9]float64.
func computeHomography(p, q [4]utils.Point) ([9]float64, bool) {
	// Build 8x8 system A*h = b for the 8 unknowns (h00..h21), h22=1.
	A := [8][8]float64{}
	b := [8]float64{}
	for i := range 4 {
		X, Y := p[i].X, p[i].Y
		x, y := q[i].X, q[i].Y
		r := 2 * i
		// x' = (h00 X + h01 Y + h02)/(h20 X + h21 Y + 1)
		A[r][0] = X
		A[r][1] = Y
		A[r][2] = 1
		A[r][3] = 0
		A[r][4] = 0
		A[r][5] = 0
		A[r][6] = -X * x
		A[r][7] = -Y * x
		b[r] = x

		// y' = (h10 X + h11 Y + h12)/(h20 X + h21 Y + 1)
		A[r+1][0] = 0
		A[r+1][1] = 0
		A[r+1][2] = 0
		A[r+1][3] = X
		A[r+1][4] = Y
		A[r+1][5] = 1
		A[r+1][6] = -X * y
		A[r+1][7] = -Y * y
		b[r+1] = y
	}

	// Solve using Gaussian elimination
	h, ok := solve8x8(A, b)
	if !ok {
		return [9]float64{}, false
	}
	H := [9]float64{h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1}

	return H, true
}

func solve8x8(a [8][8]float64, b [8]float64) ([8]float64, bool) {
	// Create working copies
	matrix := a
	vector := b

	// Forward elimination with partial pivoting
	for i := range 8 {
		if !pivotAndNormalize(&matrix, &vector, i) {
			return [8]float64{}, false
		}
		eliminateColumn(&matrix, &vector, i)
	}

	// Back substitution
	var x [8]float64
	for i := range 8 {
		x[i] = vector[i]
	}
	return x, true
}

func pivotAndNormalize(matrix *[8][8]float64, vector *[8]float64, col int) bool {
	// Find pivot row
	pivotRow := findPivotRow(*matrix, col)
	if pivotRow == -1 {
		return false
	}

	// Swap rows if needed
	if pivotRow != col {
		swapRows(matrix, vector, col, pivotRow)
	}

	// Normalize pivot row
	normalizeRow(matrix, vector, col)
	return true
}

func findPivotRow(matrix [8][8]float64, col int) int {
	maxAbs := abs(matrix[col][col])
	pivotRow := col
	for r := col + 1; r < 8; r++ {
		if abs(matrix[r][col]) > maxAbs {
			maxAbs = abs(matrix[r][col])
			pivotRow = r
		}
	}
	if maxAbs == 0 {
		return -1
	}
	return pivotRow
}

func swapRows(matrix *[8][8]float64, vector *[8]float64, row1, row2 int) {
	matrix[row1], matrix[row2] = matrix[row2], matrix[row1]
	vector[row1], vector[row2] = vector[row2], vector[row1]
}

func normalizeRow(matrix *[8][8]float64, vector *[8]float64, row int) {
	div := matrix[row][row]
	for c := row; c < 8; c++ {
		matrix[row][c] /= div
	}
	vector[row] /= div
}

func eliminateColumn(matrix *[8][8]float64, vector *[8]float64, col int) {
	for r := range 8 {
		if r == col {
			continue
		}
		factor := matrix[r][col]
		if factor == 0 {
			continue
		}
		for c := col; c < 8; c++ {
			matrix[r][c] -= factor * matrix[col][c]
		}
		vector[r] -= factor * vector[col]
	}
}

func applyHomography(h [9]float64, x, y float64) (float64, float64) {
	denom := h[6]*x + h[7]*y + h[8]
	if denom == 0 {
		return -1e9, -1e9
	}
	sx := (h[0]*x + h[1]*y + h[2]) / denom
	sy := (h[3]*x + h[4]*y + h[5]) / denom
	return sx, sy
}
