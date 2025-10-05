package detector

import (
	"testing"
)

func TestPostProcessDB_EmptyOutput(t *testing.T) {
	// All zeros prob map should yield no regions for any positive threshold
	w, h := 64, 32
	prob := make([]float32, w*h)
	regs := PostProcessDB(prob, w, h, 0.1, 0.5)
	if len(regs) != 0 {
		t.Fatalf("expected 0 regions for empty prob map, got %d", len(regs))
	}
}

func TestPostProcessDB_AllOnesSingleRegion(t *testing.T) {
	// All ones should produce a single full-image region
	w, h := 32, 16
	prob := make([]float32, w*h)
	for i := range prob {
		prob[i] = 1.0
	}
	regs := PostProcessDB(prob, w, h, 0.5, 0.5)
	if len(regs) != 1 {
		t.Fatalf("expected 1 region, got %d", len(regs))
	}
	r := regs[0]
	if r.Confidence < 0.99 {
		t.Errorf("expected high confidence near 1.0, got %f", r.Confidence)
	}
	if r.Box.MinX < 0 || r.Box.MinY < 0 || r.Box.MaxX > float64(w) || r.Box.MaxY > float64(h) {
		t.Errorf("box out of bounds: %+v (bounds %dx%d)", r.Box, w, h)
	}
}

func TestPostProcessDB_ExtremeAspectRatios(t *testing.T) {
	tests := []struct{ w, h int }{{256, 8}, {8, 256}, {320, 12}, {12, 320}}
	for _, tc := range tests {
		prob := make([]float32, tc.w*tc.h)
		// create a horizontal/vertical stripe over threshold
		for y := 0; y < tc.h; y++ {
			for x := 0; x < tc.w; x++ {
				if tc.w >= tc.h {
					// thin horizontal stripe
					if y >= tc.h/3 && y < 2*tc.h/3 {
						prob[y*tc.w+x] = 0.9
					} else {
						prob[y*tc.w+x] = 0.1
					}
				} else {
					// thin vertical stripe
					if x >= tc.w/3 && x < 2*tc.w/3 {
						prob[y*tc.w+x] = 0.9
					} else {
						prob[y*tc.w+x] = 0.1
					}
				}
			}
		}
		regs := PostProcessDB(prob, tc.w, tc.h, 0.5, 0.3)
		if len(regs) < 1 {
			t.Fatalf("%dx%d: expected at least 1 region", tc.w, tc.h)
		}
		// Validate boxes are within bounds
		for _, r := range regs {
			if r.Box.MinX < 0 || r.Box.MinY < 0 || r.Box.MaxX > float64(tc.w) || r.Box.MaxY > float64(tc.h) {
				t.Errorf("%dx%d: box out of bounds: %+v", tc.w, tc.h, r.Box)
			}
		}
	}
}
