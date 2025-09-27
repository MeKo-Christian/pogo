package detector

import (
	"testing"

	"github.com/MeKo-Tech/pogo/internal/utils"
)

func TestPostProcessDB_TwoBlobs(t *testing.T) {
	// Create a 8x4 probability map with two simple blocks above threshold
	w, h := 8, 4
	prob := make([]float32, w*h)
	set := func(x, y int, v float32) { prob[y*w+x] = v }
	// Blob 1: rectangle (1..2, 1..2)
	set(1, 1, 0.9)
	set(2, 1, 0.85)
	set(1, 2, 0.92)
	set(2, 2, 0.88)
	// Blob 2: rectangle (5..6, 0..1)
	set(5, 0, 0.95)
	set(6, 0, 0.91)
	set(5, 1, 0.93)
	set(6, 1, 0.89)

	regions := PostProcessDB(prob, w, h, 0.5, 0.6)
	if len(regions) != 2 {
		t.Fatalf("expected 2 regions, got %d", len(regions))
	}
	// Check boxes roughly match
	r0 := regions[0].Box
	r1 := regions[1].Box
	// Ensure widths/heights are positive
	if r0.Width() <= 0 || r0.Height() <= 0 || r1.Width() <= 0 || r1.Height() <= 0 {
		t.Fatalf("invalid region boxes: %+v, %+v", r0, r1)
	}
}

func TestScaleRegionsToOriginal(t *testing.T) {
	regs := []DetectedRegion{{
		Polygon:    []utils.Point{{X: 1, Y: 1}, {X: 3, Y: 1}, {X: 3, Y: 3}, {X: 1, Y: 3}},
		Box:        utils.NewBox(1, 1, 3, 3),
		Confidence: 0.9,
	}}
	scaled := ScaleRegionsToOriginal(regs, 10, 10, 100, 50)
	if len(scaled) != 1 {
		t.Fatalf("expected 1 region")
	}
	// Expect scaling by sx=10, sy=5
	b := scaled[0].Box
	if int(b.MinX) != 10 || int(b.MinY) != 5 || int(b.MaxX) != 30 || int(b.MaxY) != 15 {
		t.Fatalf("unexpected scaled box: %+v", b)
	}
}

func TestNonMaxSuppression(t *testing.T) {
	regs := []DetectedRegion{
		{Box: utils.NewBox(0, 0, 10, 10), Confidence: 0.9},
		{Box: utils.NewBox(1, 1, 9, 9), Confidence: 0.8}, // heavy overlap with #1
		{Box: utils.NewBox(20, 20, 30, 30), Confidence: 0.7},
	}
	kept := NonMaxSuppression(regs, 0.5)
	if len(kept) != 2 {
		t.Fatalf("expected 2 kept regions after NMS, got %d", len(kept))
	}
	if kept[0].Confidence < kept[1].Confidence {
		t.Fatalf("kept regions not sorted by confidence")
	}
}

func TestSoftNonMaxSuppression(t *testing.T) {
	regs := []DetectedRegion{
		{Box: utils.NewBox(0, 0, 10, 10), Confidence: 0.9},
		{Box: utils.NewBox(1, 1, 9, 9), Confidence: 0.8}, // heavy overlap
		{Box: utils.NewBox(20, 20, 30, 30), Confidence: 0.7},
	}
	// Linear Soft-NMS should keep all with decayed score for overlapping one
	kept := SoftNonMaxSuppression(regs, "linear", 0.5, 0.0, 0.1)
	if len(kept) != 3 {
		t.Fatalf("expected 3 kept regions after Soft-NMS, got %d", len(kept))
	}
	// Gaussian should also keep all, ordering by confidence
	keptG := SoftNonMaxSuppression(regs, "gaussian", 0.5, 0.5, 0.1)
	if len(keptG) != 3 {
		t.Fatalf("expected 3 kept regions after Gaussian Soft-NMS, got %d", len(keptG))
	}
}

func TestPostProcessDB_ContourPolygon_MoreThan4Points(t *testing.T) {
	// Create a 12x12 probability map with an L-shaped 1px-wide region
	w, h := 12, 12
	prob := make([]float32, w*h)
	set := func(x, y int, v float32) { prob[y*w+x] = v }
	// Horizontal segment from (2,2) to (9,2)
	for x := 2; x <= 9; x++ {
		set(x, 2, 0.9)
	}
	// Vertical segment from (2,3) to (2,9)
	for y := 3; y <= 9; y++ {
		set(2, y, 0.9)
	}

	// Use options to keep contour instead of min-area rectangle
	opts := PostProcessOptions{UseMinAreaRect: false}
	regs := PostProcessDBWithOptions(prob, w, h, 0.5, 0.3, opts)
	if len(regs) != 1 {
		t.Fatalf("expected 1 region, got %d", len(regs))
	}
	poly := regs[0].Polygon
	if len(poly) <= 4 {
		t.Fatalf("expected contour polygon with >4 points, got %d", len(poly))
	}
}

func TestPostProcessDB_MinRectPolygon_Exactly4Points(t *testing.T) {
	// Use the same L-shape as in the contour test
	w, h := 12, 12
	prob := make([]float32, w*h)
	set := func(x, y int, v float32) { prob[y*w+x] = v }
	for x := 2; x <= 9; x++ {
		set(x, 2, 0.9)
	}
	for y := 3; y <= 9; y++ {
		set(2, y, 0.9)
	}

	// Use options to force min-area rectangle output
	opts := PostProcessOptions{UseMinAreaRect: true}
	regs := PostProcessDBWithOptions(prob, w, h, 0.5, 0.3, opts)
	if len(regs) != 1 {
		t.Fatalf("expected 1 region, got %d", len(regs))
	}
	poly := regs[0].Polygon
	if len(poly) != 4 {
		t.Fatalf("expected min-rect polygon with 4 points, got %d", len(poly))
	}
}
