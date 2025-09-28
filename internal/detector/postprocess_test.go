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

func TestAdaptiveNonMaxSuppression(t *testing.T) {
	regs := []DetectedRegion{
		{Box: utils.NewBox(0, 0, 10, 10), Confidence: 0.9},
		{Box: utils.NewBox(1, 1, 9, 9), Confidence: 0.8}, // heavy overlap with #1
		{Box: utils.NewBox(20, 20, 30, 30), Confidence: 0.7},
		{Box: utils.NewBox(21, 21, 29, 29), Confidence: 0.6}, // heavy overlap with #3
	}
	// Adaptive NMS should consider region characteristics
	kept := AdaptiveNonMaxSuppression(regs, 0.3, 1.0)
	if len(kept) < 2 {
		t.Fatalf("expected at least 2 kept regions after Adaptive NMS, got %d", len(kept))
	}
	// Check that results are sorted by confidence
	for i := 1; i < len(kept); i++ {
		if kept[i].Confidence > kept[i-1].Confidence {
			t.Fatalf("regions not sorted by confidence descending")
		}
	}
}

func TestSizeAwareNonMaxSuppression(t *testing.T) {
	regs := []DetectedRegion{
		{Box: utils.NewBox(0, 0, 5, 5), Confidence: 0.9},     // small region
		{Box: utils.NewBox(1, 1, 4, 4), Confidence: 0.8},     // small overlapping region
		{Box: utils.NewBox(20, 20, 50, 50), Confidence: 0.7}, // large region
		{Box: utils.NewBox(21, 21, 49, 49), Confidence: 0.6}, // large overlapping region
	}
	// Size-aware NMS should be stricter for small regions, more lenient for large ones
	kept := SizeAwareNonMaxSuppression(regs, 0.3, 0.1, 10, 1000)
	if len(kept) < 2 {
		t.Fatalf("expected at least 2 kept regions after Size-Aware NMS, got %d", len(kept))
	}
}

func TestCalculateAdaptiveIoUThreshold(t *testing.T) {
	regionA := DetectedRegion{
		Box:        utils.NewBox(0, 0, 10, 10),
		Confidence: 0.9,
	}
	regionB := DetectedRegion{
		Box:        utils.NewBox(5, 5, 15, 15),
		Confidence: 0.8,
	}

	threshold := calculateAdaptiveIoUThreshold(0.3, 1.0, regionA, regionB)
	if threshold < 0.1 || threshold > 0.8 {
		t.Fatalf("adaptive threshold out of bounds: %f", threshold)
	}

	// Test with different scale factors
	threshold2 := calculateAdaptiveIoUThreshold(0.3, 1.5, regionA, regionB)
	if threshold2 <= threshold {
		t.Fatalf("higher scale factor should increase threshold")
	}
}

func TestCalculateSizeBasedIoUThreshold(t *testing.T) {
	regionSmallA := DetectedRegion{
		Box:        utils.NewBox(0, 0, 5, 5), // small region
		Confidence: 0.9,
	}
	regionSmallB := DetectedRegion{
		Box:        utils.NewBox(0, 0, 6, 6), // another small region
		Confidence: 0.8,
	}
	regionLargeA := DetectedRegion{
		Box:        utils.NewBox(0, 0, 50, 50), // large region
		Confidence: 0.9,
	}
	regionLargeB := DetectedRegion{
		Box:        utils.NewBox(0, 0, 51, 51), // another large region
		Confidence: 0.8,
	}

	// Small regions should get stricter thresholds (smaller than base)
	thresholdSmall := calculateSizeBasedIoUThreshold(0.3, 0.1, 10, 1000, regionSmallA, regionSmallB)
	if thresholdSmall >= 0.3 {
		t.Fatalf("small regions should get stricter threshold, got %f", thresholdSmall)
	}

	// Large regions should get more lenient thresholds (larger than base)
	thresholdLarge := calculateSizeBasedIoUThreshold(0.3, 0.1, 10, 1000, regionLargeA, regionLargeB)
	if thresholdLarge <= 0.3 {
		t.Fatalf("large regions should get more lenient threshold, got %f", thresholdLarge)
	}

	// Large threshold should be higher than small threshold
	if thresholdLarge <= thresholdSmall {
		t.Fatalf("large region threshold (%f) should be > small region threshold (%f)", thresholdLarge, thresholdSmall)
	}
}

func TestAdaptiveNMSEdgeCases(t *testing.T) {
	// Test with single region
	regs := []DetectedRegion{
		{Box: utils.NewBox(0, 0, 10, 10), Confidence: 0.9},
	}
	kept := AdaptiveNonMaxSuppression(regs, 0.3, 1.0)
	if len(kept) != 1 {
		t.Fatalf("expected 1 region for single input, got %d", len(kept))
	}

	// Test with empty regions
	kept2 := AdaptiveNonMaxSuppression([]DetectedRegion{}, 0.3, 1.0)
	if len(kept2) != 0 {
		t.Fatalf("expected 0 regions for empty input, got %d", len(kept2))
	}

	// Test size-aware NMS with edge case sizes
	regs2 := []DetectedRegion{
		{Box: utils.NewBox(0, 0, 10, 10), Confidence: 0.9},
		{Box: utils.NewBox(1, 1, 9, 9), Confidence: 0.8},
	}
	kept3 := SizeAwareNonMaxSuppression(regs2, 0.3, 0.1, 100, 100) // min=max
	if len(kept3) < 1 {
		t.Fatalf("expected at least 1 region, got %d", len(kept3))
	}
}

func TestAdaptiveNMSWithDifferentRegionSizes(t *testing.T) {
	regs := []DetectedRegion{
		{Box: utils.NewBox(0, 0, 3, 3), Confidence: 0.9},       // tiny region
		{Box: utils.NewBox(1, 1, 2, 2), Confidence: 0.8},       // tiny overlapping
		{Box: utils.NewBox(10, 10, 20, 20), Confidence: 0.7},   // medium region
		{Box: utils.NewBox(11, 11, 19, 19), Confidence: 0.6},   // medium overlapping
		{Box: utils.NewBox(50, 50, 100, 100), Confidence: 0.5}, // large region
		{Box: utils.NewBox(51, 51, 99, 99), Confidence: 0.4},   // large overlapping
	}

	// Test adaptive NMS
	keptAdaptive := AdaptiveNonMaxSuppression(regs, 0.3, 1.0)
	if len(keptAdaptive) < 3 {
		t.Fatalf("expected at least 3 regions after adaptive NMS, got %d", len(keptAdaptive))
	}

	// Test size-aware NMS
	keptSizeAware := SizeAwareNonMaxSuppression(regs, 0.3, 0.1, 5, 10000)
	if len(keptSizeAware) < 3 {
		t.Fatalf("expected at least 3 regions after size-aware NMS, got %d", len(keptSizeAware))
	}

	// Compare with standard NMS
	keptStandard := NonMaxSuppression(regs, 0.3)
	// Adaptive methods might keep more or fewer regions depending on characteristics
	if len(keptAdaptive) == 0 || len(keptSizeAware) == 0 || len(keptStandard) == 0 {
		t.Fatalf("all NMS methods should keep at least some regions")
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
