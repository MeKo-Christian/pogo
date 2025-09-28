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

func TestApplyMorphologicalOperation_None(t *testing.T) {
	w, h := 5, 5
	probMap := make([]float32, w*h)
	for i := range probMap {
		probMap[i] = 0.5
	}

	config := MorphConfig{Operation: MorphNone, KernelSize: 3, Iterations: 1}
	result := ApplyMorphologicalOperation(probMap, w, h, config)

	// Should return original map unchanged
	for i, v := range result {
		if v != probMap[i] {
			t.Fatalf("expected unchanged map for MorphNone, but pixel %d changed from %f to %f", i, probMap[i], v)
		}
	}
}

func TestDilateFloat32(t *testing.T) {
	w, h := 5, 5
	probMap := make([]float32, w*h)

	// Set center pixel to high value
	probMap[2*w+2] = 0.9 // center at (2,2)

	result := dilateFloat32(probMap, w, h, 3)

	// Check that dilation expanded the high value to neighbors
	// With 3x3 kernel, all pixels around (2,2) should have value 0.9
	for y := 1; y <= 3; y++ {
		for x := 1; x <= 3; x++ {
			if result[y*w+x] != 0.9 {
				t.Errorf("expected dilated value 0.9 at (%d,%d), got %f", x, y, result[y*w+x])
			}
		}
	}

	// Corner pixels should still be 0
	if result[0] != 0.0 || result[4] != 0.0 || result[20] != 0.0 || result[24] != 0.0 {
		t.Error("corner pixels should remain 0 after dilation")
	}
}

func TestErodeFloat32(t *testing.T) {
	w, h := 5, 5
	probMap := make([]float32, w*h)

	// Fill a 3x3 block with high values
	for y := 1; y <= 3; y++ {
		for x := 1; x <= 3; x++ {
			probMap[y*w+x] = 0.9
		}
	}

	result := erodeFloat32(probMap, w, h, 3)

	// Only the center pixel should retain the high value after erosion
	if result[2*w+2] != 0.9 {
		t.Errorf("expected center pixel to remain 0.9 after erosion, got %f", result[2*w+2])
	}

	// Edge pixels of the original block should be eroded to 0
	if result[1*w+1] != 0.0 || result[1*w+3] != 0.0 || result[3*w+1] != 0.0 || result[3*w+3] != 0.0 {
		t.Error("edge pixels should be eroded to 0")
	}
}

func TestSmoothFloat32(t *testing.T) {
	w, h := 3, 3
	probMap := []float32{
		0.0, 1.0, 0.0,
		1.0, 1.0, 1.0,
		0.0, 1.0, 0.0,
	}

	result := smoothFloat32(probMap, w, h, 3)

	// Center pixel should be average of all 9 pixels: (5*1.0 + 4*0.0)/9 = 5/9 ≈ 0.556
	expectedCenter := float32(5.0 / 9.0)
	if abs(result[1*w+1]-expectedCenter) > 0.01 {
		t.Errorf("expected center pixel to be ~0.556 after smoothing, got %f", result[1*w+1])
	}

	// Corner pixels should be average of their neighborhoods
	// Top-left corner: (1*0.0 + 2*1.0 + 1*1.0)/4 = 3/4 = 0.75
	expectedCorner := float32(0.75)
	if abs(result[0]-expectedCorner) > 0.01 {
		t.Errorf("expected corner pixel to be 0.75 after smoothing, got %f", result[0])
	}
}

func TestApplyMorphologicalOperation_Opening(t *testing.T) {
	w, h := 7, 7
	probMap := make([]float32, w*h)

	// Create a pattern with small noise and a larger region
	probMap[1*w+1] = 0.9 // small isolated pixel (noise)

	// Larger connected region
	for y := 3; y <= 5; y++ {
		for x := 3; x <= 5; x++ {
			probMap[y*w+x] = 0.9
		}
	}

	config := MorphConfig{Operation: MorphOpening, KernelSize: 3, Iterations: 1}
	result := ApplyMorphologicalOperation(probMap, w, h, config)

	// Small isolated pixel should be removed by opening
	if result[1*w+1] > 0.1 {
		t.Errorf("expected small noise pixel to be removed by opening, got %f", result[1*w+1])
	}

	// Center of larger region should survive
	if result[4*w+4] < 0.8 {
		t.Errorf("expected center of large region to survive opening, got %f", result[4*w+4])
	}
}

func TestApplyMorphologicalOperation_Closing(t *testing.T) {
	w, h := 7, 7
	probMap := make([]float32, w*h)

	// Create two regions with a small gap
	for x := 1; x <= 2; x++ {
		probMap[3*w+x] = 0.9
	}
	// Gap at (3,3)
	for x := 4; x <= 5; x++ {
		probMap[3*w+x] = 0.9
	}

	config := MorphConfig{Operation: MorphClosing, KernelSize: 3, Iterations: 1}
	result := ApplyMorphologicalOperation(probMap, w, h, config)

	// Gap should be filled by closing
	if result[3*w+3] < 0.8 {
		t.Errorf("expected gap to be filled by closing, got %f", result[3*w+3])
	}
}

func TestApplyMorphologicalOperation_MultipleIterations(t *testing.T) {
	w, h := 5, 5
	probMap := make([]float32, w*h)

	// Set center pixel
	probMap[2*w+2] = 0.9

	config := MorphConfig{Operation: MorphDilate, KernelSize: 3, Iterations: 2}
	result := ApplyMorphologicalOperation(probMap, w, h, config)

	// After 2 iterations of dilation with 3x3 kernel,
	// the effect should spread further
	// First iteration: 3x3 area around center
	// Second iteration: 5x5 area around center

	// Check that all pixels should now have value 0.9
	for i, v := range result {
		if v != 0.9 {
			t.Errorf("expected all pixels to be 0.9 after 2 dilations, but pixel %d is %f", i, v)
		}
	}
}

func TestDefaultMorphConfig(t *testing.T) {
	config := DefaultMorphConfig()

	if config.Operation != MorphNone {
		t.Errorf("expected default operation to be MorphNone, got %v", config.Operation)
	}
	if config.KernelSize != 3 {
		t.Errorf("expected default kernel size to be 3, got %d", config.KernelSize)
	}
	if config.Iterations != 1 {
		t.Errorf("expected default iterations to be 1, got %d", config.Iterations)
	}
}

func TestApplyMorphologicalOperation_InvalidParams(t *testing.T) {
	w, h := 5, 5
	probMap := make([]float32, w*h)
	for i := range probMap {
		probMap[i] = 0.5
	}

	// Test zero kernel size
	config := MorphConfig{Operation: MorphDilate, KernelSize: 0, Iterations: 1}
	result := ApplyMorphologicalOperation(probMap, w, h, config)

	// Should return original unchanged
	for i, v := range result {
		if v != probMap[i] {
			t.Fatalf("expected unchanged map for zero kernel size")
		}
	}

	// Test zero iterations
	config = MorphConfig{Operation: MorphDilate, KernelSize: 3, Iterations: 0}
	result = ApplyMorphologicalOperation(probMap, w, h, config)

	// Should return original unchanged
	for i, v := range result {
		if v != probMap[i] {
			t.Fatalf("expected unchanged map for zero iterations")
		}
	}
}

// Helper function for float comparison
func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
