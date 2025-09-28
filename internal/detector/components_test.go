package detector

import (
	"testing"
)

// Helper function for float comparison.
func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func TestBinarize(t *testing.T) {
	prob := []float32{0.2, 0.6, 0.8, 0.3}
	w, h := 2, 2
	threshold := float32(0.5)

	mask := binarize(prob, w, h, threshold)

	expected := []bool{false, true, true, false}
	for i, v := range mask {
		if v != expected[i] {
			t.Errorf("expected mask[%d] = %v, got %v", i, expected[i], v)
		}
	}
}

func TestConnectedComponents_SimpleCase(t *testing.T) {
	// Create a simple 3x3 mask with one connected component
	mask := []bool{
		false, true, false,
		true, true, true,
		false, true, false,
	}
	prob := []float32{
		0.0, 0.8, 0.0,
		0.9, 0.7, 0.6,
		0.0, 0.5, 0.0,
	}
	w, h := 3, 3

	comps, labels := connectedComponents(mask, prob, w, h)

	// Should find exactly one component
	if len(comps) != 1 {
		t.Fatalf("expected 1 component, got %d", len(comps))
	}

	// Check component stats
	comp := comps[0]
	if comp.count != 5 { // 5 true pixels in the mask
		t.Errorf("expected component count 5, got %d", comp.count)
	}

	// Check bounding box
	if comp.minX != 0 || comp.maxX != 2 || comp.minY != 0 || comp.maxY != 2 {
		t.Errorf("unexpected bounding box: minX=%d, maxX=%d, minY=%d, maxY=%d",
			comp.minX, comp.maxX, comp.minY, comp.maxY)
	}

	// Check that all component pixels are labeled
	componentPixels := 0
	for _, label := range labels {
		if label == 1 { // first component gets label 1
			componentPixels++
		}
	}
	if componentPixels != 5 {
		t.Errorf("expected 5 labeled pixels, got %d", componentPixels)
	}
}

func TestConnectedComponents_MultipleComponents(t *testing.T) {
	// Create a mask with two separate components
	mask := []bool{
		true, false, true,
		false, false, false,
		true, false, true,
	}
	prob := []float32{
		0.8, 0.0, 0.7,
		0.0, 0.0, 0.0,
		0.6, 0.0, 0.9,
	}
	w, h := 3, 3

	comps, labels := connectedComponents(mask, prob, w, h)
	_ = labels // Mark as used to avoid compiler warning

	// Should find 4 separate components (each true pixel is isolated)
	if len(comps) != 4 {
		t.Fatalf("expected 4 components, got %d", len(comps))
	}

	// Each component should have count 1
	for i, comp := range comps {
		if comp.count != 1 {
			t.Errorf("component %d: expected count 1, got %d", i, comp.count)
		}
	}
}

func TestRegionsFromComponents_SimpleCase(t *testing.T) {
	// Create a simple component
	comps := []compStats{
		{
			count: 4,
			sum:   3.0, // average confidence will be 0.75
			minX:  1,
			minY:  1,
			maxX:  2,
			maxY:  2,
		},
	}

	// Create labels corresponding to a 2x2 block
	labels := []int{
		0, 0, 0, 0,
		0, 1, 1, 0,
		0, 1, 1, 0,
		0, 0, 0, 0,
	}
	w, h := 4, 4

	regions := regionsFromComponents(comps, labels, w, h, true)

	if len(regions) != 1 {
		t.Fatalf("expected 1 region, got %d", len(regions))
	}

	region := regions[0]
	expectedConf := 3.0 / 4.0 // 0.75
	if region.Confidence != expectedConf {
		t.Errorf("expected confidence %f, got %f", expectedConf, region.Confidence)
	}

	// Check that polygon is not empty
	if len(region.Polygon) == 0 {
		t.Error("expected non-empty polygon")
	}

	// Check that box is reasonable
	if region.Box.Width() <= 0 || region.Box.Height() <= 0 {
		t.Errorf("invalid bounding box: %+v", region.Box)
	}
}

func TestRegionsFromComponents_EmptyComponent(t *testing.T) {
	// Test with a component that has zero count
	comps := []compStats{
		{
			count: 0,
			sum:   0.0,
			minX:  0,
			minY:  0,
			maxX:  0,
			maxY:  0,
		},
	}

	labels := []int{0, 0, 0, 0}
	w, h := 2, 2

	regions := regionsFromComponents(comps, labels, w, h, true)
	_ = labels // Mark as used to avoid compiler warning

	// Should return empty regions since component has zero count
	if len(regions) != 0 {
		t.Fatalf("expected 0 regions for empty component, got %d", len(regions))
	}
}

func TestUpdateComponentStats(t *testing.T) {
	st := compStats{
		count: 0,
		sum:   0.0,
		minX:  10,
		minY:  10,
		maxX:  -1,
		maxY:  -1,
	}

	// Update with first pixel
	updateComponentStats(&st, 0.8, 2, 3)

	if st.count != 1 {
		t.Errorf("expected count 1, got %d", st.count)
	}
	if abs32(float32(st.sum-0.8)) > 1e-6 {
		t.Errorf("expected sum 0.8, got %f", st.sum)
	}
	if st.minX != 2 || st.maxX != 2 || st.minY != 3 || st.maxY != 3 {
		t.Errorf("unexpected bounds after first pixel: minX=%d, maxX=%d, minY=%d, maxY=%d",
			st.minX, st.maxX, st.minY, st.maxY)
	}

	// Update with second pixel
	updateComponentStats(&st, 0.6, 1, 4)

	if st.count != 2 {
		t.Errorf("expected count 2, got %d", st.count)
	}
	if abs32(float32(st.sum-1.4)) > 1e-6 {
		t.Errorf("expected sum 1.4, got %f", st.sum)
	}
	if st.minX != 1 || st.maxX != 2 || st.minY != 3 || st.maxY != 4 {
		t.Errorf("unexpected bounds after second pixel: minX=%d, maxX=%d, minY=%d, maxY=%d",
			st.minX, st.maxX, st.minY, st.maxY)
	}
}

func TestIsValidNeighbor(t *testing.T) {
	mask := []bool{
		true, false, true,
		false, true, false,
		true, false, true,
	}
	visited := []int{0, 0, 0, 0, 1, 0, 0, 0, 0} // center pixel already visited
	w, h := 3, 3

	// Test valid unvisited neighbor
	if !isValidNeighbor(mask, visited, w, h, 0, 0) {
		t.Error("expected (0,0) to be valid neighbor")
	}

	// Test invalid neighbor (already visited)
	if isValidNeighbor(mask, visited, w, h, 1, 1) {
		t.Error("expected (1,1) to be invalid (already visited)")
	}

	// Test invalid neighbor (false in mask)
	if isValidNeighbor(mask, visited, w, h, 1, 0) {
		t.Error("expected (1,0) to be invalid (false in mask)")
	}

	// Test out of bounds
	if isValidNeighbor(mask, visited, w, h, -1, 0) {
		t.Error("expected (-1,0) to be invalid (out of bounds)")
	}
	if isValidNeighbor(mask, visited, w, h, 3, 0) {
		t.Error("expected (3,0) to be invalid (out of bounds)")
	}
}
