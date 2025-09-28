package detector

import (
	"testing"
)

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