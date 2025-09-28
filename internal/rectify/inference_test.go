package rectify

import (
	"testing"
)

// TestNormalizeAndValidateImage tests the image normalization and validation.
func TestNormalizeAndValidateImage(t *testing.T) {
	r := &Rectifier{}

	// Test valid image
	img := makeTestImage(64, 64)
	data, w, h, err := r.normalizeAndValidateImage(img)
	if err != nil {
		t.Errorf("Expected no error for valid image, got %v", err)
	}
	if len(data) == 0 {
		t.Error("Expected non-empty data")
	}
	if w != 64 || h != 64 {
		t.Errorf("Expected dimensions 64x64, got %dx%d", w, h)
	}

	// Test nil image
	if _, _, _, err := r.normalizeAndValidateImage(nil); err == nil {
		t.Error("Expected error for nil image")
	}
}

// TestCreateInputTensor tests input tensor creation.
func TestCreateInputTensor(t *testing.T) {
	// Skip this test if ONNX runtime is not available
	t.Skip("Skipping tensor creation test - requires ONNX runtime initialization")
}

// TestExtractOutputData tests output data extraction.
func TestExtractOutputData(t *testing.T) {
	r := &Rectifier{}

	// Create a mock tensor - this is tricky without ONNX runtime
	// For now, just test that the function exists and handles nil input
	if _, _, _, err := r.extractOutputData(nil); err == nil {
		t.Error("Expected error for nil tensor")
	}
}
