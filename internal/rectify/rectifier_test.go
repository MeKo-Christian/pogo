package rectify

import (
	"image"
	"image/color"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/testutil"
)

// makeTestImage creates a simple RGB image.
func makeTestImage(w, h int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.Set(x, y, color.RGBA{R: 200, G: 200, B: 200, A: 255})
		}
	}
	return img
}

func TestRectifier_Disabled_NoOp(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Enabled = false
	r, err := New(cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer r.Close()
	base := makeTestImage(64, 64)
	out, err := r.Apply(base)
	if err != nil {
		t.Fatalf("apply error: %v", err)
	}
	if out == nil {
		t.Fatal("expected non-nil image")
	}
}

func TestRectifier_Enabled_ModelMissing(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Enabled = true
	cfg.ModelPath = "/non/existent/uvdoc.onnx"
	if _, err := New(cfg); err == nil {
		t.Fatal("expected error for missing model, got nil")
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Enabled {
		t.Error("Expected Enabled to be false by default")
	}
	if cfg.ModelPath == "" {
		t.Error("Expected ModelPath to be set")
	}
	if cfg.MaskThreshold <= 0 || cfg.MaskThreshold > 1 {
		t.Errorf("Expected MaskThreshold to be in (0,1], got %f", cfg.MaskThreshold)
	}
	if cfg.OutputHeight <= 0 {
		t.Errorf("Expected OutputHeight to be positive, got %d", cfg.OutputHeight)
	}
	if cfg.MinMaskCoverage <= 0 || cfg.MinMaskCoverage > 1 {
		t.Errorf("Expected MinMaskCoverage to be in (0,1], got %f", cfg.MinMaskCoverage)
	}
	if cfg.MinRectAreaRatio <= 0 || cfg.MinRectAreaRatio > 1 {
		t.Errorf("Expected MinRectAreaRatio to be in (0,1], got %f", cfg.MinRectAreaRatio)
	}
	if cfg.MinRectAspect <= 0 {
		t.Errorf("Expected MinRectAspect to be positive, got %f", cfg.MinRectAspect)
	}
	if cfg.MaxRectAspect <= cfg.MinRectAspect {
		t.Errorf("Expected MaxRectAspect > MinRectAspect, got %f <= %f", cfg.MaxRectAspect, cfg.MinRectAspect)
	}
}

func TestConfig_UpdateModelPath(t *testing.T) {
	cfg := DefaultConfig()
	originalPath := cfg.ModelPath

	// Test with models directory
	cfg.UpdateModelPath("/custom/models")
	if cfg.ModelPath == originalPath {
		t.Error("Expected ModelPath to change when models directory is provided")
	}
	if !strings.Contains(cfg.ModelPath, "/custom/models") {
		t.Errorf("Expected ModelPath to contain custom models dir, got %s", cfg.ModelPath)
	}

	// Test with empty models directory
	cfg = DefaultConfig()
	cfg.UpdateModelPath("")
	if cfg.ModelPath != originalPath {
		t.Error("Expected ModelPath to remain unchanged with empty models directory")
	}
}

func TestValidateModelFile(t *testing.T) {
	// Test with existing file
	tempDir := testutil.CreateTempDir(t)
	defer func() { _ = os.RemoveAll(tempDir) }()

	validFile := filepath.Join(tempDir, "test.onnx")
	if err := os.WriteFile(validFile, []byte("fake model"), 0o644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	if err := validateModelFile(validFile); err != nil {
		t.Errorf("Expected no error for valid file, got %v", err)
	}

	// Test with non-existent file
	if err := validateModelFile("/non/existent/file.onnx"); err == nil {
		t.Error("Expected error for non-existent file, got nil")
	}
}

func TestFindProjectRoot(t *testing.T) {
	root, err := findProjectRoot()
	if err != nil {
		t.Errorf("Expected to find project root, got error: %v", err)
	}
	if root == "" {
		t.Error("Expected non-empty project root")
	}

	// Check that go.mod exists in the root
	goModPath := filepath.Join(root, "go.mod")
	if _, err := os.Stat(goModPath); os.IsNotExist(err) {
		t.Errorf("Expected go.mod to exist in project root %s", root)
	}
}

func TestGetONNXLibraryName(t *testing.T) {
	tests := []struct {
		goos string
		want string
	}{
		{"linux", "libonnxruntime.so"},
		{"darwin", "libonnxruntime.dylib"},
		{"windows", "onnxruntime.dll"},
	}

	for _, tt := range tests {
		t.Run(tt.goos, func(t *testing.T) {
			// We can't easily test different GOOS values in the same test
			// but we can verify the function doesn't panic and returns a valid library name
			name, err := getONNXLibraryName()
			if err != nil {
				t.Errorf("getONNXLibraryName() error = %v", err)
			}
			if name == "" {
				t.Error("Expected non-empty library name")
			}
			// Verify it contains "onnxruntime"
			if !strings.Contains(name, "onnxruntime") {
				t.Errorf("Expected library name to contain 'onnxruntime', got %s", name)
			}
		})
	}
}

func TestFindSystemONNXLibrary(t *testing.T) {
	// This test just verifies the function doesn't panic
	// We don't know if ONNX is installed on the system
	path := findSystemONNXLibrary()
	// path can be empty if ONNX is not installed, which is fine
	if path != "" {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Errorf("findSystemONNXLibrary returned non-existent path: %s", path)
		}
	}
}

func TestFindProjectONNXLibrary(t *testing.T) {
	// Test when ONNX runtime is available in project
	path, err := findProjectONNXLibrary()
	if err != nil {
		// This is expected if ONNX runtime is not set up in the project
		t.Logf("findProjectONNXLibrary failed (expected if ONNX not set up): %v", err)
		return
	}

	if path == "" {
		t.Error("Expected non-empty path when no error")
	}

	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Errorf("findProjectONNXLibrary returned non-existent path: %s", path)
	}
}

func TestRectifier_Apply_NilImage(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Enabled = false
	r, err := New(cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer r.Close()

	_, err = r.Apply(nil)
	// When disabled, it returns the input image as-is (even if nil)
	// So no error is expected
	if err != nil {
		t.Errorf("Unexpected error for nil image when disabled: %v", err)
	}
}

func TestRectifier_Apply_Disabled(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Enabled = false
	r, err := New(cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer r.Close()

	base := makeTestImage(64, 64)
	out, err := r.Apply(base)
	if err != nil {
		t.Fatalf("apply error: %v", err)
	}
	if out != base {
		t.Error("Expected same image when disabled")
	}
}

func TestRectifier_Close(t *testing.T) {
	// Test closing nil rectifier
	var r *Rectifier
	r.Close() // Should not panic

	// Test closing rectifier with nil session
	r = &Rectifier{}
	r.Close() // Should not panic
}

func TestRectifier_Apply_EdgeCases(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Enabled = false
	r, err := New(cfg)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer r.Close()

	// Test various edge cases
	testCases := []struct {
		name string
		img  image.Image
	}{
		{"nil image", nil},
		{"empty image", image.NewRGBA(image.Rect(0, 0, 0, 0))},
		{"very small image", image.NewRGBA(image.Rect(0, 0, 1, 1))},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := r.Apply(tc.img)
			if tc.name == "nil image" && err != nil {
				t.Errorf("Unexpected error for %s: %v", tc.name, err)
			}
			if tc.name != "nil image" && tc.img != nil && result == nil {
				t.Errorf("Expected non-nil result for %s", tc.name)
			}
		})
	}
}
