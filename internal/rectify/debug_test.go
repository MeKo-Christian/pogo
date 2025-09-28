package rectify

import (
	"os"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/MeKo-Tech/pogo/internal/utils"
)

// TestClamp01 tests value clamping.
func TestClamp01(t *testing.T) {
	tests := []struct {
		input, want float64
	}{
		{-0.5, 0.0},
		{0.5, 0.5},
		{1.5, 1.0},
		{0.0, 0.0},
		{1.0, 1.0},
	}

	for _, tt := range tests {
		got := clamp01(tt.input)
		if got != tt.want {
			t.Errorf("clamp01(%f) = %f, want %f", tt.input, got, tt.want)
		}
	}
}

// TestDumpMaskPNG tests mask PNG dumping.
func TestDumpMaskPNG(t *testing.T) {
	tempDir := testutil.CreateTempDir(t)
	defer func() { _ = os.RemoveAll(tempDir) }()

	// Create test mask data
	w, h := 32, 32
	mask := make([]float32, w*h)
	for i := range mask {
		if i%2 == 0 {
			mask[i] = 0.8 // above threshold
		} else {
			mask[i] = 0.2 // below threshold
		}
	}

	err := dumpMaskPNG(tempDir, mask, w, h, 0.5)
	if err != nil {
		t.Errorf("dumpMaskPNG failed: %v", err)
	}

	// Check if file was created
	files, err := os.ReadDir(tempDir)
	if err != nil {
		t.Errorf("Failed to read temp dir: %v", err)
	}
	if len(files) == 0 {
		t.Error("Expected PNG file to be created")
	}
}

// TestDumpOverlayPNG tests overlay PNG dumping.
func TestDumpOverlayPNG(t *testing.T) {
	tempDir := testutil.CreateTempDir(t)
	defer func() { _ = os.RemoveAll(tempDir) }()

	src := makeTestImage(64, 64)
	quad := []utils.Point{
		{X: 10, Y: 10},
		{X: 50, Y: 10},
		{X: 50, Y: 50},
		{X: 10, Y: 50},
	}

	err := dumpOverlayPNG(tempDir, src, quad)
	if err != nil {
		t.Errorf("dumpOverlayPNG failed: %v", err)
	}

	// Check if file was created
	files, err := os.ReadDir(tempDir)
	if err != nil {
		t.Errorf("Failed to read temp dir: %v", err)
	}
	if len(files) == 0 {
		t.Error("Expected PNG file to be created")
	}
}

// TestDumpComparePNG tests comparison PNG dumping.
func TestDumpComparePNG(t *testing.T) {
	tempDir := testutil.CreateTempDir(t)
	defer func() { _ = os.RemoveAll(tempDir) }()

	src := makeTestImage(64, 64)
	dst := makeTestImage(32, 32)
	quad := []utils.Point{
		{X: 10, Y: 10},
		{X: 50, Y: 10},
		{X: 50, Y: 50},
		{X: 10, Y: 50},
	}

	err := dumpComparePNG(tempDir, src, quad, dst)
	if err != nil {
		t.Errorf("dumpComparePNG failed: %v", err)
	}

	// Check if file was created
	files, err := os.ReadDir(tempDir)
	if err != nil {
		t.Errorf("Failed to read temp dir: %v", err)
	}
	if len(files) == 0 {
		t.Error("Expected PNG file to be created")
	}
}
