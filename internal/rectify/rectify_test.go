package rectify

import (
	"image"
	"image/color"
	"testing"
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
