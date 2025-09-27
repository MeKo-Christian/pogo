package onnx

import "testing"

func TestNewImageTensorAndVerify(t *testing.T) {
	c, h, w := 3, 4, 5
	data := make([]float32, c*h*w)
	ten, err := NewImageTensor(data, c, h, w)
	if err != nil {
		t.Fatalf("NewImageTensor error: %v", err)
	}
	if err := ValidateNCHW(ten.Shape); err != nil {
		t.Fatalf("ValidateNCHW: %v", err)
	}
	if err := VerifyImageTensor(ten); err != nil {
		t.Fatalf("VerifyImageTensor: %v", err)
	}
}

func TestNewBatchImageTensor(t *testing.T) {
	c, h, w := 3, 2, 2
	per := c * h * w
	img1 := make([]float32, per)
	img2 := make([]float32, per)
	ten, err := NewBatchImageTensor([][]float32{img1, img2}, c, h, w)
	if err != nil {
		t.Fatalf("NewBatchImageTensor error: %v", err)
	}
	if len(ten.Shape) != 4 || ten.Shape[0] != 2 || ten.Shape[1] != int64(c) || ten.Shape[2] != int64(h) || ten.Shape[3] != int64(w) {
		t.Fatalf("unexpected shape: %v", ten.Shape)
	}
	if err := VerifyImageTensor(ten); err != nil {
		t.Fatalf("VerifyImageTensor: %v", err)
	}
}

func TestTensorStats(t *testing.T) {
	data := []float32{0, 0.5, 1.0}
	minVal, maxVal, mean := TensorStats(data)
	if minVal != 0 || maxVal != 1.0 || mean < 0.49 || mean > 0.51 {
		t.Fatalf("unexpected stats: min=%v max=%v mean=%v", minVal, maxVal, mean)
	}
}
