package onnx

import (
	"errors"
	"fmt"
)

// Tensor represents a simple float32 tensor prepared for ONNX input.
// Data layout is row-major, with NCHW for images.
type Tensor struct {
	Data  []float32
	Shape []int64 // e.g., [N, C, H, W]
}

// NewImageTensor builds a single-image tensor with shape [1, C, H, W].
// data must be length C*H*W in NCHW order.
func NewImageTensor(data []float32, c, h, w int) (Tensor, error) {
	if data == nil {
		return Tensor{}, errors.New("nil data")
	}
	expected := c * h * w
	if len(data) != expected {
		return Tensor{}, fmt.Errorf("unexpected data length: got %d, want %d", len(data), expected)
	}
	shape := []int64{1, int64(c), int64(h), int64(w)}
	return Tensor{Data: data, Shape: shape}, nil
}

// NewBatchImageTensor stacks multiple images into [N, C, H, W]. All images must
// share the same (C, H, W) and be in NCHW order.
func NewBatchImageTensor(images [][]float32, c, h, w int) (Tensor, error) {
	if len(images) == 0 {
		return Tensor{}, errors.New("empty batch")
	}
	per := c * h * w
	total := per * len(images)
	out := make([]float32, total)
	for i, d := range images {
		if len(d) != per {
			return Tensor{}, fmt.Errorf("image %d has length %d, want %d", i, len(d), per)
		}
		copy(out[i*per:(i+1)*per], d)
	}
	shape := []int64{int64(len(images)), int64(c), int64(h), int64(w)}
	return Tensor{Data: out, Shape: shape}, nil
}

// ValidateNCHW ensures a shape is [N, C, H, W] with positive dimensions.
func ValidateNCHW(shape []int64) error {
	if len(shape) != 4 {
		return fmt.Errorf("shape rank %d != 4", len(shape))
	}
	for i, v := range shape {
		if v <= 0 {
			return fmt.Errorf("dimension %d must be > 0, got %d", i, v)
		}
	}
	return nil
}

// TensorStats computes simple statistics for debug output.
func TensorStats(data []float32) (float32, float32, float32) {
	if len(data) == 0 {
		return 0, 0, 0
	}
	var minVal, maxVal, mean float32
	minVal, maxVal = data[0], data[0]
	var sum float64
	for _, v := range data {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
		sum += float64(v)
	}
	mean = float32(sum / float64(len(data)))
	return minVal, maxVal, mean
}

// VerifyImageTensor checks data length matches the provided NCHW shape.
func VerifyImageTensor(t Tensor) error {
	if err := ValidateNCHW(t.Shape); err != nil {
		return err
	}
	n, c, h, w := t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3]
	expected := int(n * c * h * w)
	if len(t.Data) != expected {
		return fmt.Errorf("tensor data length %d != expected %d for shape %v", len(t.Data), expected, t.Shape)
	}
	return nil
}
