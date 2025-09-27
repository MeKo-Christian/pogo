package mock

import (
	"math"
)

// ImageMap represents a synthetic ONNX output for detection with NCHW shape [1,1,H,W].
type ImageMap struct {
	Data   []float32
	Width  int
	Height int
}

// NewUniformMap creates a uniform probability map of size WxH with the given value in [0,1].
func NewUniformMap(w, h int, value float32) ImageMap {
	if w <= 0 || h <= 0 {
		return ImageMap{Data: nil, Width: 0, Height: 0}
	}
	size := w * h
	data := make([]float32, size)
	for i := range data {
		data[i] = clamp01(value)
	}
	return ImageMap{Data: data, Width: w, Height: h}
}

// NewCenteredBlobMap creates a Gaussian-like blob centered in the map.
// sigma controls spread; higher values = wider blob.
func NewCenteredBlobMap(w, h int, peak float32, sigma float64) ImageMap {
	if w <= 0 || h <= 0 {
		return ImageMap{Data: nil, Width: 0, Height: 0}
	}
	data := make([]float32, w*h)
	cx := float64(w-1) / 2.0
	cy := float64(h-1) / 2.0
	inv2s2 := 1.0 / (2.0 * sigma * sigma)
	for y := range h {
		for x := range w {
			dx := float64(x) - cx
			dy := float64(y) - cy
			v := float32(math.Exp(-(dx*dx+dy*dy)*inv2s2)) * peak
			data[y*w+x] = clamp01(v)
		}
	}
	return ImageMap{Data: data, Width: w, Height: h}
}

// NewTextStripeMap creates horizontal stripes to mimic text lines in detection maps.
// lineHeight is the height of bright stripes; gap is distance between stripes.
func NewTextStripeMap(w, h int, lineHeight, gap int, hi, lo float32) ImageMap {
	if w <= 0 || h <= 0 || lineHeight <= 0 || gap < 0 {
		return ImageMap{Data: nil, Width: 0, Height: 0}
	}
	data := make([]float32, w*h)
	period := lineHeight + gap
	for y := range h {
		v := lo
		if (y % period) < lineHeight {
			v = hi
		}
		v = clamp01(v)
		off := y * w
		for x := range w {
			data[off+x] = v
		}
	}
	return ImageMap{Data: data, Width: w, Height: h}
}

// Logits represents synthetic recognition network output as a flat array with shape.
// Typical shapes seen are [N, T, C] or [N, C, T].
type Logits struct {
	Data  []float32
	Shape []int64
}

// NewGreedyPathLogits constructs logits for a single sequence (N=1) with
// time steps T and classes C such that greedy argmax yields the given indices.
// indices should be over [0..C-1]; use 0 for CTC blank if desired.
// If classesFirst is true, shape is [1, C, T], otherwise [1, T, C].
func NewGreedyPathLogits(indices []int, classes int, classesFirst bool, high, low float32) Logits {
	if classes <= 0 || len(indices) == 0 {
		return Logits{Data: nil, Shape: []int64{}}
	}
	t := len(indices)
	if classesFirst {
		// [1, C, T]
		shape := []int64{1, int64(classes), int64(t)}
		data := make([]float32, 1*classes*t)
		for ti, c := range indices {
			for cls := range classes {
				v := low
				if cls == c {
					v = high
				}
				data[0*classes*t+cls*t+ti] = v
			}
		}
		return Logits{Data: data, Shape: shape}
	}
	// [1, T, C]
	shape := []int64{1, int64(t), int64(classes)}
	data := make([]float32, 1*t*classes)
	for ti, c := range indices {
		for cls := range classes {
			v := low
			if cls == c {
				v = high
			}
			data[0*t*classes+ti*classes+cls] = v
		}
	}
	return Logits{Data: data, Shape: shape}
}

func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
