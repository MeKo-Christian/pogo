package detector

import (
	"errors"
	"fmt"
	"image"
	"os"

	"github.com/MeKo-Tech/pogo/internal/onnx"
	"github.com/yalue/onnxruntime_go"
)

// getWarmupDimensions returns appropriate dimensions for warmup based on model input info.
func (d *Detector) getWarmupDimensions() (int, int) {
	d.mu.RLock()
	in := d.inputInfo
	d.mu.RUnlock()

	h, w := 320, 320
	if len(in.Dimensions) == 4 {
		if in.Dimensions[2] > 0 {
			h = int(in.Dimensions[2])
		}
		if in.Dimensions[3] > 0 {
			w = int(in.Dimensions[3])
		}
	}
	return h, w
}

// runWarmupIteration performs a single warmup inference iteration.
func (d *Detector) runWarmupIteration(sess *onnxruntime_go.DynamicAdvancedSession, tensor onnx.Tensor) error {
	inputTensor, err := onnxruntime_go.NewTensor(onnxruntime_go.NewShape(tensor.Shape...), tensor.Data)
	if err != nil {
		return err
	}
	defer func() {
		if err := inputTensor.Destroy(); err != nil {
			fmt.Fprintf(os.Stderr, "Error destroying input tensor: %v\n", err)
		}
	}()

	outputs := []onnxruntime_go.Value{nil}
	runErr := sess.Run([]onnxruntime_go.Value{inputTensor}, outputs)
	if runErr != nil {
		return runErr
	}

	for _, o := range outputs {
		if o != nil {
			if err := o.Destroy(); err != nil {
				fmt.Fprintf(os.Stderr, "Error destroying output tensor: %v\n", err)
			}
		}
	}
	return nil
}

// Warmup runs a number of forward passes with a blank image to reduce first-run latency.
func (d *Detector) Warmup(iterations int) error {
	if iterations <= 0 {
		return nil
	}

	d.mu.RLock()
	sess := d.session
	d.mu.RUnlock()

	if sess == nil {
		return errors.New("detector session is nil")
	}

	h, w := d.getWarmupDimensions()

	// Create a black image of WxH
	img := image.NewRGBA(image.Rect(0, 0, w, h))

	// Preprocess once
	tensor, err := d.preprocessImage(img)
	if err != nil {
		return err
	}

	for range iterations {
		if err := d.runWarmupIteration(sess, tensor); err != nil {
			return err
		}
	}
	return nil
}
