package rectify

import (
	"errors"
	"image"

	"github.com/MeKo-Tech/pogo/internal/utils"
	onnxrt "github.com/yalue/onnxruntime_go"
)

// runModelInference runs the ONNX model and returns the output tensor data.
func (r *Rectifier) runModelInference(resized image.Image) ([]float32, int, int, error) {
	data, w, h, err := r.normalizeAndValidateImage(resized)
	if err != nil {
		return nil, 0, 0, err
	}

	input, err := r.createInputTensor(data, w, h)
	if err != nil {
		return nil, 0, 0, err
	}
	defer func() { _ = input.Destroy() }()

	output, err := r.runInference(input)
	if err != nil {
		return nil, 0, 0, err
	}
	defer func() { _ = output.Destroy() }()

	return r.extractOutputData(output)
}

// normalizeAndValidateImage normalizes the image and validates dimensions.
func (r *Rectifier) normalizeAndValidateImage(resized image.Image) ([]float32, int, int, error) {
	data, w, h, err := utils.NormalizeImage(resized)
	if err != nil || w <= 0 || h <= 0 {
		return nil, 0, 0, err
	}
	return data, w, h, nil
}

// createInputTensor creates the input tensor for the model.
func (r *Rectifier) createInputTensor(data []float32, w, h int) (onnxrt.Value, error) {
	return onnxrt.NewTensor(onnxrt.NewShape(1, 3, int64(h), int64(w)), data)
}

// runInference runs the model inference.
func (r *Rectifier) runInference(input onnxrt.Value) (onnxrt.Value, error) {
	outs := []onnxrt.Value{nil}
	if err := r.session.Run([]onnxrt.Value{input}, outs); err != nil {
		return nil, err
	}
	if len(outs) == 0 || outs[0] == nil {
		return nil, errors.New("no output from model")
	}
	return outs[0], nil
}

// extractOutputData extracts and validates the output tensor data.
func (r *Rectifier) extractOutputData(output onnxrt.Value) ([]float32, int, int, error) {
	t, ok := output.(*onnxrt.Tensor[float32])
	if !ok {
		return nil, 0, 0, errors.New("invalid output tensor type")
	}

	shape := t.GetShape()
	if len(shape) != 4 || shape[1] < 3 {
		return nil, 0, 0, errors.New("unexpected output shape")
	}

	oh, ow := int(shape[2]), int(shape[3])
	return t.GetData(), oh, ow, nil
}
