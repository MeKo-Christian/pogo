package rectify

import (
	"errors"
	"image"

	"github.com/MeKo-Tech/pogo/internal/onnx"
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

// Rectification input normalization constants.
//
// These are deliberately the identity parameters, i.e. plain [0,1] scaling,
// which is exactly what this package did before normalization became
// configurable. What the UVDoc model actually expects is not established in
// this repository, and guessing a mean/std here would silently change
// rectification output. The call site is parameterized so the values can be
// corrected once they are known.
const (
	// rectifyChannels is the number of input channels of the rectification model.
	rectifyChannels = 3
	// rectifyScale converts the raw 0..255 component to [0,1].
	rectifyScale = 1.0 / 255.0
)

// defaultNormalizeParams returns the normalization parameters used for
// rectification input (unchanged [0,1] scaling, see above).
func defaultNormalizeParams() utils.NormalizeParams {
	return utils.NormalizeParams{
		Channels: rectifyChannels,
		Scale:    rectifyScale,
		Mean:     [3]float32{0, 0, 0},
		Std:      [3]float32{1, 1, 1},
	}
}

// normalizeAndValidateImage normalizes the image and validates dimensions.
func (r *Rectifier) normalizeAndValidateImage(resized image.Image) ([]float32, int, int, error) {
	data, w, h, err := utils.NormalizeImageWith(resized, defaultNormalizeParams())
	if err != nil || w <= 0 || h <= 0 {
		return nil, 0, 0, err
	}
	return data, w, h, nil
}

// createInputTensor creates the input tensor for the model.
func (r *Rectifier) createInputTensor(data []float32, w, h int) (onnxrt.Value, error) {
	tensor, err := onnx.NewImageTensor(data, defaultNormalizeParams().Channels, h, w)
	if err != nil {
		return nil, err
	}
	return onnxrt.NewTensor(onnxrt.NewShape(tensor.Shape...), tensor.Data)
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
