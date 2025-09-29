package rectify

import (
	"errors"
	"fmt"
	"image"
	"os"

	"github.com/MeKo-Tech/pogo/internal/utils"
	onnxrt "github.com/yalue/onnxruntime_go"
)

// Rectifier runs the UVDoc model and prepares data for perspective correction.
// Minimal CPU-only version: runs the model and currently returns the original image.
type Rectifier struct {
	cfg        Config
	session    *onnxrt.DynamicAdvancedSession
	inputInfo  onnxrt.InputOutputInfo
	outputInfo onnxrt.InputOutputInfo
}

// New creates a rectifier. If disabled in config, returns a stub (no session).
func New(cfg Config) (*Rectifier, error) {
	r := &Rectifier{cfg: cfg}
	if !cfg.Enabled {
		return r, nil
	}

	if err := validateModelFile(cfg.ModelPath); err != nil {
		return nil, err
	}

	session, inputInfo, outputInfo, err := createONNXSession(cfg)
	if err != nil {
		return nil, err
	}

	r.session = session
	r.inputInfo = inputInfo
	r.outputInfo = outputInfo
	return r, nil
}

func validateModelFile(modelPath string) error {
	if _, err := os.Stat(modelPath); err != nil {
		return fmt.Errorf("rectify model not found: %s", modelPath)
	}
	return nil
}

// Close releases ONNX resources.
func (r *Rectifier) Close() {
	if r == nil || r.session == nil {
		return
	}
	_ = r.session.Destroy()
	r.session = nil
}

// Apply runs rectification. Supports both UVDoc and DocTR methods.
func (r *Rectifier) Apply(img image.Image) (image.Image, error) {
	if r == nil || !r.cfg.Enabled || r.session == nil {
		return img, nil
	}
	if img == nil {
		return nil, errors.New("nil image")
	}

	switch r.cfg.Method {
	case RectificationDocTR:
		return r.applyDocTR(img)
	case RectificationUVDoc:
		return r.applyUVDoc(img)
	default:
		return r.applyUVDoc(img) // fallback to UVDoc
	}
}

// applyUVDoc runs UVDoc rectification (mask-based approach).
func (r *Rectifier) applyUVDoc(img image.Image) (image.Image, error) {
	// Prepare input image
	resized, err := utils.ResizeImage(img, utils.DefaultImageConstraints())
	if err != nil {
		return img, err
	}

	// Run model inference
	outData, oh, ow, err := r.runModelInference(resized)
	if err != nil {
		return img, err
	}

	// Process mask and find rectangle
	rect, valid := r.processMaskAndFindRectangle(outData, oh, ow)
	if !valid {
		return img, nil
	}

	// Transform and warp
	return r.transformAndWarpImage(img, resized, rect)
}

// applyDocTR runs DocTR rectification (corner prediction approach).
func (r *Rectifier) applyDocTR(img image.Image) (image.Image, error) {
	// Prepare input image
	resized, err := utils.ResizeImage(img, utils.DefaultImageConstraints())
	if err != nil {
		return img, err
	}

	// Run model inference
	outData, oh, ow, err := r.runModelInference(resized)
	if err != nil {
		return img, err
	}

	// Process DocTR output and find document corners
	rect, valid := r.processDocTROutput(outData, oh, ow)
	if !valid {
		return img, nil
	}

	// Transform and warp
	return r.transformAndWarpImage(img, resized, rect)
}
