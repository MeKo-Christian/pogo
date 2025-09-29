package rectify

import (
	"github.com/MeKo-Tech/pogo/internal/models"
)

// RectificationMethod represents the type of rectification to use.
type RectificationMethod string

const (
	// RectificationUVDoc uses the UVDoc model for document rectification.
	RectificationUVDoc RectificationMethod = "uvdoc"
	// RectificationDocTR uses the DocTR model for document rectification.
	RectificationDocTR RectificationMethod = "doctr"
)

// Config holds configuration for the rectification process.
type Config struct {
	Enabled          bool                // whether rectification is enabled
	Method           RectificationMethod // rectification method to use (uvdoc or doctr)
	ModelPath        string              // path to the rectification ONNX model
	MaskThreshold    float64             // threshold for mask extraction (0-1)
	OutputHeight     int                 // target output height in pixels
	NumThreads       int                 // number of threads for ONNX inference (0 = auto)
	MinMaskCoverage  float64             // minimum mask coverage ratio (0-1)
	MinRectAreaRatio float64             // minimum rectangle area ratio (0-1)
	MinRectAspect    float64             // min acceptable aspect ratio (width/height)
	MaxRectAspect    float64             // max acceptable aspect ratio (width/height)
	// Debug dumping
	DebugDir string // if non-empty, writes mask and overlay PNGs here
}

// DefaultConfig returns sensible defaults for rectification.
func DefaultConfig() Config {
	return Config{
		Enabled:          false,
		Method:           RectificationUVDoc,
		ModelPath:        models.GetLayoutModelPath("", models.LayoutUVDoc),
		MaskThreshold:    0.5,
		OutputHeight:     1024,
		NumThreads:       0,
		MinMaskCoverage:  0.05,
		MinRectAreaRatio: 0.20,
		MinRectAspect:    0.20,
		MaxRectAspect:    8.0,
		DebugDir:         "",
	}
}

// UpdateModelPath relocates the model under the provided models directory.
func (c *Config) UpdateModelPath(modelsDir string) {
	var filename string
	switch c.Method {
	case RectificationDocTR:
		filename = models.LayoutDocTR
	default: // RectificationUVDoc or unknown
		filename = models.LayoutUVDoc
	}
	if filename == "." || filename == "" || filename == "/" {
		filename = models.LayoutUVDoc // fallback
	}
	c.ModelPath = models.GetLayoutModelPath(modelsDir, filename)
}
