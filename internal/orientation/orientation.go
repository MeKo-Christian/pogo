package orientation

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"runtime"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/onnx"
	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/disintegration/imaging"
	onnxrt "github.com/yalue/onnxruntime_go"
)

// Config controls document orientation detection behavior.
type Config struct {
	Enabled             bool
	ModelPath           string
	ConfidenceThreshold float64
	NumThreads          int
	// If true, falls back to a simple heuristic when the ONNX model is unavailable
	// or fails to initialize (useful for tests without model/runtime).
	UseHeuristicFallback bool
	GPU                  onnx.GPUConfig // GPU acceleration configuration
}

// DefaultConfig provides sensible defaults.
func DefaultConfig() Config {
	return Config{
		Enabled:              false,
		ModelPath:            models.GetLayoutModelPath("", models.LayoutPPLCNetX10Doc),
		ConfidenceThreshold:  0.7,
		NumThreads:           0,
		UseHeuristicFallback: true,
		GPU:                  onnx.DefaultGPUConfig(),
	}
}

// DefaultTextLineConfig returns defaults for per-text-line orientation classification
// using a lighter textline orientation model by default.
func DefaultTextLineConfig() Config {
	c := DefaultConfig()
	c.ModelPath = models.GetLayoutModelPath("", models.LayoutPPLCNetX025Textline)
	c.ConfidenceThreshold = 0.6
	c.Enabled = false
	c.GPU = onnx.DefaultGPUConfig()
	return c
}

// UpdateModelPath updates the ModelPath based on the provided models directory.
func (c *Config) UpdateModelPath(modelsDir string) {
	// Preserve current filename, relocate under provided models dir.
	filename := filepath.Base(c.ModelPath)
	if filename == "." || filename == "" || filename == "/" {
		// fallback to doc model
		filename = models.LayoutPPLCNetX10Doc
	}
	c.ModelPath = models.GetLayoutModelPath(modelsDir, filename)
}

// Result represents the predicted orientation.
type Result struct {
	Angle      int     // one of {0, 90, 180, 270}
	Confidence float64 // model probability for chosen angle (0..1)
}

// Classifier detects document orientation using an ONNX model when available.
// If unavailable and UseHeuristicFallback is true, a simple heuristic is used.
type Classifier struct {
	cfg        Config
	session    *onnxrt.DynamicAdvancedSession
	inputInfo  onnxrt.InputOutputInfo
	outputInfo onnxrt.InputOutputInfo
	// expected input dims (H, W). If 0, auto from inputInfo.
	inH, inW  int
	heuristic bool
}

// NewClassifier attempts to create an ONNX-backed classifier. If the model is
// not available and UseHeuristicFallback is true, it creates a heuristic-only classifier.
func NewClassifier(cfg Config) (*Classifier, error) {
	if !cfg.Enabled {
		return &Classifier{cfg: cfg, heuristic: true}, nil
	}

	// Try to create an ONNX-backed classifier first
	if c, err := tryCreateONNXClassifier(cfg); err == nil {
		return c, nil
	} else if cfg.UseHeuristicFallback {
		return &Classifier{cfg: cfg, heuristic: true}, nil
	} else {
		return nil, fmt.Errorf("onnx init: %w", err)
	}
}

// tryCreateONNXClassifier encapsulates the ONNX initialization path to reduce nesting in NewClassifier.
func tryCreateONNXClassifier(cfg Config) (*Classifier, error) {
	if err := validateModelPath(cfg.ModelPath); err != nil {
		return nil, err
	}

	if err := initializeONNXEnvironment(); err != nil {
		return nil, err
	}

	inputs, outputs, err := getModelIOInfo(cfg.ModelPath)
	if err != nil {
		return nil, err
	}

	in, out, err := validateModelIO(inputs, outputs)
	if err != nil {
		return nil, err
	}

	opts, err := createSessionOptions(cfg)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err := opts.Destroy(); err != nil {
			fmt.Fprintf(os.Stderr, "Error destroying session options: %v\n", err)
		}
	}()

	sess, err := createONNXSession(cfg.ModelPath, in, out, opts)
	if err != nil {
		return nil, err
	}

	return buildClassifier(cfg, sess, in, out), nil
}

func validateModelPath(modelPath string) error {
	if modelPath == "" {
		return errors.New("empty model path")
	}
	if _, err := os.Stat(modelPath); err != nil {
		return err
	}
	return nil
}

func initializeONNXEnvironment() error {
	if err := setONNXLibraryPath(); err != nil {
		return fmt.Errorf("onnx lib path: %w", err)
	}
	if !onnxrt.IsInitialized() {
		if err := onnxrt.InitializeEnvironment(); err != nil {
			return fmt.Errorf("init onnx: %w", err)
		}
	}
	return nil
}

func getModelIOInfo(modelPath string) ([]onnxrt.InputOutputInfo, []onnxrt.InputOutputInfo, error) {
	inputs, outputs, err := onnxrt.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, nil, fmt.Errorf("io info: %w", err)
	}
	return inputs, outputs, nil
}

func validateModelIO(inputs, outputs []onnxrt.InputOutputInfo) (onnxrt.InputOutputInfo, onnxrt.InputOutputInfo, error) {
	if len(inputs) != 1 || len(outputs) != 1 {
		return onnxrt.InputOutputInfo{}, onnxrt.InputOutputInfo{},
			fmt.Errorf("unexpected io (in:%d out:%d)", len(inputs), len(outputs))
	}

	in := inputs[0]
	out := outputs[0]

	if len(in.Dimensions) != 4 {
		return onnxrt.InputOutputInfo{}, onnxrt.InputOutputInfo{},
			fmt.Errorf("expected 4D input, got %dD", len(in.Dimensions))
	}

	return in, out, nil
}

func createSessionOptions(cfg Config) (*onnxrt.SessionOptions, error) {
	opts, err := onnxrt.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("session opts: %w", err)
	}

	if err := onnx.ConfigureSessionForGPU(opts, cfg.GPU); err != nil {
		return nil, fmt.Errorf("failed to configure GPU: %w", err)
	}

	if cfg.NumThreads > 0 {
		_ = opts.SetIntraOpNumThreads(cfg.NumThreads)
	}

	return opts, nil
}

func createONNXSession(modelPath string, in, out onnxrt.InputOutputInfo,
	opts *onnxrt.SessionOptions) (*onnxrt.DynamicAdvancedSession, error) {
	sess, err := onnxrt.NewDynamicAdvancedSession(modelPath, []string{in.Name}, []string{out.Name}, opts)
	if err != nil {
		return nil, fmt.Errorf("session: %w", err)
	}
	return sess, nil
}

func buildClassifier(cfg Config, sess *onnxrt.DynamicAdvancedSession, in, out onnxrt.InputOutputInfo) *Classifier {
	c := &Classifier{cfg: cfg, session: sess, inputInfo: in, outputInfo: out}

	if len(in.Dimensions) == 4 {
		if h := in.Dimensions[2]; h > 0 {
			c.inH = int(h)
		}
		if w := in.Dimensions[3]; w > 0 {
			c.inW = int(w)
		}
	}

	return c
}

// Close releases resources.
func (c *Classifier) Close() {
	if c.session != nil {
		if err := c.session.Destroy(); err != nil {
			fmt.Fprintf(os.Stderr, "Error destroying session: %v\n", err)
		}
		c.session = nil
	}
}

// Predict returns the document orientation. If confidence is below threshold,
// returns angle 0 by default.
func (c *Classifier) Predict(img image.Image) (Result, error) {
	if img == nil {
		return Result{}, errors.New("nil image")
	}

	if c.heuristic || c.session == nil {
		return c.predictWithHeuristic(img)
	}

	return c.predictWithONNX(img)
}

func (c *Classifier) predictWithHeuristic(img image.Image) (Result, error) {
	ang, conf := heuristicOrientation(img)
	if conf < c.cfg.ConfidenceThreshold {
		return Result{Angle: 0, Confidence: conf}, nil
	}
	return Result{Angle: ang, Confidence: conf}, nil
}

func (c *Classifier) predictWithONNX(img image.Image) (Result, error) {
	inputTensor, cleanupInput, err := c.prepareInputTensor(img)
	if err != nil {
		return Result{}, err
	}
	defer cleanupInput()

	outputs, cleanupOutputs, err := c.runInference(inputTensor)
	if err != nil {
		return Result{}, err
	}
	defer cleanupOutputs()

	logits, err := c.extractLogits(outputs)
	if err != nil {
		return Result{}, err
	}

	angle, confidence := c.computeOrientationFromLogits(logits)
	if confidence < c.cfg.ConfidenceThreshold {
		return Result{Angle: 0, Confidence: confidence}, nil
	}

	return Result{Angle: angle, Confidence: confidence}, nil
}

func (c *Classifier) prepareInputTensor(img image.Image) (*onnxrt.Tensor[float32], func(), error) {
	inH, inW := c.inH, c.inW
	if inH <= 0 || inW <= 0 {
		inH, inW = 192, 192
	}

	resized := imaging.Resize(img, inW, inH, imaging.Lanczos)
	data, w, h, err := utils.NormalizeImage(resized)
	if err != nil {
		return nil, nil, err
	}

	tensor, err := onnx.NewImageTensor(data, 3, h, w)
	if err != nil {
		return nil, nil, err
	}

	if err := onnx.VerifyImageTensor(tensor); err != nil {
		return nil, nil, err
	}

	input, err := onnxrt.NewTensor(onnxrt.NewShape(tensor.Shape...), tensor.Data)
	if err != nil {
		return nil, nil, fmt.Errorf("tensor: %w", err)
	}

	cleanup := func() {
		if err := input.Destroy(); err != nil {
			fmt.Fprintf(os.Stderr, "Error destroying input tensor: %v\n", err)
		}
	}

	return input, cleanup, nil
}

func (c *Classifier) runInference(input *onnxrt.Tensor[float32]) ([]onnxrt.Value, func(), error) {
	outputs := []onnxrt.Value{nil}
	if err := c.session.Run([]onnxrt.Value{input}, outputs); err != nil {
		return nil, nil, fmt.Errorf("run: %w", err)
	}

	cleanup := func() {
		for _, o := range outputs {
			if o != nil {
				if err := o.Destroy(); err != nil {
					fmt.Fprintf(os.Stderr, "Error destroying output tensor: %v\n", err)
				}
			}
		}
	}

	return outputs, cleanup, nil
}

func (c *Classifier) extractLogits(outputs []onnxrt.Value) ([]float32, error) {
	t, ok := outputs[0].(*onnxrt.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected output type %T", outputs[0])
	}

	shape := t.GetShape()
	if len(shape) != 2 || shape[1] < 4 {
		return nil, fmt.Errorf("unexpected output shape %v", shape)
	}

	return t.GetData(), nil
}

func (c *Classifier) computeOrientationFromLogits(logits []float32) (int, float64) {
	probs := softmax(logits[:4]) // Only use first 4 logits for 4 orientations

	idx := argmax(probs)
	angle := []int{0, 90, 180, 270}[idx]
	confidence := probs[idx]

	return angle, confidence
}

func softmax(logits []float32) []float64 {
	if len(logits) == 0 {
		return nil
	}

	// Find max for numerical stability
	maxLogit := logits[0]
	for _, v := range logits[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}

	// Compute exp and sum
	var sum float64
	probs := make([]float64, len(logits))
	for i, v := range logits {
		exp := math.Exp(float64(v - maxLogit))
		probs[i] = exp
		sum += exp
	}

	// Normalize
	for i := range probs {
		probs[i] /= sum
	}

	return probs
}

func argmax(values []float64) int {
	if len(values) == 0 {
		return -1
	}

	maxIdx := 0
	maxVal := values[0]
	for i, v := range values[1:] {
		if v > maxVal {
			maxVal = v
			maxIdx = i + 1
		}
	}
	return maxIdx
}

// heuristicOrientation uses a simple gradient-projection heuristic to
// distinguish between 0/180 vs 90/270. It cannot tell 0 vs 180 or 90 vs 270 reliably,
// so it returns either 0 or 90 with a confidence score.
func heuristicOrientation(img image.Image) (int, float64) {
	if img == nil {
		return 0, 0
	}

	thumb := prepareThumbnail(img)
	if !isValidThumbnail(thumb) {
		return 0, 0
	}

	meanLuminance := calculateMeanLuminance(thumb)
	rowTransitions := countTransitionsInRows(thumb, meanLuminance)
	colTransitions := countTransitionsInColumns(thumb, meanLuminance)

	return determineOrientation(rowTransitions, colTransitions, thumb.Bounds())
}

func prepareThumbnail(img image.Image) image.Image {
	return imaging.Resize(img, 128, 128, imaging.Lanczos)
}

func isValidThumbnail(img image.Image) bool {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	return w > 1 && h > 1
}

func calculateMeanLuminance(img image.Image) float64 {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()

	var sum float64
	for y := b.Min.Y; y < b.Min.Y+h; y++ {
		for x := b.Min.X; x < b.Min.X+w; x++ {
			r, g, bb, _ := img.At(x, y).RGBA()
			sum += 0.299*float64(r>>8) + 0.587*float64(g>>8) + 0.114*float64(bb>>8)
		}
	}
	return sum / float64(w*h)
}

func countTransitionsInRows(img image.Image, meanLuminance float64) float64 {
	b := img.Bounds()
	var transitions float64

	for y := b.Min.Y; y < b.Min.Y+b.Dy(); y++ {
		var prev int
		for x := b.Min.X; x < b.Min.X+b.Dx(); x++ {
			lum := calculateLuminance(img.At(x, y))
			cur := luminanceToBinary(lum, meanLuminance)

			if x == b.Min.X {
				prev = cur
				continue
			}

			if cur != prev {
				transitions++
			}
			prev = cur
		}
	}
	return transitions
}

func countTransitionsInColumns(img image.Image, meanLuminance float64) float64 {
	b := img.Bounds()
	var transitions float64

	for x := b.Min.X; x < b.Min.X+b.Dx(); x++ {
		var prev int
		for y := b.Min.Y; y < b.Min.Y+b.Dy(); y++ {
			lum := calculateLuminance(img.At(x, y))
			cur := luminanceToBinary(lum, meanLuminance)

			if y == b.Min.Y {
				prev = cur
				continue
			}

			if cur != prev {
				transitions++
			}
			prev = cur
		}
	}
	return transitions
}

func calculateLuminance(c color.Color) float64 {
	r, g, bb, _ := c.RGBA()
	return 0.299*float64(r>>8) + 0.587*float64(g>>8) + 0.114*float64(bb>>8)
}

func luminanceToBinary(luminance, threshold float64) int {
	if luminance < threshold {
		return 1
	}
	return 0
}

func determineOrientation(rowTransitions, colTransitions float64, bounds image.Rectangle) (int, float64) {
	total := rowTransitions + colTransitions
	if total == 0 {
		return 0, 0
	}

	ar := calculateAspectRatio(bounds)

	if colTransitions >= rowTransitions {
		base := (colTransitions - rowTransitions) / total
		if ar > 1.2 {
			base = math.Min(1.0, base+0.15)
		}
		return 90, base
	}

	base := (rowTransitions - colTransitions) / total
	if ar < 0.8 {
		base = math.Min(1.0, base+0.1)
	}
	return 0, base
}

func calculateAspectRatio(bounds image.Rectangle) float64 {
	w, h := float64(bounds.Dx()), float64(bounds.Dy())
	return h / w
}

// findProjectRoot finds the project root directory by looking for go.mod.
func findProjectRoot() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	root := cwd
	for {
		if _, err := os.Stat(filepath.Join(root, "go.mod")); err == nil {
			return root, nil
		}
		parent := filepath.Dir(root)
		if parent == root {
			return "", errors.New("could not find project root")
		}
		root = parent
	}
}

// getONNXLibName returns the appropriate library filename for the current OS.
func getONNXLibName() (string, error) {
	switch runtime.GOOS {
	case "linux":
		return "libonnxruntime.so", nil
	case "darwin":
		return "libonnxruntime.dylib", nil
	case "windows":
		return "onnxruntime.dll", nil
	default:
		return "", fmt.Errorf("unsupported OS: %s", runtime.GOOS)
	}
}

// setONNXLibraryPath attempts to locate the ONNX Runtime shared library similar to detector.
func setONNXLibraryPath() error {
	// Common system paths
	system := []string{
		"/usr/local/lib/libonnxruntime.so",
		"/usr/lib/libonnxruntime.so",
		"/opt/onnxruntime/cpu/lib/libonnxruntime.so",
	}
	for _, p := range system {
		if _, err := os.Stat(p); err == nil {
			onnxrt.SetSharedLibraryPath(p)
			return nil
		}
	}

	// Project-relative
	root, err := findProjectRoot()
	if err != nil {
		return err
	}

	libName, err := getONNXLibName()
	if err != nil {
		return err
	}

	p := filepath.Join(root, "onnxruntime", "lib", libName)
	if _, err := os.Stat(p); err != nil {
		return fmt.Errorf("ONNX Runtime library not found at %s", p)
	}
	onnxrt.SetSharedLibraryPath(p)
	return nil
}
