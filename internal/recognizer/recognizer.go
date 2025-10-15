package recognizer

import (
	"errors"
	"fmt"
	"image"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/onnx"
	"github.com/MeKo-Tech/pogo/internal/orientation"
	onnxrt "github.com/yalue/onnxruntime_go"
)

// Config holds configuration for the text recognizer.
type Config struct {
	ModelPath      string   // Path to ONNX recognition model
	DictPath       string   // Path to character dictionary (single) - must match model output classes
	DictPaths      []string // Optional multiple dictionary paths to merge - must match model output classes
	FilterDictPath string   // Optional path to filter dictionary (restricts output to subset of characters)
	FilterDictPaths []string // Optional multiple filter dictionary paths to merge
	ImageHeight    int      // Expected input height (e.g., 32 or 48)
	UseServerModel bool     // Use server model instead of mobile
	NumThreads     int      // Number of CPU threads (0 for default)
	// Preprocessing parameters
	MaxWidth         int            // Optional max width clamp (0 = no clamp)
	PadWidthMultiple int            // If >0, right-pad width to this multiple
	Language         string         // Optional language for post-processing rules
	GPU              onnx.GPUConfig // GPU acceleration configuration
	// Decoding parameters
	DecodingMethod string // "greedy" or "beam_search"
	BeamWidth      int    // Beam width for beam search (ignored for greedy)
}

// DefaultConfig returns a default recognizer configuration.
func DefaultConfig() Config {
	return Config{
		ModelPath:        models.GetRecognitionModelPath("", false),
		DictPath:         models.GetDictionaryPath("", models.DictionaryPPOCRv5),
		ImageHeight:      48, // Use 48 for mobile models (PP-OCRv5_mobile_rec.onnx)
		UseServerModel:   false,
		NumThreads:       0,
		MaxWidth:         0,
		PadWidthMultiple: 8,
		Language:         "",
		GPU:              onnx.DefaultGPUConfig(),
		DecodingMethod:   "greedy",
		BeamWidth:        10,
	}
}

// UpdateModelPath updates the ModelPath and DictPath based on modelsDir and UseServerModel flag.
func (c *Config) UpdateModelPath(modelsDir string) {
    c.ModelPath = models.GetRecognitionModelPath(modelsDir, c.UseServerModel)
    // Update DictPath if using single dictionary (not multiple)
    if len(c.DictPaths) == 0 {
        // Default to PP-OCRv5 dictionary which matches PP-OCRv5 models
        c.DictPath = models.GetDictionaryPath(modelsDir, models.DictionaryPPOCRv5)
    }
}

// Recognizer performs text recognition using ONNX Runtime.
type Recognizer struct {
	config     Config
	session    *onnxrt.DynamicAdvancedSession
	inputInfo  onnxrt.InputOutputInfo
	outputInfo onnxrt.InputOutputInfo
	charset    *Charset        // Model dictionary - must match ONNX model output classes
	filterCharset *Charset     // Optional filter dictionary - restricts output characters
	mu         sync.RWMutex
	// Optional per-text-line orientation classifier (0/90/180/270)
	textLineOrienter *orientation.Classifier
}

// NewRecognizer creates a new text recognizer with the given configuration.
func NewRecognizer(config Config) (*Recognizer, error) {
	if err := validateRecognizerConfig(config); err != nil {
		return nil, err
	}

	if err := initializeONNXRuntimeForRecognizer(); err != nil {
		return nil, err
	}

	inputInfo, outputInfo, err := getModelInfoForRecognizer(config.ModelPath)
	if err != nil {
		return nil, err
	}

	if len(inputInfo.Dimensions) != 4 {
		return nil, fmt.Errorf("expected 4D input tensor, got %dD", len(inputInfo.Dimensions))
	}

	// Auto-adjust recognition height if model specifies a fixed height and config left it zero.
	// Input is [N, C, H, W]. If H>0 and ImageHeight<=0, adopt model's H.
	if h := inputInfo.Dimensions[2]; h > 0 && config.ImageHeight <= 0 {
		config.ImageHeight = int(h)
	}

	charset, err := loadCharsetForRecognizer(config)
	if err != nil {
		return nil, err
	}

	// Load optional filter charset
	filterCharset, err := loadFilterCharsetForRecognizer(config)
	if err != nil {
		return nil, err
	}

	session, err := createONNXSessionForRecognizer(config, inputInfo, outputInfo)
	if err != nil {
		return nil, err
	}

	r := &Recognizer{
		config:        config,
		session:       session,
		inputInfo:     inputInfo,
		outputInfo:    outputInfo,
		charset:       charset,
		filterCharset: filterCharset,
	}
	return r, nil
}

func validateRecognizerConfig(config Config) error {
	if config.ModelPath == "" {
		return errors.New("model path cannot be empty")
	}
	if config.DictPath == "" && len(config.DictPaths) == 0 {
		return errors.New("dictionary path cannot be empty")
	}

	if _, err := os.Stat(config.ModelPath); os.IsNotExist(err) {
		return fmt.Errorf("model file not found: %s", config.ModelPath)
	}

	if len(config.DictPaths) > 0 {
		for _, p := range config.DictPaths {
			if _, err := os.Stat(p); os.IsNotExist(err) {
				return fmt.Errorf("dictionary file not found: %s", p)
			}
		}
	} else {
		if _, err := os.Stat(config.DictPath); os.IsNotExist(err) {
			return fmt.Errorf("dictionary file not found: %s", config.DictPath)
		}
	}
	return nil
}

func initializeONNXRuntimeForRecognizer() error {
	if err := setONNXLibraryPath(); err != nil {
		return fmt.Errorf("failed to set ONNX Runtime library path: %w", err)
	}

	if !onnxrt.IsInitialized() {
		if err := onnxrt.InitializeEnvironment(); err != nil {
			return fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
		}
	}
	return nil
}

func getModelInfoForRecognizer(modelPath string) (onnxrt.InputOutputInfo, onnxrt.InputOutputInfo, error) {
	inputs, outputs, err := onnxrt.GetInputOutputInfo(modelPath)
	if err != nil {
		return onnxrt.InputOutputInfo{}, onnxrt.InputOutputInfo{},
			fmt.Errorf("failed to get model input/output info: %w", err)
	}
	if len(inputs) != 1 {
		return onnxrt.InputOutputInfo{}, onnxrt.InputOutputInfo{}, fmt.Errorf("expected 1 input, got %d", len(inputs))
	}
	if len(outputs) != 1 {
		return onnxrt.InputOutputInfo{}, onnxrt.InputOutputInfo{}, fmt.Errorf("expected 1 output, got %d", len(outputs))
	}
	return inputs[0], outputs[0], nil
}

func loadCharsetForRecognizer(config Config) (*Charset, error) {
	var charset *Charset
	var err error

	if len(config.DictPaths) > 0 {
		slog.Debug("Loading merged dictionaries", "count", len(config.DictPaths), "paths", config.DictPaths)
		charset, err = LoadCharsets(config.DictPaths)
	} else {
		slog.Debug("Loading single dictionary", "path", config.DictPath)
		charset, err = LoadCharset(config.DictPath)
	}

	if err != nil {
		return nil, err
	}

	slog.Debug("Dictionary loaded successfully", "charset_size", charset.Size())
	return charset, nil
}

func loadFilterCharsetForRecognizer(config Config) (*Charset, error) {
	// Filter charset is optional - if not configured, return nil (no filtering)
	if config.FilterDictPath == "" && len(config.FilterDictPaths) == 0 {
		return nil, nil
	}

	var filterCharset *Charset
	var err error

	if len(config.FilterDictPaths) > 0 {
		slog.Debug("Loading merged filter dictionaries", "count", len(config.FilterDictPaths), "paths", config.FilterDictPaths)
		filterCharset, err = LoadCharsets(config.FilterDictPaths)
	} else {
		slog.Debug("Loading single filter dictionary", "path", config.FilterDictPath)
		filterCharset, err = LoadCharset(config.FilterDictPath)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to load filter dictionary: %w", err)
	}

	slog.Debug("Filter dictionary loaded successfully", "filter_charset_size", filterCharset.Size())
	return filterCharset, nil
}

func createONNXSessionForRecognizer(
	config Config,
	inputInfo, outputInfo onnxrt.InputOutputInfo,
) (*onnxrt.DynamicAdvancedSession, error) {
	sessionOptions, err := onnxrt.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer func() {
		if err := sessionOptions.Destroy(); err != nil {
			fmt.Fprintf(os.Stderr, "Error destroying session options: %v\n", err)
		}
	}()

	if err := configureSessionOptionsForRecognizer(sessionOptions, config); err != nil {
		return nil, err
	}

	session, err := onnxrt.NewDynamicAdvancedSession(
		config.ModelPath,
		[]string{inputInfo.Name},
		[]string{outputInfo.Name},
		sessionOptions,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return session, nil
}

func configureSessionOptionsForRecognizer(sessionOptions *onnxrt.SessionOptions, config Config) error {
	if err := onnx.ConfigureSessionForGPU(sessionOptions, config.GPU); err != nil {
		return fmt.Errorf("failed to configure GPU: %w", err)
	}

	if config.NumThreads > 0 {
		if err := sessionOptions.SetIntraOpNumThreads(config.NumThreads); err != nil {
			return fmt.Errorf("failed to set thread count: %w", err)
		}
	}

	return nil
}

// Close releases resources used by the recognizer.
func (r *Recognizer) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.session != nil {
		if err := r.session.Destroy(); err != nil {
			fmt.Fprintf(os.Stderr, "Error destroying session: %v\n", err)
		}
		r.session = nil
	}
	return nil
}

// GetConfig returns a copy of the recognizer's configuration.
func (r *Recognizer) GetConfig() Config {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.config
}

// GetInputShape returns the model input shape.
func (r *Recognizer) GetInputShape() []int64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	shape := make([]int64, len(r.inputInfo.Dimensions))
	copy(shape, r.inputInfo.Dimensions)
	return shape
}

// GetOutputShape returns the model output shape.
func (r *Recognizer) GetOutputShape() []int64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	shape := make([]int64, len(r.outputInfo.Dimensions))
	copy(shape, r.outputInfo.Dimensions)
	return shape
}

// GetCharset returns the loaded character set.
func (r *Recognizer) GetCharset() *Charset {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.charset
}

// GetModelInfo returns information about the loaded recognition model.
func (r *Recognizer) GetModelInfo() map[string]interface{} {
	r.mu.RLock()
	defer r.mu.RUnlock()
	info := map[string]interface{}{
		"model_path":       r.config.ModelPath,
		"dict_path":        r.config.DictPath,
		"input_name":       r.inputInfo.Name,
		"output_name":      r.outputInfo.Name,
		"input_shape":      r.inputInfo.Dimensions,
		"output_shape":     r.outputInfo.Dimensions,
		"input_data_type":  r.inputInfo.DataType,
		"output_data_type": r.outputInfo.DataType,
		"image_height":     r.config.ImageHeight,
		"use_server_model": r.config.UseServerModel,
		"num_threads":      r.config.NumThreads,
		"charset_size":     r.charset.Size(),
		"language":         r.config.Language,
		"decoding_method":  r.config.DecodingMethod,
		"beam_width":       r.config.BeamWidth,
		"gpu": map[string]interface{}{
			"enabled":                r.config.GPU.UseGPU,
			"device_id":              r.config.GPU.DeviceID,
			"memory_limit_bytes":     r.config.GPU.GPUMemLimit,
			"arena_extend_strategy":  r.config.GPU.ArenaExtendStrategy,
			"cudnn_conv_algo_search": r.config.GPU.CUDNNConvAlgoSearch,
			"copy_in_default_stream": r.config.GPU.DoCopyInDefaultStream,
		},
	}
	return info
}

// SetTextLineOrienter assigns an optional per-region orientation classifier.
func (r *Recognizer) SetTextLineOrienter(cls *orientation.Classifier) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.textLineOrienter = cls
	slog.Debug("Text-line orientation classifier assigned to recognizer")
}

// Warmup runs a number of forward passes on a blank synthetic image to reduce cold-start latency.
func (r *Recognizer) Warmup(iterations int) error {
	if iterations <= 0 {
		return nil
	}

	r.mu.RLock()
	sess := r.session
	in := r.inputInfo
	cfg := r.config
	r.mu.RUnlock()

	if sess == nil {
		return errors.New("recognizer session is nil")
	}

	// Prepare warmup data
	warmupData, err := r.prepareWarmupData(cfg, in)
	if err != nil {
		return err
	}

	// Run warmup iterations
	return r.runWarmupIterations(sess, warmupData, iterations)
}

func (r *Recognizer) prepareWarmupData(cfg Config, in onnxrt.InputOutputInfo) (*onnx.Tensor, error) {
	// Determine target H from config or model
	h := cfg.ImageHeight
	if h <= 0 && len(in.Dimensions) == 4 && in.Dimensions[2] > 0 {
		h = int(in.Dimensions[2])
	}
	if h <= 0 {
		h = 32
	}

	// Choose a modest width, pad as needed
	w := h * 4
	img := image.NewRGBA(image.Rect(0, 0, w, h))

	// Resize/pad (no-op for correct size)
	resized, _, _, err := ResizeForRecognition(img, h, 0, 8)
	if err != nil {
		return nil, err
	}

	// Normalize
	ten, err := NormalizeForRecognition(resized)
	if err != nil {
		return nil, err
	}

	return &ten, nil
}

func (r *Recognizer) runWarmupIterations(
	sess *onnxrt.DynamicAdvancedSession,
	tensor *onnx.Tensor,
	iterations int,
) error {
	for range iterations {
		if err := r.runSingleWarmupIteration(sess, tensor); err != nil {
			return err
		}
	}
	return nil
}

func (r *Recognizer) runSingleWarmupIteration(sess *onnxrt.DynamicAdvancedSession, tensor *onnx.Tensor) error {
	inputTensor, err := onnxrt.NewTensor(onnxrt.NewShape(tensor.Shape...), tensor.Data)
	if err != nil {
		return err
	}
	defer func() { _ = inputTensor.Destroy() }()

	outputs := []onnxrt.Value{nil}
	runErr := sess.Run([]onnxrt.Value{inputTensor}, outputs)
	if runErr != nil {
		return runErr
	}

	// Clean up outputs
	for _, o := range outputs {
		if o != nil {
			_ = o.Destroy()
		}
	}

	return nil
}

// setONNXLibraryPath sets the onnxruntime shared library path from common locations.
func setONNXLibraryPath() error {
	// Try system paths first
	if path := findSystemLibraryPath(); path != "" {
		onnxrt.SetSharedLibraryPath(path)
		return nil
	}

	// Try project-relative path
	root, err := findProjectRoot()
	if err != nil {
		return err
	}

	libPath, err := getProjectLibraryPath(root)
	if err != nil {
		return err
	}

	onnxrt.SetSharedLibraryPath(libPath)
	return nil
}

// findSystemLibraryPath checks common system locations for the ONNX Runtime library.
func findSystemLibraryPath() string {
	systemPaths := []string{
		"/usr/local/lib/libonnxruntime.so",
		"/usr/lib/libonnxruntime.so",
		"/opt/onnxruntime/cpu/lib/libonnxruntime.so",
	}
	for _, p := range systemPaths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

// findProjectRoot finds the project root by looking for go.mod.
func findProjectRoot() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get current directory: %w", err)
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

// getProjectLibraryPath constructs the project-relative library path.
func getProjectLibraryPath(root string) (string, error) {
	libName, err := getLibraryName()
	if err != nil {
		return "", err
	}
	libPath := filepath.Join(root, "onnxruntime", "lib", libName)
	if _, err := os.Stat(libPath); err != nil {
		return "", fmt.Errorf("ONNX Runtime library not found at %s", libPath)
	}
	return libPath, nil
}

// getLibraryName returns the appropriate library name for the current OS.
func getLibraryName() (string, error) {
	switch runtime.GOOS {
	case "linux":
		return "libonnxruntime.so", nil
	case "darwin":
		return "libonnxruntime.dylib", nil
	case "windows":
		return "onnxruntime.dll", nil
	default:
		return "", fmt.Errorf("unsupported operating system: %s", runtime.GOOS)
	}
}
