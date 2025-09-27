package pipeline

import (
	"errors"
	"fmt"
	"os"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/orientation"
	"github.com/MeKo-Tech/pogo/internal/recognizer"
	"github.com/MeKo-Tech/pogo/internal/rectify"
)

// Config holds configuration for the OCR pipeline and its components.
type Config struct {
	ModelsDir           string
	EnableOrientation   bool // deprecated: use Orientation.Enabled
	Orientation         orientation.Config
	TextLineOrientation orientation.Config
	Rectification       rectify.Config
	Detector            detector.Config
	Recognizer          recognizer.Config
	WarmupIterations    int // optional warmup runs per model to reduce first-run latency

	// Parallel processing configuration
	Parallel ParallelConfig // Configuration for parallel processing
	Resource ResourceConfig // Configuration for resource management
}

// DefaultConfig returns a default pipeline config with component defaults.
func DefaultConfig() Config {
	return Config{
		ModelsDir:           models.GetModelsDir(""),
		EnableOrientation:   false,
		Orientation:         orientation.DefaultConfig(),
		TextLineOrientation: orientation.DefaultTextLineConfig(),
		Rectification:       rectify.DefaultConfig(),
		Detector:            detector.DefaultConfig(),
		Recognizer:          recognizer.DefaultConfig(),
		WarmupIterations:    0,
		Parallel:            DefaultParallelConfig(),
		Resource:            DefaultResourceConfig(),
	}
}

// Builder constructs a Pipeline with fluent configuration.
type Builder struct {
	cfg Config
}

// NewBuilder creates a new pipeline builder with defaults.
func NewBuilder() *Builder { return &Builder{cfg: DefaultConfig()} }

// WithModelsDir sets the models directory and updates component model paths.
func (b *Builder) WithModelsDir(dir string) *Builder {
	if dir != "" {
		b.cfg.ModelsDir = dir
	}
	b.cfg.Detector.UpdateModelPath(b.cfg.ModelsDir)
	b.cfg.Recognizer.UpdateModelPath(b.cfg.ModelsDir)
	b.cfg.Orientation.UpdateModelPath(b.cfg.ModelsDir)
	b.cfg.TextLineOrientation.UpdateModelPath(b.cfg.ModelsDir)
	b.cfg.Rectification.UpdateModelPath(b.cfg.ModelsDir)
	return b
}

// WithDetectorModelPath overrides the detector model path directly.
func (b *Builder) WithDetectorModelPath(path string) *Builder {
	if path != "" {
		b.cfg.Detector.ModelPath = path
	}
	return b
}

// WithRecognizerModelPath overrides the recognizer model path directly.
func (b *Builder) WithRecognizerModelPath(path string) *Builder {
	if path != "" {
		b.cfg.Recognizer.ModelPath = path
	}
	return b
}

// WithDictionaryPath overrides the dictionary path directly.
func (b *Builder) WithDictionaryPath(path string) *Builder {
	if path != "" {
		b.cfg.Recognizer.DictPath = path
	}
	return b
}

// WithDictionaryPaths overrides the dictionary paths with a merged list.
func (b *Builder) WithDictionaryPaths(paths []string) *Builder {
	// Clean empty entries
	cleaned := make([]string, 0, len(paths))
	for _, p := range paths {
		if p != "" {
			cleaned = append(cleaned, p)
		}
	}
	if len(cleaned) > 0 {
		b.cfg.Recognizer.DictPaths = cleaned
		// Clear single DictPath to avoid ambiguity
		b.cfg.Recognizer.DictPath = ""
	}
	return b
}

// WithServerModels toggles using server variants for both detector and recognizer.
func (b *Builder) WithServerModels(useServer bool) *Builder {
	b.cfg.Detector.UseServerModel = useServer
	b.cfg.Recognizer.UseServerModel = useServer
	b.cfg.Detector.UpdateModelPath(b.cfg.ModelsDir)
	b.cfg.Recognizer.UpdateModelPath(b.cfg.ModelsDir)
	return b
}

// WithLanguage sets recognition language for post-processing.
func (b *Builder) WithLanguage(lang string) *Builder {
	b.cfg.Recognizer.Language = lang
	return b
}

// WithDetectorThresholds sets DB thresholds.
func (b *Builder) WithDetectorThresholds(dbThresh, dbBoxThresh float32) *Builder {
	b.cfg.Detector.DbThresh = dbThresh
	b.cfg.Detector.DbBoxThresh = dbBoxThresh
	return b
}

// WithDetectorNMS configures non-max suppression.
func (b *Builder) WithDetectorNMS(enabled bool, iou float64) *Builder {
	b.cfg.Detector.UseNMS = enabled
	if iou > 0 {
		b.cfg.Detector.NMSThreshold = iou
	}
	// reset to hard NMS if previously configured to Soft-NMS
	if !enabled {
		b.cfg.Detector.NMSMethod = "hard"
	}
	return b
}

// WithDetectorSoftNMS enables Soft-NMS with the given method ("linear" or "gaussian"),
// IoU threshold, sigma (for gaussian), and score threshold for final filtering.
// It sets UseNMS=true. If method is empty or unknown, "linear" is used.
func (b *Builder) WithDetectorSoftNMS(method string, iou, sigma, scoreThresh float64) *Builder {
	b.cfg.Detector.UseNMS = true
	if method == "" {
		method = "linear"
	}
	b.cfg.Detector.NMSMethod = method
	if iou > 0 {
		b.cfg.Detector.NMSThreshold = iou
	}
	if sigma > 0 {
		b.cfg.Detector.SoftNMSSigma = sigma
	}
	if scoreThresh > 0 {
		b.cfg.Detector.SoftNMSThresh = scoreThresh
	}
	return b
}

// WithDetectorPolygonMode selects the detector polygon mode:
// "minrect" (default) or "contour".
func (b *Builder) WithDetectorPolygonMode(mode string) *Builder {
	if mode != "" {
		b.cfg.Detector.PolygonMode = mode
	}
	return b
}

// WithThreads sets intra-op thread counts for both components (if >0).
func (b *Builder) WithThreads(n int) *Builder {
	if n > 0 {
		b.cfg.Detector.NumThreads = n
		b.cfg.Recognizer.NumThreads = n
	}
	return b
}

// WithImageHeight sets target recognition image height.
func (b *Builder) WithImageHeight(h int) *Builder {
	if h > 0 {
		b.cfg.Recognizer.ImageHeight = h
	}
	return b
}

// WithRecognizeWidthPadding sets recognition width clamp and padding multiple.
func (b *Builder) WithRecognizeWidthPadding(maxWidth, multiple int) *Builder {
	if maxWidth >= 0 {
		b.cfg.Recognizer.MaxWidth = maxWidth
	}
	if multiple > 0 {
		b.cfg.Recognizer.PadWidthMultiple = multiple
	}
	return b
}

// WithOrientation enables/disables orientation (placeholder only in 5.1).
func (b *Builder) WithOrientation(enabled bool) *Builder {
	b.cfg.EnableOrientation = enabled
	b.cfg.Orientation.Enabled = enabled
	return b
}

// WithTextLineOrientation enables/disables per-region text line orientation.
func (b *Builder) WithTextLineOrientation(enabled bool) *Builder {
	b.cfg.TextLineOrientation.Enabled = enabled
	return b
}

// WithRectification enables/disables document rectification.
func (b *Builder) WithRectification(enabled bool) *Builder {
	b.cfg.Rectification.Enabled = enabled
	return b
}

// WithRectifyModelPath overrides the rectification model path.
func (b *Builder) WithRectifyModelPath(path string) *Builder {
	if path != "" {
		b.cfg.Rectification.ModelPath = path
	}
	return b
}

// WithRectifyMaskThreshold sets the mask threshold used to gate rectification.
func (b *Builder) WithRectifyMaskThreshold(th float64) *Builder {
	if th > 0 {
		b.cfg.Rectification.MaskThreshold = th
	}
	return b
}

// WithRectifyOutputHeight sets the advisory output height for rectification.
func (b *Builder) WithRectifyOutputHeight(h int) *Builder {
	if h > 0 {
		b.cfg.Rectification.OutputHeight = h
	}
	return b
}

// WithRectifyDebugDir enables debug dumps for rectification stage into dir.
func (b *Builder) WithRectifyDebugDir(dir string) *Builder {
	if dir != "" {
		b.cfg.Rectification.DebugDir = dir
	}
	return b
}

// WithTextLineOrientationThreshold sets the text line orientation classifier threshold.
func (b *Builder) WithTextLineOrientationThreshold(th float64) *Builder {
	if th > 0 {
		b.cfg.TextLineOrientation.ConfidenceThreshold = th
	}
	return b
}

// WithOrientationThreshold sets the orientation classifier confidence threshold.
func (b *Builder) WithOrientationThreshold(th float64) *Builder {
	if th > 0 {
		b.cfg.Orientation.ConfidenceThreshold = th
	}
	return b
}

// WithWarmupIterations sets model warmup runs to reduce cold-start latency.
func (b *Builder) WithWarmupIterations(n int) *Builder {
	if n >= 0 {
		b.cfg.WarmupIterations = n
	}
	return b
}

// WithParallelWorkers sets the number of parallel workers for batch processing.
func (b *Builder) WithParallelWorkers(workers int) *Builder {
	if workers > 0 {
		b.cfg.Parallel.MaxWorkers = workers
	}
	return b
}

// WithBatchSize sets the batch size for micro-batching in parallel processing.
func (b *Builder) WithBatchSize(size int) *Builder {
	if size >= 0 {
		b.cfg.Parallel.BatchSize = size
	}
	return b
}

// WithMemoryLimit sets the memory limit for resource management.
func (b *Builder) WithMemoryLimit(bytes uint64) *Builder {
	b.cfg.Resource.MaxMemoryBytes = bytes
	b.cfg.Parallel.MemoryLimitBytes = bytes
	return b
}

// WithMaxGoroutines sets the maximum number of concurrent goroutines.
func (b *Builder) WithMaxGoroutines(max int) *Builder {
	if max > 0 {
		b.cfg.Resource.MaxGoroutines = max
	}
	return b
}

// WithProgressCallback sets the progress callback for batch processing.
func (b *Builder) WithProgressCallback(callback ProgressCallback) *Builder {
	b.cfg.Parallel.ProgressCallback = callback
	return b
}

// WithResourceThreshold sets the memory pressure threshold (0.0-1.0).
func (b *Builder) WithResourceThreshold(threshold float64) *Builder {
	if threshold > 0 && threshold <= 1.0 {
		b.cfg.Resource.MemoryThreshold = threshold
	}
	return b
}

// WithAdaptiveScaling enables or disables adaptive worker scaling.
func (b *Builder) WithAdaptiveScaling(enabled bool) *Builder {
	b.cfg.Resource.EnableAdaptiveScale = enabled
	return b
}

// WithBackpressure enables or disables backpressure when resources are constrained.
func (b *Builder) WithBackpressure(enabled bool) *Builder {
	b.cfg.Resource.EnableBackpressure = enabled
	return b
}

// WithGPU enables GPU acceleration for all components.
func (b *Builder) WithGPU(enabled bool) *Builder {
	b.cfg.Detector.GPU.UseGPU = enabled
	b.cfg.Recognizer.GPU.UseGPU = enabled
	b.cfg.Orientation.GPU.UseGPU = enabled
	b.cfg.TextLineOrientation.GPU.UseGPU = enabled
	return b
}

// WithGPUDevice sets the CUDA device ID for all components.
func (b *Builder) WithGPUDevice(deviceID int) *Builder {
	b.cfg.Detector.GPU.DeviceID = deviceID
	b.cfg.Recognizer.GPU.DeviceID = deviceID
	b.cfg.Orientation.GPU.DeviceID = deviceID
	b.cfg.TextLineOrientation.GPU.DeviceID = deviceID
	return b
}

// WithGPUMemoryLimit sets the GPU memory limit for all components.
func (b *Builder) WithGPUMemoryLimit(limitBytes uint64) *Builder {
	b.cfg.Detector.GPU.GPUMemLimit = limitBytes
	b.cfg.Recognizer.GPU.GPUMemLimit = limitBytes
	b.cfg.Orientation.GPU.GPUMemLimit = limitBytes
	b.cfg.TextLineOrientation.GPU.GPUMemLimit = limitBytes
	return b
}

// Config returns a copy of the current config.
func (b *Builder) Config() Config { return b.cfg }

// Validate checks that model files exist and configuration looks sane.
func (b *Builder) Validate() error {
	// Ensure model paths are updated for the current models dir
	b.cfg.Detector.UpdateModelPath(b.cfg.ModelsDir)
	b.cfg.Recognizer.UpdateModelPath(b.cfg.ModelsDir)

	if b.cfg.Detector.ModelPath == "" {
		return errors.New("detector model path is empty")
	}
	if b.cfg.Recognizer.ModelPath == "" {
		return errors.New("recognizer model path is empty")
	}
	if b.cfg.Recognizer.DictPath == "" {
		return errors.New("recognizer dictionary path is empty")
	}
	if _, err := os.Stat(b.cfg.Detector.ModelPath); err != nil {
		return fmt.Errorf("detector model not found: %s", b.cfg.Detector.ModelPath)
	}
	if _, err := os.Stat(b.cfg.Recognizer.ModelPath); err != nil {
		return fmt.Errorf("recognizer model not found: %s", b.cfg.Recognizer.ModelPath)
	}
	if _, err := os.Stat(b.cfg.Recognizer.DictPath); err != nil {
		return fmt.Errorf("dictionary not found: %s", b.cfg.Recognizer.DictPath)
	}
	if b.cfg.Recognizer.ImageHeight <= 0 {
		return errors.New("recognizer image height must be > 0")
	}
	return nil
}

// Pipeline wires together the detector and recognizer.
type Pipeline struct {
	cfg             Config
	Detector        *detector.Detector
	Recognizer      *recognizer.Recognizer
	Orienter        *orientation.Classifier
	Rectifier       *rectify.Rectifier
	ResourceManager *ResourceManager
}

// Build initializes the OCR pipeline components.
func (b *Builder) Build() (*Pipeline, error) {
	// Update model paths prior to build
	b.cfg.Detector.UpdateModelPath(b.cfg.ModelsDir)
	b.cfg.Recognizer.UpdateModelPath(b.cfg.ModelsDir)
	b.cfg.Orientation.UpdateModelPath(b.cfg.ModelsDir)
	b.cfg.TextLineOrientation.UpdateModelPath(b.cfg.ModelsDir)
	b.cfg.Rectification.UpdateModelPath(b.cfg.ModelsDir)

	if err := b.Validate(); err != nil {
		return nil, err
	}

	det, err := detector.NewDetector(b.cfg.Detector)
	if err != nil {
		return nil, fmt.Errorf("init detector: %w", err)
	}
	rec, err := recognizer.NewRecognizer(b.cfg.Recognizer)
	if err != nil {
		_ = det.Close()
		return nil, fmt.Errorf("init recognizer: %w", err)
	}
	p := &Pipeline{cfg: b.cfg, Detector: det, Recognizer: rec}

	// Optional orientation
	if b.cfg.Orientation.Enabled || b.cfg.EnableOrientation {
		cls, err := orientation.NewClassifier(b.cfg.Orientation)
		if err != nil {
			// Do not fail pipeline if orientation is optional; keep running without it
			// Users can inspect info() to see if it's active.
		} else {
			p.Orienter = cls
		}
	}
	// Optional text line per-region orientation
	if b.cfg.TextLineOrientation.Enabled {
		tl, err := orientation.NewClassifier(b.cfg.TextLineOrientation)
		if err == nil && tl != nil {
			p.Recognizer.SetTextLineOrienter(tl)
		}
	}

	// Optional rectification (minimal; does not fail pipeline)
	if b.cfg.Rectification.Enabled {
		rx, err := rectify.New(b.cfg.Rectification)
		if err == nil && rx != nil {
			p.Rectifier = rx
		}
	}

	// Optional warmup
	if b.cfg.WarmupIterations > 0 {
		if err := p.Detector.Warmup(b.cfg.WarmupIterations); err != nil {
			_ = p.Close()
			return nil, fmt.Errorf("detector warmup failed: %w", err)
		}
		if err := p.Recognizer.Warmup(b.cfg.WarmupIterations); err != nil {
			_ = p.Close()
			return nil, fmt.Errorf("recognizer warmup failed: %w", err)
		}
	}

	// Initialize resource manager for parallel processing
	if b.cfg.Resource.MaxMemoryBytes > 0 || b.cfg.Resource.MaxGoroutines > 0 || b.cfg.Resource.EnableAdaptiveScale {
		p.ResourceManager = NewResourceManager(b.cfg.Resource)
		p.ResourceManager.Start()
	}

	return p, nil
}

// Close releases all resources.
func (p *Pipeline) Close() error {
	var firstErr error
	if p.ResourceManager != nil {
		p.ResourceManager.Stop()
		p.ResourceManager = nil
	}
	if p.Orienter != nil {
		p.Orienter.Close()
		p.Orienter = nil
	}
	if p.Rectifier != nil {
		p.Rectifier.Close()
		p.Rectifier = nil
	}
	if p.Recognizer != nil {
		if err := p.Recognizer.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		p.Recognizer = nil
	}
	if p.Detector != nil {
		if err := p.Detector.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		p.Detector = nil
	}
	return firstErr
}

// Config returns the pipeline configuration.
func (p *Pipeline) Config() Config { return p.cfg }

// Info returns a map with key pipeline properties and model info.
func (p *Pipeline) Info() map[string]interface{} {
	info := map[string]interface{}{
		"models_dir":         p.cfg.ModelsDir,
		"enable_orientation": p.cfg.EnableOrientation,
	}
	if p.Orienter != nil {
		info["orientation"] = map[string]interface{}{
			"enabled":              true,
			"model_path":           p.cfg.Orientation.ModelPath,
			"confidence_threshold": p.cfg.Orientation.ConfidenceThreshold,
			"heuristic":            p.cfg.Orientation.UseHeuristicFallback && p.Orienter != nil,
		}
	} else {
		info["orientation"] = map[string]interface{}{"enabled": false}
	}
	// Rectification info
	info["rectification"] = map[string]interface{}{
		"enabled":    p.cfg.Rectification.Enabled,
		"model_path": p.cfg.Rectification.ModelPath,
	}
	// Text line orientation config exposure
	info["textline_orientation"] = map[string]interface{}{
		"enabled":              p.cfg.TextLineOrientation.Enabled,
		"model_path":           p.cfg.TextLineOrientation.ModelPath,
		"confidence_threshold": p.cfg.TextLineOrientation.ConfidenceThreshold,
	}
	if p.Detector != nil {
		info["detector"] = p.Detector.GetModelInfo()
	}
	if p.Recognizer != nil {
		info["recognizer"] = p.Recognizer.GetModelInfo()
	}

	// Parallel processing configuration
	info["parallel"] = map[string]interface{}{
		"max_workers":           p.cfg.Parallel.MaxWorkers,
		"batch_size":            p.cfg.Parallel.BatchSize,
		"memory_limit_bytes":    p.cfg.Parallel.MemoryLimitBytes,
		"has_progress_callback": p.cfg.Parallel.ProgressCallback != nil,
	}

	// Resource management configuration
	info["resource_management"] = map[string]interface{}{
		"max_memory_bytes":      p.cfg.Resource.MaxMemoryBytes,
		"max_goroutines":        p.cfg.Resource.MaxGoroutines,
		"memory_threshold":      p.cfg.Resource.MemoryThreshold,
		"enable_backpressure":   p.cfg.Resource.EnableBackpressure,
		"enable_adaptive_scale": p.cfg.Resource.EnableAdaptiveScale,
		"active":                p.ResourceManager != nil,
	}

	// Include current resource stats if manager is active
	if p.ResourceManager != nil {
		info["resource_stats"] = p.ResourceManager.GetStats()
	}

	return info
}
