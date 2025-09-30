package server

import (
	"fmt"
	"hash/fnv"
	"image"
	"net/http"
	"strconv"
	"sync"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
)

// pipelineInterface defines the methods needed by the server from a pipeline.
type pipelineInterface interface {
	ProcessImage(img image.Image) (*pipeline.OCRImageResult, error)
	ProcessPDF(filename string, pageRange string) (*pipeline.OCRPDFResult, error)
	Close() error
}

// PipelineCache caches pipelines by configuration to avoid recreating them.
type PipelineCache struct {
	mu        sync.RWMutex
	pipelines map[string]pipelineInterface
}

// NewPipelineCache creates a new pipeline cache.
func NewPipelineCache() *PipelineCache {
	return &PipelineCache{
		pipelines: make(map[string]pipelineInterface),
	}
}

// GetOrCreate gets a cached pipeline or creates a new one for the given config.
func (c *PipelineCache) GetOrCreate(config pipeline.Config) (pipelineInterface, error) {
	// Create a hash of the configuration for caching
	configHash := c.hashConfig(config)

	c.mu.RLock()
	if pipeline, exists := c.pipelines[configHash]; exists {
		c.mu.RUnlock()
		return pipeline, nil
	}
	c.mu.RUnlock()

	// Create new pipeline
	c.mu.Lock()
	defer c.mu.Unlock()

	// Double-check in case another goroutine created it
	if pipeline, exists := c.pipelines[configHash]; exists {
		return pipeline, nil
	}

	pipeline, err := c.createPipeline(config)
	if err != nil {
		return nil, err
	}

	c.pipelines[configHash] = pipeline
	return pipeline, nil
}

// hashConfig creates a hash of the pipeline configuration for caching.
func (c *PipelineCache) hashConfig(config pipeline.Config) string {
	// Create a string representation of key configuration fields
	key := fmt.Sprintf("%s|%s|%s|%s|%s",
		config.ModelsDir,
		config.Detector.ModelPath,
		config.Recognizer.ModelPath,
		fmt.Sprintf("%v", config.Recognizer.DictPaths),
		config.Recognizer.Language,
	)
	
	h := fnv.New64a()
	_, err := h.Write([]byte(key))
	if err != nil {
		// FNV hash write should never fail, but handle it gracefully
		return fmt.Sprintf("%x", 0)
	}
	return strconv.FormatUint(h.Sum64(), 16)
}

// createPipeline creates a new pipeline with the given configuration.
func (c *PipelineCache) createPipeline(config pipeline.Config) (pipelineInterface, error) {
	builder := pipeline.NewBuilder().
		WithModelsDir(config.ModelsDir).
		WithDetectorModelPath(config.Detector.ModelPath).
		WithRecognizerModelPath(config.Recognizer.ModelPath).
		WithLanguage(config.Recognizer.Language)

	// Set dictionary paths if specified
	if len(config.Recognizer.DictPaths) > 0 {
		builder = builder.WithDictionaryPaths(config.Recognizer.DictPaths)
	} else if config.Recognizer.DictPath != "" {
		builder = builder.WithDictionaryPath(config.Recognizer.DictPath)
	}

	// Apply other configuration options
	builder = builder.WithThreads(config.Detector.NumThreads)
	builder = builder.WithDetectorThresholds(config.Detector.DbThresh, config.Detector.DbBoxThresh)
	if config.Detector.UseNMS {
		builder = builder.WithDetectorNMS(true, config.Detector.NMSThreshold)
	}
	builder = builder.WithImageHeight(config.Recognizer.ImageHeight)
	builder = builder.WithRecognizeWidthPadding(config.Recognizer.MaxWidth, config.Recognizer.PadWidthMultiple)

	return builder.Build()
}

// Close closes all cached pipelines.
func (c *PipelineCache) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	var firstErr error
	for _, pipeline := range c.pipelines {
		if err := pipeline.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	c.pipelines = nil
	return firstErr
}

// Server holds the HTTP server state and dependencies.
type Server struct {
	pipeline         pipelineInterface
	corsOrigin       string
	maxUploadMB      int64
	timeoutSec       int
	overlayEnabled   bool
	overlayBoxColor  string
	overlayPolyColor string
	rateLimiter      *RateLimiter
	pipelineCache    *PipelineCache
	baseConfig       pipeline.Config // Base configuration for creating custom pipelines
}

// Config holds server configuration.
type Config struct {
	Host             string
	Port             int
	CORSOrigin       string
	MaxUploadMB      int64
	TimeoutSec       int
	PipelineConfig   pipeline.Config
	OverlayEnabled   bool
	OverlayBoxColor  string
	OverlayPolyColor string
	RateLimit        RateLimitConfig
}

// RateLimitConfig holds rate limiting configuration.
type RateLimitConfig struct {
	Enabled           bool
	RequestsPerMinute int
	RequestsPerHour   int
	MaxRequestsPerDay int
	MaxDataPerDay     int64 // in bytes
}

// Response types for API endpoints.
type HealthResponse struct {
	Status  string `json:"status"`
	Version string `json:"version,omitempty"`
	Time    string `json:"time"`
}

type ModelInfo struct {
	Name        string      `json:"name"`
	Path        string      `json:"path"`
	Type        string      `json:"type"`
	Description string      `json:"description"`
	Config      interface{} `json:"config,omitempty"`
}

type ModelsResponse struct {
	Models []ModelInfo `json:"models"`
	Count  int         `json:"count"`
}

type DetectionBox struct {
	X1         float64 `json:"x1"`
	Y1         float64 `json:"y1"`
	X2         float64 `json:"x2"`
	Y2         float64 `json:"y2"`
	Confidence float64 `json:"confidence"`
}

type OCRResult struct {
	Text       string         `json:"text,omitempty"`
	Boxes      []DetectionBox `json:"boxes,omitempty"`
	Language   string         `json:"language,omitempty"`
	Confidence float64        `json:"confidence"`
	Width      int            `json:"width"`
	Height     int            `json:"height"`
	Processing struct {
		DetectionTimeMs int64 `json:"detection_time_ms"`
		TotalTimeMs     int64 `json:"total_time_ms"`
	} `json:"processing"`
}

type OCRResponse struct {
	Success bool      `json:"success"`
	Result  OCRResult `json:"result,omitempty"`
	Error   string    `json:"error,omitempty"`
}

// NewServer creates a new OCR server instance.
func NewServer(config Config) (*Server, error) {
	// Build pipeline from provided config
	cfg := config.PipelineConfig
	cfg.Detector.UpdateModelPath(cfg.ModelsDir)
	cfg.Recognizer.UpdateModelPath(cfg.ModelsDir)

	// Build components using builder fluent API
	nb := pipeline.NewBuilder().WithModelsDir(cfg.ModelsDir).WithLanguage(cfg.Recognizer.Language)
	nb = nb.WithThreads(cfg.Detector.NumThreads)
	nb = nb.WithDetectorThresholds(cfg.Detector.DbThresh, cfg.Detector.DbBoxThresh)
	if cfg.Detector.UseNMS {
		nb = nb.WithDetectorNMS(true, cfg.Detector.NMSThreshold)
	}
	nb = nb.WithImageHeight(cfg.Recognizer.ImageHeight)
	nb = nb.WithRecognizeWidthPadding(cfg.Recognizer.MaxWidth, cfg.Recognizer.PadWidthMultiple)
	if cfg.Detector.ModelPath != "" {
		nb = nb.WithDetectorModelPath(cfg.Detector.ModelPath)
	}
	if cfg.Recognizer.ModelPath != "" {
		nb = nb.WithRecognizerModelPath(cfg.Recognizer.ModelPath)
	}
	if cfg.Recognizer.DictPath != "" {
		nb = nb.WithDictionaryPath(cfg.Recognizer.DictPath)
	}

	pl, err := nb.Build()
	if err != nil {
		return nil, err
	}

	// Initialize rate limiter if enabled
	var rateLimiter *RateLimiter
	if config.RateLimit.Enabled {
		rateLimiter = NewRateLimiter(
			config.RateLimit.RequestsPerMinute,
			config.RateLimit.RequestsPerHour,
			config.RateLimit.MaxRequestsPerDay,
			config.RateLimit.MaxDataPerDay,
		)
	}

	return &Server{
		pipeline:         pl,
		corsOrigin:       config.CORSOrigin,
		maxUploadMB:      config.MaxUploadMB,
		timeoutSec:       config.TimeoutSec,
		overlayEnabled:   config.OverlayEnabled,
		overlayBoxColor:  config.OverlayBoxColor,
		overlayPolyColor: config.OverlayPolyColor,
		rateLimiter:      rateLimiter,
		pipelineCache:    NewPipelineCache(),
		baseConfig:       config.PipelineConfig,
	}, nil
}

// Close releases server resources.
func (s *Server) Close() error {
	var firstErr error

	if s.pipeline != nil {
		if err := s.pipeline.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	if s.pipelineCache != nil {
		if err := s.pipelineCache.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	return firstErr
}

// SetupRoutes configures the HTTP routes.
func (s *Server) SetupRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/health", s.corsMiddleware(s.healthHandler))
	mux.HandleFunc("/models", s.corsMiddleware(s.modelsHandler))
	mux.HandleFunc("/metrics", s.corsMiddleware(s.metricsHandler))
	mux.HandleFunc("/ws/ocr", s.corsMiddleware(s.ocrWebSocketHandler))
	mux.HandleFunc("/ocr/batch", s.corsMiddleware(s.rateLimitMiddleware(s.ocrBatchHandler)))
	mux.HandleFunc("/ocr/image", s.corsMiddleware(s.rateLimitMiddleware(s.ocrImageHandler)))
	mux.HandleFunc("/ocr/pdf", s.corsMiddleware(s.rateLimitMiddleware(s.ocrPdfHandler)))
}
