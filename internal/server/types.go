package server

import (
	"image"
	"net/http"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
)

// pipelineInterface defines the methods needed by the server from a pipeline.
type pipelineInterface interface {
	ProcessImage(img image.Image) (*pipeline.OCRImageResult, error)
	ProcessPDF(filename string, pageRange string) (*pipeline.OCRPDFResult, error)
	Close() error
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

	return &Server{
		pipeline:         pl,
		corsOrigin:       config.CORSOrigin,
		maxUploadMB:      config.MaxUploadMB,
		timeoutSec:       config.TimeoutSec,
		overlayEnabled:   config.OverlayEnabled,
		overlayBoxColor:  config.OverlayBoxColor,
		overlayPolyColor: config.OverlayPolyColor,
	}, nil
}

// Close releases server resources.
func (s *Server) Close() error {
	if s.pipeline != nil {
		return s.pipeline.Close()
	}
	return nil
}

// SetupRoutes configures the HTTP routes.
func (s *Server) SetupRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/health", s.corsMiddleware(s.healthHandler))
	mux.HandleFunc("/models", s.corsMiddleware(s.modelsHandler))
	mux.HandleFunc("/ocr/image", s.corsMiddleware(s.ocrImageHandler))
	mux.HandleFunc("/ocr/pdf", s.corsMiddleware(s.ocrPdfHandler))
}
