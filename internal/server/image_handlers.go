package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/pipeline"
)

// RequestConfig holds per-request configuration overrides
type RequestConfig struct {
	Language  string
	DictLangs []string
	DictPath  string
	DetModel  string
	RecModel  string
}

// ocrImageHandler processes image OCR requests.
func (s *Server) ocrImageHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse and validate request
	img, reqConfig, err := s.parseImageRequest(w, r)
	if err != nil {
		ocrRequestsTotal.WithLabelValues("image", "error").Inc()
		return // error already written
	}

	// Get pipeline for this request
	pipeline, err := s.getPipelineForRequest(reqConfig)
	if err != nil {
		s.writeErrorResponse(w, fmt.Sprintf("Failed to create pipeline: %v", err), http.StatusInternalServerError)
		ocrRequestsTotal.WithLabelValues("image", "error").Inc()
		return
	}

	// Run full OCR pipeline with timing
	start := time.Now()
	res, err := pipeline.ProcessImage(img)
	duration := time.Since(start)

	if err != nil {
		ocrRequestsTotal.WithLabelValues("image", "error").Inc()
		s.writeErrorResponse(w, fmt.Sprintf("OCR processing failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Record successful metrics
	ocrRequestsTotal.WithLabelValues("image", "success").Inc()
	ocrProcessingDuration.WithLabelValues("image").Observe(duration.Seconds())

	// Calculate total text length from all regions
	var totalTextLength int
	for _, region := range res.Regions {
		totalTextLength += len(region.Text)
	}
	ocrTextLength.WithLabelValues("image").Observe(float64(totalTextLength))
	ocrRegionsDetected.WithLabelValues("image").Observe(float64(len(res.Regions)))

	// Format and send response
	s.writeImageResponse(w, r, img, res)
}

func (s *Server) parseImageRequest(w http.ResponseWriter, r *http.Request) (image.Image, *RequestConfig, error) {
	// Set content length limit
	r.Body = http.MaxBytesReader(w, r.Body, s.maxUploadMB*1024*1024)

	// Parse multipart form
	err := r.ParseMultipartForm(s.maxUploadMB * 1024 * 1024)
	if err != nil {
		s.writeErrorResponse(w, "Failed to parse form data", http.StatusBadRequest)
		return nil, nil, err
	}

	// Get uploaded file
	file, header, err := r.FormFile("image")
	if err != nil {
		s.writeErrorResponse(w, "No image file provided", http.StatusBadRequest)
		return nil, nil, err
	}
	defer func() { _ = file.Close() }()

	// Validate file size
	if header.Size > s.maxUploadMB*1024*1024 {
		s.writeErrorResponse(w, "File too large", http.StatusRequestEntityTooLarge)
		return nil, nil, err
	}

	// Record upload size metric
	uploadSizeBytes.Observe(float64(header.Size))

	// Read file content
	imageData, err := io.ReadAll(file)
	if err != nil {
		s.writeErrorResponse(w, "Failed to read image data", http.StatusInternalServerError)
		return nil, nil, err
	}

	// Decode image
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		s.writeErrorResponse(w, "Invalid image format", http.StatusBadRequest)
		return nil, nil, err
	}

	// Extract request configuration
	reqConfig := &RequestConfig{
		Language: r.FormValue("language"),
		DictPath: r.FormValue("dict"),
		DetModel: r.FormValue("det-model"),
		RecModel: r.FormValue("rec-model"),
	}

	// Parse dict-langs
	if dictLangsStr := r.FormValue("dict-langs"); dictLangsStr != "" {
		reqConfig.DictLangs = strings.Split(dictLangsStr, ",")
		for i, lang := range reqConfig.DictLangs {
			reqConfig.DictLangs[i] = strings.TrimSpace(lang)
		}
	}

	return img, reqConfig, nil
}

func (s *Server) writeImageResponse(
	w http.ResponseWriter,
	r *http.Request,
	img image.Image,
	res *pipeline.OCRImageResult,
) {
	// Determine output format: default json; allow 'format' in query or form
	format := r.FormValue("format")
	if format == "" {
		format = r.URL.Query().Get("format")
	}

	switch format {
	case "csv":
		s.writeCSVResponse(w, res)
	case formatText:
		s.writeTextResponse(w, res)
	case "overlay":
		s.handleOverlayOutput(w, r, img, res)
	default:
		// Check for overlay parameter
		if r.FormValue("overlay") == "1" {
			s.handleOverlayOutput(w, r, img, res)
		} else {
			s.writeJSONResponse(w, res)
		}
	}
}

func (s *Server) writeCSVResponse(w http.ResponseWriter, res *pipeline.OCRImageResult) {
	w.Header().Set("Content-Type", "text/csv")
	csvStr, err := pipeline.ToCSVImage(res)
	if err != nil {
		http.Error(w, fmt.Sprintf("formatting failed: %v", err), http.StatusInternalServerError)
		return
	}
	_, _ = w.Write([]byte(csvStr))
}

func (s *Server) writeTextResponse(w http.ResponseWriter, res *pipeline.OCRImageResult) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	textStr, err := pipeline.ToPlainTextImage(res)
	if err != nil {
		http.Error(w, fmt.Sprintf("formatting failed: %v", err), http.StatusInternalServerError)
		return
	}
	_, _ = w.Write([]byte(textStr))
}

func (s *Server) writeJSONResponse(w http.ResponseWriter, res *pipeline.OCRImageResult) {
	w.Header().Set("Content-Type", "application/json")
	obj := struct {
		OCR *pipeline.OCRImageResult `json:"ocr"`
	}{OCR: res}
	if err := json.NewEncoder(w).Encode(obj); err != nil {
		fmt.Fprintf(os.Stderr, "Error encoding OCR image response: %v\n", err)
	}
}

// parseHexColor parses colors like "#RRGGBB" or "RRGGBB".
func parseHexColor(s string) color.Color {
	if s == "" {
		return nil
	}
	if s[0] == '#' {
		s = s[1:]
	}
	if len(s) != 6 {
		return nil
	}
	var r, g, b uint8
	var rv, gv, bv int
	if _, err := fmt.Sscanf(s, "%02x%02x%02x", &rv, &gv, &bv); err != nil {
		return nil
	}
	r, g, b = uint8(rv), uint8(gv), uint8(bv) //nolint:gosec // G115: Safe conversion for RGB color values
	return color.RGBA{r, g, b, 255}
}

// handleOverlayOutput handles overlay image output for OCR results.
func (s *Server) handleOverlayOutput(
	w http.ResponseWriter,
	r *http.Request,
	img image.Image,
	res *pipeline.OCRImageResult,
) {
	if !s.overlayEnabled {
		http.Error(w, "overlay output disabled", http.StatusForbidden)
		return
	}

	boxCol := parseHexColor(r.FormValue("box"))
	if boxCol == nil {
		boxCol = parseHexColor(s.overlayBoxColor)
	}
	if boxCol == nil {
		boxCol = color.RGBA{255, 0, 0, 255}
	}

	polyCol := parseHexColor(r.FormValue("poly"))
	if polyCol == nil {
		polyCol = parseHexColor(s.overlayPolyColor)
	}
	if polyCol == nil {
		polyCol = color.RGBA{0, 255, 0, 255}
	}

	ov := pipeline.RenderOverlay(img, res, boxCol, polyCol)
	if ov == nil {
		http.Error(w, "overlay failed", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "image/png")
	_ = png.Encode(w, ov)
}

// getPipelineForRequest returns a pipeline configured for the specific request.
// Creates or retrieves a cached pipeline based on the request configuration.
func (s *Server) getPipelineForRequest(reqConfig *RequestConfig) (pipelineInterface, error) {
	// Start with base configuration
	config := s.baseConfig

	// Override detector model if specified
	if reqConfig.DetModel != "" {
		config.Detector.ModelPath = reqConfig.DetModel
	}

	// Override recognizer model if specified
	if reqConfig.RecModel != "" {
		config.Recognizer.ModelPath = reqConfig.RecModel
	}

	// Override language if specified
	if reqConfig.Language != "" {
		config.Recognizer.Language = reqConfig.Language
	}

	// Override dictionary if specified
	if reqConfig.DictPath != "" {
		config.Recognizer.DictPath = reqConfig.DictPath
		config.Recognizer.DictPaths = nil // Clear multiple dicts when single dict is specified
	}

	// Override dictionary languages if specified
	if len(reqConfig.DictLangs) > 0 {
		dictPaths := models.GetDictionaryPathsForLanguages(config.ModelsDir, reqConfig.DictLangs)
		if len(dictPaths) > 0 {
			config.Recognizer.DictPaths = dictPaths
			config.Recognizer.DictPath = "" // Clear single dict when multiple dicts are specified
		}
	}

	// Get or create cached pipeline
	return s.pipelineCache.GetOrCreate(config)
}
