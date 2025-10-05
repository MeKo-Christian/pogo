package server

import (
    "bytes"
    "encoding/json"
    "errors"
    "fmt"
    "image"
    "image/color"
    "image/png"
    "io"
    "net/http"
    "os"
    "strconv"
    "strings"
    "time"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/pipeline"
)

// RequestConfig holds per-request configuration overrides.
type RequestConfig struct {
	Language  string
	DictLangs []string
	DictPath  string
	DetModel  string
	RecModel  string

	// PDF enhancement options
	UserPassword     string  `json:"user_password,omitempty"`
	OwnerPassword    string  `json:"owner_password,omitempty"`
	EnableVectorText bool    `json:"enable_vector_text,omitempty"`
	EnableHybrid     bool    `json:"enable_hybrid,omitempty"`
    QualityThreshold float64 `json:"quality_threshold,omitempty"`

    // Barcode options
    EnableBarcodes  bool   `json:"barcodes,omitempty"`
    BarcodeTypes    string `json:"barcode_types,omitempty"`
    BarcodeMinSize  int    `json:"barcode_min_size,omitempty"`
}

// validateLanguageCode validates a language code format.
func validateLanguageCode(code string) error {
	if len(code) > 10 {
		return fmt.Errorf("language code too long: %s", code)
	}
	// Basic validation - should be letters (case insensitive), optionally with numbers/hyphens/underscores
	for _, r := range strings.ToLower(code) {
		if (r < 'a' || r > 'z') && (r < '0' || r > '9') && r != '-' && r != '_' {
			return fmt.Errorf("invalid language code format: %s", code)
		}
	}
	return nil
}

// validatePath validates a file path for basic security.
func validatePath(path string, fieldName string) error {
	if len(path) > 500 {
		return fmt.Errorf("%s path too long", fieldName)
	}
	// Check for obviously dangerous characters
	if strings.Contains(path, "..") || strings.Contains(path, "\n") || strings.Contains(path, "\r") {
		return fmt.Errorf("invalid %s path", fieldName)
	}
	return nil
}

// Validate validates the request configuration parameters.
func (c *RequestConfig) Validate() error {
	// Validate language code
	if c.Language != "" {
		if err := validateLanguageCode(c.Language); err != nil {
			return err
		}
	}

	// Validate dict-langs
	for _, lang := range c.DictLangs {
		if lang == "" {
			continue
		}
		if err := validateLanguageCode(lang); err != nil {
			return fmt.Errorf("dictionary %s", err.Error())
		}
	}

	// Validate model paths - basic sanity checks
	validations := []struct {
		value string
		name  string
	}{
		{c.DetModel, "detector model"},
		{c.RecModel, "recognizer model"},
		{c.DictPath, "dictionary"},
	}

	for _, v := range validations {
		if v.value != "" {
			if err := validatePath(v.value, v.name); err != nil {
				return err
			}
		}
	}

	return nil
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

    // Barcode options (per-request overrides)
    if v := r.FormValue("barcodes"); v != "" {
        reqConfig.EnableBarcodes = v == "1" || strings.ToLower(v) == "true"
    }
    if v := r.FormValue("barcode-types"); v != "" {
        reqConfig.BarcodeTypes = v
    }
    if v := r.FormValue("barcode-min-size"); v != "" {
        if n, err := strconv.Atoi(v); err == nil {
            reqConfig.BarcodeMinSize = n
        }
    }

	// Parse dict-langs
	if dictLangsStr := r.FormValue("dict-langs"); dictLangsStr != "" {
		reqConfig.DictLangs = strings.Split(dictLangsStr, ",")
		for i, lang := range reqConfig.DictLangs {
			reqConfig.DictLangs[i] = strings.TrimSpace(lang)
		}
	}

	// Validate request configuration
	if err := reqConfig.Validate(); err != nil {
		s.writeErrorResponse(w, fmt.Sprintf("Invalid request parameters: %v", err), http.StatusBadRequest)
		return nil, nil, err
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
	// If no custom configuration is requested, use the default pipeline
	hasCustomConfig := reqConfig.DetModel != "" || reqConfig.RecModel != "" ||
		reqConfig.Language != "" || reqConfig.DictPath != "" || len(reqConfig.DictLangs) > 0

	if !hasCustomConfig && s.pipeline != nil {
		return s.pipeline, nil
	}

	// If pipeline cache is not available, return error
	if s.pipelineCache == nil {
		if s.pipeline != nil {
			return s.pipeline, nil
		}
		return nil, errors.New("OCR pipeline not initialized")
	}

	// Start with base configuration and apply overrides
	config := s.applyRequestOverrides(s.baseConfig, reqConfig)

	// Get or create cached pipeline
	return s.pipelineCache.GetOrCreate(config)
}

// applyRequestOverrides applies request-specific configuration overrides to the base config.
func (s *Server) applyRequestOverrides(baseConfig pipeline.Config, reqConfig *RequestConfig) pipeline.Config {
	config := baseConfig

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

    // Barcode overrides
    if reqConfig.EnableBarcodes || reqConfig.BarcodeTypes != "" || reqConfig.BarcodeMinSize > 0 {
        config.Barcode.Enabled = reqConfig.EnableBarcodes || config.Barcode.Enabled
        if strings.TrimSpace(reqConfig.BarcodeTypes) != "" {
            config.Barcode.Types = strings.Split(reqConfig.BarcodeTypes, ",")
        }
        if reqConfig.BarcodeMinSize > 0 {
            config.Barcode.MinSize = reqConfig.BarcodeMinSize
        }
    }

    return config
}
