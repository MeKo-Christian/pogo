package server

import (
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/pdf"
	"github.com/MeKo-Tech/pogo/internal/pipeline"
)

// ocrPdfHandler processes PDF OCR requests.
func (s *Server) ocrPdfHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse and validate request
	file, header, pageRange, reqConfig, err := s.parsePdfRequest(w, r)
	if err != nil {
		ocrRequestsTotal.WithLabelValues("pdf", "error").Inc()
		return // error already written
	}
	defer func() { _ = file.Close() }()

	// Save uploaded file to temporary location
	tempFile, err := s.saveUploadedFile(file, header)
	if err != nil {
		s.writeErrorResponse(w, fmt.Sprintf("Failed to save uploaded file: %v", err), http.StatusInternalServerError)
		ocrRequestsTotal.WithLabelValues("pdf", "error").Inc()
		return
	}
	defer func() { _ = os.Remove(tempFile) }() // Clean up temp file

	// Process PDF and handle response
	s.processPdfAndRespond(w, r, tempFile, pageRange, reqConfig)
}

// processPdfAndRespond handles PDF processing and response formatting.
func (s *Server) processPdfAndRespond(w http.ResponseWriter, r *http.Request,
	tempFile, pageRange string, reqConfig *RequestConfig,
) {
	// Check if we need enhanced PDF processing (passwords or vector text)
	needsEnhanced := reqConfig.UserPassword != "" || reqConfig.OwnerPassword != "" ||
		reqConfig.EnableVectorText || reqConfig.EnableHybrid

	var res *pipeline.OCRPDFResult
	var duration time.Duration
	var err error

	if needsEnhanced {
		// Use enhanced PDF processor
		start := time.Now()
		res, err = s.processEnhancedPDF(tempFile, pageRange, reqConfig)
		duration = time.Since(start)
	} else {
		// Use standard pipeline processing
		pipeline, pipelineErr := s.getPipelineForRequest(reqConfig)
		if pipelineErr != nil {
			s.writeErrorResponse(w, fmt.Sprintf("Failed to create pipeline: %v", pipelineErr), http.StatusInternalServerError)
			ocrRequestsTotal.WithLabelValues("pdf", "error").Inc()
			return
		}

		start := time.Now()
		res, err = pipeline.ProcessPDF(tempFile, pageRange)
		duration = time.Since(start)
	}

	if err != nil {
		ocrRequestsTotal.WithLabelValues("pdf", "error").Inc()
		s.writeErrorResponse(w, fmt.Sprintf("OCR processing failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Record successful metrics
	ocrRequestsTotal.WithLabelValues("pdf", "success").Inc()
	ocrProcessingDuration.WithLabelValues("pdf").Observe(duration.Seconds())

	// Calculate total text length and regions from all pages
	var totalTextLength int
	var totalRegions int
	for _, page := range res.Pages {
		for _, img := range page.Images {
			totalRegions += len(img.Regions)
			for _, region := range img.Regions {
				totalTextLength += len(region.Text)
			}
		}
	}
	ocrTextLength.WithLabelValues("pdf").Observe(float64(totalTextLength))
	ocrRegionsDetected.WithLabelValues("pdf").Observe(float64(totalRegions))

	// Format and send response
	s.writePdfResponse(w, r, res)
}

func (s *Server) parsePdfRequest(
	w http.ResponseWriter,
	r *http.Request,
) (multipart.File, *multipart.FileHeader, string, *RequestConfig, error) {
	// Set content length limit
	r.Body = http.MaxBytesReader(w, r.Body, s.maxUploadMB*1024*1024)

	// Parse multipart form
	err := r.ParseMultipartForm(s.maxUploadMB * 1024 * 1024)
	if err != nil {
		s.handleFormParseError(w, err)
		return nil, nil, "", nil, err
	}

	// Get uploaded file
	file, header, err := r.FormFile("pdf")
	if err != nil {
		s.writeErrorResponse(w, "No PDF file provided", http.StatusBadRequest)
		return nil, nil, "", nil, err
	}

	// Validate file size
	if header.Size > s.maxUploadMB*1024*1024 {
		s.writeErrorResponse(w, "File too large", http.StatusRequestEntityTooLarge)
		return nil, nil, "", nil, err
	}

	// Record upload size metric
	uploadSizeBytes.Observe(float64(header.Size))

	// Get page range parameter
	pageRange := r.FormValue("pages")

	// Extract and validate request configuration
	reqConfig, err := s.parseRequestConfig(r)
	if err != nil {
		s.writeErrorResponse(w, fmt.Sprintf("Invalid request parameters: %v", err), http.StatusBadRequest)
		return nil, nil, "", nil, err
	}

	return file, header, pageRange, reqConfig, nil
}

// parseRequestConfig extracts and validates request configuration from form values.
func (s *Server) parseRequestConfig(r *http.Request) (*RequestConfig, error) {
	reqConfig := &RequestConfig{
		Language: r.FormValue("language"),
		DictPath: r.FormValue("dict"),
		DetModel: r.FormValue("det-model"),
		RecModel: r.FormValue("rec-model"),

		// PDF enhancement options
		UserPassword:     r.FormValue("user-password"),
		OwnerPassword:    r.FormValue("owner-password"),
		EnableVectorText: r.FormValue("enable-vector-text") != "false", // Default to true
		EnableHybrid:     r.FormValue("enable-hybrid") == "true",
	}

	// Parse quality threshold
	if qthreshStr := r.FormValue("quality-threshold"); qthreshStr != "" {
		if qthresh, err := strconv.ParseFloat(qthreshStr, 64); err == nil && qthresh > 0 && qthresh <= 1.0 {
			reqConfig.QualityThreshold = qthresh
		}
	}
	if reqConfig.QualityThreshold == 0 {
		reqConfig.QualityThreshold = 0.7 // Default threshold
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
		return nil, err
	}

	return reqConfig, nil
}

func (s *Server) handleFormParseError(w http.ResponseWriter, err error) {
	// Distinguish body-too-large from generic parse error
	if strings.Contains(strings.ToLower(err.Error()), "body too large") ||
		strings.Contains(strings.ToLower(err.Error()), "request body too large") {
		s.writeErrorResponse(w, "File too large", http.StatusRequestEntityTooLarge)
	} else {
		s.writeErrorResponse(w, "Failed to parse form data", http.StatusBadRequest)
	}
}

func (s *Server) writePdfResponse(w http.ResponseWriter, r *http.Request, res *pipeline.OCRPDFResult) {
	// Determine output format: default json; allow 'format' in query or form
	format := r.FormValue("format")
	if format == "" {
		format = r.URL.Query().Get("format")
	}

	if format == formatText {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		s.writePDFTextResponse(w, res)
		return
	}

	// default: json
	w.Header().Set("Content-Type", "application/json")
	obj := struct {
		OCR *pipeline.OCRPDFResult `json:"ocr"`
	}{OCR: res}
	if err := json.NewEncoder(w).Encode(obj); err != nil {
		fmt.Fprintf(os.Stderr, "Error encoding OCR PDF response: %v\n", err)
	}
}

// writePDFTextResponse writes a plain text representation of PDF OCR results.
func (s *Server) writePDFTextResponse(w http.ResponseWriter, result *pipeline.OCRPDFResult) {
	var output strings.Builder

	output.WriteString(fmt.Sprintf("File: %s\n", result.Filename))
	output.WriteString(fmt.Sprintf("Total Pages: %d\n", result.TotalPages))
	output.WriteString(fmt.Sprintf("Processing Time: %dns\n\n", result.Processing.TotalNs))

	for _, page := range result.Pages {
		output.WriteString(fmt.Sprintf("Page %d (%dx%d):\n", page.PageNumber, page.Width, page.Height))

		for _, img := range page.Images {
			output.WriteString(fmt.Sprintf("  Image %d (%dx%d): %d region(s), confidence: %.3f\n",
				img.ImageIndex, img.Width, img.Height, len(img.Regions), img.Confidence))

			for i, region := range img.Regions {
				output.WriteString(fmt.Sprintf("    #%d box=(%d,%d %dx%d) conf=%.3f text='%s'\n",
					i+1,
					region.Box.X, region.Box.Y, region.Box.W, region.Box.H,
					region.RecConfidence, region.Text))
			}
		}
		output.WriteString("\n")
	}

	if _, err := w.Write([]byte(output.String())); err != nil {
		// Log error, but can't send another response
		fmt.Fprintf(os.Stderr, "Error writing response: %v\n", err)
	}
}

// saveUploadedFile saves the uploaded multipart file to a temporary location.
func (s *Server) saveUploadedFile(file multipart.File, header *multipart.FileHeader) (string, error) {
	// Create temporary file with proper extension
	ext := filepath.Ext(header.Filename)
	if ext == "" {
		ext = ".pdf"
	}

	tempFile, err := os.CreateTemp("", "upload-*"+ext)
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	defer func() { _ = tempFile.Close() }()

	// Copy uploaded file to temp location
	_, err = io.Copy(tempFile, file)
	if err != nil {
		_ = os.Remove(tempFile.Name()) // Clean up on error
		return "", fmt.Errorf("failed to copy uploaded file: %w", err)
	}

	return tempFile.Name(), nil
}

// processEnhancedPDF processes a PDF using the enhanced processor with password and vector text support.
func (s *Server) processEnhancedPDF(filename, pageRange string,
	reqConfig *RequestConfig,
) (*pipeline.OCRPDFResult, error) {
	// Create detector for enhanced PDF processor
	detectorConfig := detector.Config{
		ModelPath:    s.baseConfig.Detector.ModelPath,
		NumThreads:   s.baseConfig.Detector.NumThreads,
		DbThresh:     s.baseConfig.Detector.DbThresh,
		DbBoxThresh:  s.baseConfig.Detector.DbBoxThresh,
		UseNMS:       s.baseConfig.Detector.UseNMS,
		NMSThreshold: s.baseConfig.Detector.NMSThreshold,
	}

	// Update model paths if needed
	detectorConfig.UpdateModelPath(s.baseConfig.ModelsDir)

	det, err := detector.NewDetector(detectorConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create detector: %w", err)
	}
	defer func() { _ = det.Close() }()

	// Create enhanced PDF processor configuration
	processorConfig := &pdf.ProcessorConfig{
		EnableVectorText:    reqConfig.EnableVectorText,
		EnableHybrid:        reqConfig.EnableHybrid,
		VectorTextQuality:   reqConfig.QualityThreshold,
		VectorTextCoverage:  0.8, // Default coverage threshold
		AllowPasswords:      true,
		AllowPasswordPrompt: false, // Don't allow prompts in server mode
	}

	// Create enhanced PDF processor
	processor := pdf.NewProcessorWithConfig(det, processorConfig)

	// Set up credentials if provided
	var creds *pdf.PasswordCredentials
	if reqConfig.UserPassword != "" || reqConfig.OwnerPassword != "" {
		creds = &pdf.PasswordCredentials{
			UserPassword:  reqConfig.UserPassword,
			OwnerPassword: reqConfig.OwnerPassword,
		}
	}

	// Process the PDF with enhanced features
	var result *pdf.DocumentResult
	if creds != nil {
		result, err = processor.ProcessFileWithCredentials(filename, pageRange, creds)
	} else {
		result, err = processor.ProcessFile(filename, pageRange)
	}

	if err != nil {
		return nil, fmt.Errorf("enhanced PDF processing failed: %w", err)
	}

	// Convert enhanced result to pipeline format for compatibility
	return s.convertEnhancedResultToPipeline(result), nil
}

// convertEnhancedResultToPipeline converts enhanced PDF results to pipeline format for API compatibility.
func (s *Server) convertEnhancedResultToPipeline(result *pdf.DocumentResult) *pipeline.OCRPDFResult {
	pipelineResult := &pipeline.OCRPDFResult{
		Filename:   result.Filename,
		TotalPages: result.TotalPages,
		Processing: struct {
			ExtractionNs int64 `json:"extraction_ns"`
			TotalNs      int64 `json:"total_ns"`
		}{
			ExtractionNs: result.Processing.ExtractionTimeMs * 1000000, // Convert ms to ns
			TotalNs:      result.Processing.TotalTimeMs * 1000000,      // Convert ms to ns
		},
	}

	// Convert pages
	for _, page := range result.Pages {
		pipelinePage := pipeline.OCRPDFPageResult{
			PageNumber: page.PageNumber,
			Width:      page.Width,
			Height:     page.Height,
			Processing: struct {
				TotalNs int64 `json:"total_ns"`
			}{
				TotalNs: page.Processing.TotalTimeMs * 1000000, // Convert ms to ns
			},
		}

		// Convert images from each page
		for _, img := range page.Images {
			pipelineImage := pipeline.OCRPDFImageResult{
				ImageIndex: img.ImageIndex,
				Width:      img.Width,
				Height:     img.Height,
				Confidence: img.Confidence,
			}

			// Convert OCR regions to pipeline format
			if len(img.OCRRegions) > 0 {
				// Use enriched OCR regions if available
				for _, region := range img.OCRRegions {
					pipelineRegion := pipeline.OCRRegionResult{
						Box: struct{ X, Y, W, H int }{
							X: region.Box.X,
							Y: region.Box.Y,
							W: region.Box.W,
							H: region.Box.H,
						},
						DetConfidence: region.DetConfidence,
						Text:          region.Text,
						RecConfidence: region.RecConfidence,
						Language:      region.Language,
					}

					// Convert polygon points
					for _, point := range region.Polygon {
						pipelineRegion.Polygon = append(pipelineRegion.Polygon, struct{ X, Y float64 }{
							X: float64(point.X),
							Y: float64(point.Y),
						})
					}

					pipelineImage.Regions = append(pipelineImage.Regions, pipelineRegion)
				}
			} else {
				// Fall back to basic detected regions
				for _, region := range img.Regions {
					pipelineRegion := pipeline.OCRRegionResult{
						Box: struct{ X, Y, W, H int }{
							X: int(region.Box.MinX),
							Y: int(region.Box.MinY),
							W: int(region.Box.MaxX - region.Box.MinX),
							H: int(region.Box.MaxY - region.Box.MinY),
						},
						DetConfidence: region.Confidence,
						Text:          "", // No text available from detection only
						RecConfidence: 0,  // No recognition performed
					}

					// Convert polygon points if available
					for _, point := range region.Polygon {
						pipelineRegion.Polygon = append(pipelineRegion.Polygon, struct{ X, Y float64 }{
							X: point.X,
							Y: point.Y,
						})
					}

					pipelineImage.Regions = append(pipelineImage.Regions, pipelineRegion)
				}
			}

			pipelinePage.Images = append(pipelinePage.Images, pipelineImage)
		}

		pipelineResult.Pages = append(pipelineResult.Pages, pipelinePage)
	}

	return pipelineResult
}
