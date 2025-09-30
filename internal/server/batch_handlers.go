package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"net/http"
	"os"
	"strings"
	"time"
)

// BatchOCRRequest represents a batch OCR request.
type BatchOCRRequest struct {
	Images []BatchImageRequest `json:"images,omitempty"`
	PDFs   []BatchPDFRequest   `json:"pdfs,omitempty"`
	Format string              `json:"format,omitempty"`
}

// BatchImageRequest represents a single image in a batch request.
type BatchImageRequest struct {
	Name    string                 `json:"name"`
	Data    []byte                 `json:"data"`
	Options map[string]interface{} `json:"options,omitempty"`
}

// BatchPDFRequest represents a single PDF in a batch request.
type BatchPDFRequest struct {
	Name    string                 `json:"name"`
	Data    []byte                 `json:"data"`
	Pages   string                 `json:"pages,omitempty"`
	Options map[string]interface{} `json:"options,omitempty"`
}

// BatchOCRResponse represents the response for batch OCR processing.
type BatchOCRResponse struct {
	Success bool                   `json:"success"`
	Results []BatchOCRResult       `json:"results,omitempty"`
	Error   string                 `json:"error,omitempty"`
	Summary BatchProcessingSummary `json:"summary"`
}

// BatchOCRResult represents a single result in batch processing.
type BatchOCRResult struct {
	Type     string      `json:"type"` // "image" or "pdf"
	Name     string      `json:"name"`
	Success  bool        `json:"success"`
	Result   interface{} `json:"result,omitempty"`
	Error    string      `json:"error,omitempty"`
	Duration float64     `json:"duration_seconds"`
}

// BatchProcessingSummary provides summary statistics for batch processing.
type BatchProcessingSummary struct {
	TotalItems    int     `json:"total_items"`
	Successful    int     `json:"successful"`
	Failed        int     `json:"failed"`
	TotalDuration float64 `json:"total_duration_seconds"`
	AvgItemTime   float64 `json:"avg_item_time_seconds"`
}

// ocrBatchHandler processes batch OCR requests.
func (s *Server) ocrBatchHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse JSON request body
	var req BatchOCRRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeErrorResponse(w, fmt.Sprintf("Failed to parse JSON request: %v", err), http.StatusBadRequest)
		return
	}

	// Validate request
	if len(req.Images) == 0 && len(req.PDFs) == 0 {
		s.writeErrorResponse(w, "No images or PDFs provided in batch request", http.StatusBadRequest)
		return
	}

	totalItems := len(req.Images) + len(req.PDFs)
	if totalItems > 10 { // Limit batch size
		s.writeErrorResponse(w, "Batch size too large (maximum 10 items)", http.StatusBadRequest)
		return
	}

	// Check if pipeline is available
	if s.pipeline == nil {
		s.writeErrorResponse(w, "OCR pipeline not initialized", http.StatusServiceUnavailable)
		return
	}

	// Process batch
	start := time.Now()
	results, summary := s.processBatchRequest(req)
	totalDuration := time.Since(start)

	summary.TotalDuration = totalDuration.Seconds()
	if summary.TotalItems > 0 {
		summary.AvgItemTime = summary.TotalDuration / float64(summary.TotalItems)
	}

	// Record batch metrics
	ocrRequestsTotal.WithLabelValues("batch", "success").Inc()
	ocrProcessingDuration.WithLabelValues("batch").Observe(totalDuration.Seconds())

	response := BatchOCRResponse{
		Success: summary.Failed == 0,
		Results: results,
		Summary: summary,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		fmt.Fprintf(os.Stderr, "Error encoding batch OCR response: %v\n", err)
	}
}

// processBatchRequest processes all items in a batch request.
func (s *Server) processBatchRequest(req BatchOCRRequest) ([]BatchOCRResult, BatchProcessingSummary) {
	results := make([]BatchOCRResult, 0, len(req.Images)+len(req.PDFs))
	summary := BatchProcessingSummary{
		TotalItems: len(req.Images) + len(req.PDFs),
	}

	// Process images
	for _, imgReq := range req.Images {
		result := s.processBatchImage(imgReq)
		results = append(results, result)
		if result.Success {
			summary.Successful++
		} else {
			summary.Failed++
		}
	}

	// Process PDFs
	for _, pdfReq := range req.PDFs {
		result := s.processBatchPDF(pdfReq)
		results = append(results, result)
		if result.Success {
			summary.Successful++
		} else {
			summary.Failed++
		}
	}

	return results, summary
}

// processBatchImage processes a single image in a batch request.
func (s *Server) processBatchImage(req BatchImageRequest) BatchOCRResult {
	result := BatchOCRResult{
		Type: "image",
		Name: req.Name,
	}

	if len(req.Data) == 0 {
		result.Error = "No image data provided"
		return result
	}

	// Decode image
	img, _, err := image.Decode(bytes.NewReader(req.Data))
	if err != nil {
		result.Error = fmt.Sprintf("Failed to decode image: %v", err)
		return result
	}

	// Extract request configuration from options
	reqConfig := s.extractBatchConfig(req.Options)

	// Get pipeline for this request
	pipeline, err := s.getPipelineForRequest(reqConfig)
	if err != nil {
		result.Error = fmt.Sprintf("Failed to create pipeline: %v", err)
		return result
	}

	// Process image
	start := time.Now()
	ocrResult, err := pipeline.ProcessImage(img)
	duration := time.Since(start)

	result.Duration = duration.Seconds()

	if err != nil {
		result.Error = fmt.Sprintf("OCR processing failed: %v", err)
		return result
	}

	result.Success = true
	result.Result = ocrResult

	// Record individual metrics
	var totalTextLength int
	for _, region := range ocrResult.Regions {
		totalTextLength += len(region.Text)
	}
	ocrTextLength.WithLabelValues("batch_image").Observe(float64(totalTextLength))
	ocrRegionsDetected.WithLabelValues("batch_image").Observe(float64(len(ocrResult.Regions)))

	return result
}

// processBatchPDF processes a single PDF in a batch request.
func (s *Server) processBatchPDF(req BatchPDFRequest) BatchOCRResult {
	result := BatchOCRResult{
		Type: "pdf",
		Name: req.Name,
	}

	if len(req.Data) == 0 {
		result.Error = "No PDF data provided"
		return result
	}

	// Create temporary file for PDF processing
	tempFile, err := os.CreateTemp("", "batch_pdf_*.pdf")
	if err != nil {
		result.Error = fmt.Sprintf("Failed to create temporary file: %v", err)
		return result
	}
	defer func() {
		_ = tempFile.Close()
		_ = os.Remove(tempFile.Name())
	}()

	// Write PDF data to temp file
	if _, err := tempFile.Write(req.Data); err != nil {
		result.Error = fmt.Sprintf("Failed to write PDF data: %v", err)
		return result
	}

	// Extract request configuration from options
	reqConfig := s.extractBatchConfig(req.Options)

	// Get pipeline for this request
	pipeline, err := s.getPipelineForRequest(reqConfig)
	if err != nil {
		result.Error = fmt.Sprintf("Failed to create pipeline: %v", err)
		return result
	}

	// Process PDF
	start := time.Now()
	ocrResult, err := pipeline.ProcessPDF(tempFile.Name(), req.Pages)
	duration := time.Since(start)

	result.Duration = duration.Seconds()

	if err != nil {
		result.Error = fmt.Sprintf("PDF OCR processing failed: %v", err)
		return result
	}

	result.Success = true
	result.Result = ocrResult

	// Record individual metrics
	var totalTextLength, totalRegions int
	for _, page := range ocrResult.Pages {
		for _, img := range page.Images {
			totalRegions += len(img.Regions)
			for _, region := range img.Regions {
				totalTextLength += len(region.Text)
			}
		}
	}
	ocrTextLength.WithLabelValues("batch_pdf").Observe(float64(totalTextLength))
	ocrRegionsDetected.WithLabelValues("batch_pdf").Observe(float64(totalRegions))

	return result
}

// extractBatchConfig extracts RequestConfig from batch options.
func (s *Server) extractBatchConfig(options map[string]interface{}) *RequestConfig {
	config := &RequestConfig{}

	if options == nil {
		return config
	}

	// Extract string values
	stringFields := map[string]*string{
		"language":  &config.Language,
		"dict":      &config.DictPath,
		"det-model": &config.DetModel,
		"rec-model": &config.RecModel,
	}

	for key, field := range stringFields {
		if val, ok := options[key].(string); ok {
			*field = val
		}
	}

	// Extract dict-langs as string or []string
	s.extractDictLangs(options, config)

	// Validate the configuration (ignore validation errors in batch processing to avoid failing entire batch)
	// The individual item processing will handle validation errors appropriately
	if err := config.Validate(); err != nil {
		// For batch processing, we'll log the validation error but continue
		// Individual item validation will be handled in the processing functions
		fmt.Fprintf(os.Stderr, "Warning: invalid batch configuration: %v\n", err)
	}

	return config
}

// extractDictLangs extracts dictionary languages from batch options.
func (s *Server) extractDictLangs(options map[string]interface{}, config *RequestConfig) {
	if val, ok := options["dict-langs"].(string); ok {
		config.DictLangs = strings.Split(val, ",")
		for i, lang := range config.DictLangs {
			config.DictLangs[i] = strings.TrimSpace(lang)
		}
	} else if val, ok := options["dict-langs"].([]interface{}); ok {
		config.DictLangs = make([]string, len(val))
		for i, lang := range val {
			if langStr, ok := lang.(string); ok {
				config.DictLangs[i] = strings.TrimSpace(langStr)
			}
		}
	}
}
