package server

import (
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"os"
	"strings"
	"time"

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

	// Get pipeline for this request
	pipeline, err := s.getPipelineForRequest(reqConfig)
	if err != nil {
		s.writeErrorResponse(w, fmt.Sprintf("Failed to create pipeline: %v", err), http.StatusInternalServerError)
		ocrRequestsTotal.WithLabelValues("pdf", "error").Inc()
		return
	}

	// Run full OCR pipeline on PDF with timing
	start := time.Now()
	res, err := pipeline.ProcessPDF(header.Filename, pageRange)
	duration := time.Since(start)

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

	return file, header, pageRange, reqConfig, nil
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
