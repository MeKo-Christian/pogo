package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/pipeline"
	_ "golang.org/x/image/bmp"
)

const (
	formatText = "text"
)

// healthHandler returns server health status.
func (s *Server) healthHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	response := HealthResponse{
		Status: "healthy",
		Time:   time.Now().UTC().Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		fmt.Fprintf(os.Stderr, "Error encoding health response: %v\n", err)
	}
}

// modelsHandler returns information about available models.
func (s *Server) modelsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Return available model information
	modelInfos := models.ListAvailableModels()
	modelList := make([]ModelInfo, len(modelInfos))
	for i, info := range modelInfos {
		modelList[i] = ModelInfo{
			Name:        info.Name,
			Path:        models.ResolveModelPath("", info.Type, info.Variant, info.Filename),
			Type:        info.Type,
			Description: info.Description,
		}
	}

	response := ModelsResponse{
		Models: modelList,
		Count:  len(modelList),
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		fmt.Fprintf(os.Stderr, "Error encoding models response: %v\n", err)
	}
}

// ocrImageHandler processes image OCR requests.
func (s *Server) ocrImageHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Set content length limit
	r.Body = http.MaxBytesReader(w, r.Body, s.maxUploadMB*1024*1024)

	// Parse multipart form
	err := r.ParseMultipartForm(s.maxUploadMB * 1024 * 1024)
	if err != nil {
		s.writeErrorResponse(w, "Failed to parse form data", http.StatusBadRequest)
		return
	}

	// Get uploaded file
	file, header, err := r.FormFile("image")
	if err != nil {
		s.writeErrorResponse(w, "No image file provided", http.StatusBadRequest)
		return
	}
	defer func() { _ = file.Close() }()

	// Validate file size
	if header.Size > s.maxUploadMB*1024*1024 {
		s.writeErrorResponse(w, "File too large", http.StatusRequestEntityTooLarge)
		return
	}

	// Read file content
	imageData, err := io.ReadAll(file)
	if err != nil {
		s.writeErrorResponse(w, "Failed to read image data", http.StatusInternalServerError)
		return
	}

	// Decode image
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		s.writeErrorResponse(w, "Invalid image format", http.StatusBadRequest)
		return
	}

	// Check if pipeline is available
	if s.pipeline == nil {
		s.writeErrorResponse(w, "OCR pipeline not initialized", http.StatusServiceUnavailable)
		return
	}

	// Run full OCR pipeline
	res, err := s.pipeline.ProcessImage(img)
	if err != nil {
		s.writeErrorResponse(w, fmt.Sprintf("OCR processing failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Determine output format: default json; allow 'format' in query or form
	format := r.FormValue("format")
	if format == "" {
		format = r.URL.Query().Get("format")
	}
	if format == "csv" {
		w.Header().Set("Content-Type", "text/csv")
		s, err := pipeline.ToCSVImage(res)
		if err != nil {
			http.Error(w, fmt.Sprintf("formatting failed: %v", err), http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte(s))
		return
	}
	if format == formatText {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		s, err := pipeline.ToPlainTextImage(res)
		if err != nil {
			http.Error(w, fmt.Sprintf("formatting failed: %v", err), http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte(s))
		return
	}

	// overlay image output
	if format == "overlay" || r.FormValue("overlay") == "1" {
		s.handleOverlayOutput(w, r, img, res)
		return
	}

	// default: json
	w.Header().Set("Content-Type", "application/json")
	obj := struct {
		OCR *pipeline.OCRImageResult `json:"ocr"`
	}{OCR: res}
	if err := json.NewEncoder(w).Encode(obj); err != nil {
		fmt.Fprintf(os.Stderr, "Error encoding OCR image response: %v\n", err)
	}
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

// ocrPdfHandler processes PDF OCR requests.
func (s *Server) ocrPdfHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Set content length limit
	r.Body = http.MaxBytesReader(w, r.Body, s.maxUploadMB*1024*1024)

	// Parse multipart form
	err := r.ParseMultipartForm(s.maxUploadMB * 1024 * 1024)
	if err != nil {
		// Distinguish body-too-large from generic parse error
		if strings.Contains(strings.ToLower(err.Error()), "body too large") ||
			strings.Contains(strings.ToLower(err.Error()), "request body too large") {
			s.writeErrorResponse(w, "File too large", http.StatusRequestEntityTooLarge)
		} else {
			s.writeErrorResponse(w, "Failed to parse form data", http.StatusBadRequest)
		}
		return
	}

	// Get uploaded file
	file, header, err := r.FormFile("pdf")
	if err != nil {
		s.writeErrorResponse(w, "No PDF file provided", http.StatusBadRequest)
		return
	}
	defer func() { _ = file.Close() }()

	// Validate file size
	if header.Size > s.maxUploadMB*1024*1024 {
		s.writeErrorResponse(w, "File too large", http.StatusRequestEntityTooLarge)
		return
	}

	// Get page range parameter
	pageRange := r.FormValue("pages")

	// Check if pipeline is available
	if s.pipeline == nil {
		// For PDF endpoint, treat missing pipeline as internal error to align with tests
		s.writeErrorResponse(w, "OCR pipeline not initialized", http.StatusInternalServerError)
		return
	}

	// Run full OCR pipeline on PDF
	res, err := s.pipeline.ProcessPDF(header.Filename, pageRange)
	if err != nil {
		s.writeErrorResponse(w, fmt.Sprintf("OCR processing failed: %v", err), http.StatusInternalServerError)
		return
	}

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

// writeErrorResponse writes a JSON error response.
func (s *Server) writeErrorResponse(w http.ResponseWriter, message string, statusCode int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	response := OCRResponse{
		Success: false,
		Error:   message,
	}

	if err := json.NewEncoder(w).Encode(response); err != nil {
		// Log error, but can't send another response
		fmt.Fprintf(os.Stderr, "Error writing error response: %v\n", err)
	}
}
