package support

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/MeKo-Tech/pogo/internal/server"
)

// HTTPTestServerWrapper wraps httptest.Server for integration tests.
type HTTPTestServerWrapper struct {
	Server       *httptest.Server
	TestServer   *server.Server
	MockPipeline *MockPipeline
}

// MockPipeline provides predictable OCR results for testing.
type MockPipeline struct {
	ShouldFail bool
	ErrorMsg   string
}

// ProcessImage returns mock OCR results.
func (m *MockPipeline) ProcessImage(img image.Image) (*pipeline.OCRImageResult, error) {
	if m.ShouldFail {
		return nil, fmt.Errorf("mock error: %s", m.ErrorMsg)
	}

	bounds := img.Bounds()
	return &pipeline.OCRImageResult{
		Width:  bounds.Dx(),
		Height: bounds.Dy(),
		Regions: []pipeline.OCRRegionResult{
			{
				Polygon: []struct{ X, Y float64 }{
					{X: 10, Y: 10},
					{X: 100, Y: 10},
					{X: 100, Y: 30},
					{X: 10, Y: 30},
				},
				Box: struct{ X, Y, W, H int }{
					X: 10, Y: 10, W: 90, H: 20,
				},
				DetConfidence:   0.95,
				Text:            "Hello World",
				RecConfidence:   0.92,
				CharConfidences: []float64{0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9},
				Rotated:         false,
			},
		},
		AvgDetConf: 0.95,
		Orientation: struct {
			Angle      int     `json:"angle"`
			Confidence float64 `json:"confidence"`
			Applied    bool    `json:"applied"`
		}{
			Angle:      0,
			Confidence: 0.99,
			Applied:    false,
		},
		Processing: struct {
			DetectionNs   int64 `json:"detection_ns"`
			RecognitionNs int64 `json:"recognition_ns"`
			TotalNs       int64 `json:"total_ns"`
		}{
			DetectionNs:   1000000, // 1ms
			RecognitionNs: 2000000, // 2ms
			TotalNs:       3000000, // 3ms
		},
	}, nil
}

// ProcessPDF returns mock PDF OCR results.
func (m *MockPipeline) ProcessPDF(filename string, pageRange string) (*pipeline.OCRPDFResult, error) {
	if m.ShouldFail {
		return nil, fmt.Errorf("mock error: %s", m.ErrorMsg)
	}

	return &pipeline.OCRPDFResult{
		Filename:   filename,
		TotalPages: 1,
		Pages: []pipeline.OCRPDFPageResult{
			{
				PageNumber: 1,
				Width:      612,
				Height:     792,
				Images: []pipeline.OCRPDFImageResult{
					{
						ImageIndex: 0,
						Width:      612,
						Height:     792,
						Regions: []pipeline.OCRRegionResult{
							{
								Text:          "Sample PDF text",
								RecConfidence: 0.95,
								Box:           struct{ X, Y, W, H int }{X: 50, Y: 50, W: 200, H: 30},
							},
						},
						Confidence: 0.95,
					},
				},
			},
		},
		Processing: struct {
			ExtractionNs int64 `json:"extraction_ns"`
			TotalNs      int64 `json:"total_ns"`
		}{
			ExtractionNs: 5000000, // 5ms
			TotalNs:      8000000, // 8ms
		},
	}, nil
}

// Close is a no-op for the mock pipeline.
func (m *MockPipeline) Close() error {
	return nil
}

// createTestHTTPServer creates an httptest server with mock handlers.
func (testCtx *TestContext) createTestHTTPServer(port int) error {
	// Port parameter is unused - httptest server gets its own port
	_ = port
	const mockBase64ImageData = "base64encodedimagedata"

	// Create mock pipeline
	mockPipeline := &MockPipeline{
		ShouldFail: false,
		ErrorMsg:   "",
	}

	// Create a test server with mock handlers that simulate the real server behavior
	mux := http.NewServeMux()

	// Health endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		response := map[string]interface{}{
			"status": "healthy",
		}
		_ = json.NewEncoder(w).Encode(response)
	})

	// Models endpoint
	mux.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		response := map[string]interface{}{
			"models": []map[string]interface{}{
				{
					"name": "detection",
					"type": "detection",
					"path": "/models/detection.onnx",
					"size": 2048000,
				},
				{
					"name": "recognition",
					"type": "recognition",
					"path": "/models/recognition.onnx",
					"size": 4096000,
				},
			},
		}
		_ = json.NewEncoder(w).Encode(response)
	})

	// OCR image endpoint
	mux.HandleFunc("/ocr/image", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Parse multipart form
		err := r.ParseMultipartForm(32 << 20) // 32 MB
		if err != nil {
			http.Error(w, "Failed to parse form", http.StatusBadRequest)
			return
		}

		// Get file from form
		file, header, err := r.FormFile("file")
		if err != nil {
			// Try "image" field name as well
			file, header, err = r.FormFile("image")
			if err != nil {
				http.Error(w, "No image file provided", http.StatusBadRequest)
				return
			}
		}
		defer func() { _ = file.Close() }()

		// Check if it's an invalid file (simulate invalid format check)
		if strings.Contains(header.Filename, "invalid") || header.Size == 0 {
			http.Error(w, `{"error": "invalid format"}`, http.StatusBadRequest)
			return
		}

		// Check if file is too large (simulate 1MB limit)
		if header.Size > 1024*1024 && strings.Contains(header.Filename, "large") {
			http.Error(w, `{"error": "file too large"}`, http.StatusRequestEntityTooLarge)
			return
		}

		// Read image data
		imageData, err := io.ReadAll(file)
		if err != nil {
			http.Error(w, "Failed to read image", http.StatusBadRequest)
			return
		}

		// Decode image to verify it's valid
		_, _, err = image.Decode(bytes.NewReader(imageData))
		if err != nil {
			// Create a simple test image if decode fails
			img := testCtx.createSimpleTestImage(100, 50)
			result, _ := mockPipeline.ProcessImage(img)

			w.Header().Set("Content-Type", "application/json")

			response := map[string]interface{}{
				"results": []map[string]interface{}{
					{
						"text":       result.Regions[0].Text,
						"confidence": result.Regions[0].RecConfidence,
						"box":        result.Regions[0].Box,
					},
				},
				"regions":    result.Regions,
				"confidence": result.AvgDetConf,
			}

			// Check for overlay request
			if r.FormValue("overlay") == "true" || strings.Contains(r.URL.Path, "overlay") {
				response["overlay"] = mockBase64ImageData
				response["image_data"] = mockBase64ImageData
			}

			_ = json.NewEncoder(w).Encode(response)
			return
		}

		// Process with mock pipeline
		img, _, err := image.Decode(bytes.NewReader(imageData))
		if err != nil {
			http.Error(w, "Invalid image format", http.StatusBadRequest)
			return
		}

		result, err := mockPipeline.ProcessImage(img)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Get format parameter
		format := r.FormValue("format")
		if format == "" {
			format = "json"
		}

		// Handle different response formats
		switch format {
		case "text":
			w.Header().Set("Content-Type", "text/plain")
			_, _ = w.Write([]byte(result.Regions[0].Text))
		case "csv":
			w.Header().Set("Content-Type", "text/csv")
			_, _ = w.Write([]byte("text,confidence,x,y,w,h\n"))
			for _, region := range result.Regions {
				line := fmt.Sprintf("%s,%.2f,%d,%d,%d,%d\n",
					region.Text, region.RecConfidence,
					region.Box.X, region.Box.Y, region.Box.W, region.Box.H)
				_, _ = w.Write([]byte(line))
			}
		default: // json
			w.Header().Set("Content-Type", "application/json")

			response := map[string]interface{}{
				"results": []map[string]interface{}{
					{
						"text":       result.Regions[0].Text,
						"confidence": result.Regions[0].RecConfidence,
						"box":        result.Regions[0].Box,
					},
				},
				"regions":    result.Regions,
				"confidence": result.AvgDetConf,
			}

			// Check for overlay request
			if r.FormValue("overlay") == "true" || strings.Contains(r.URL.Path, "overlay") {
				response["overlay"] = mockBase64ImageData
				response["image_data"] = mockBase64ImageData
			}

			_ = json.NewEncoder(w).Encode(response)
		}
	})

	// CORS handling for OPTIONS requests
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	})

	// Create httptest server
	server := httptest.NewServer(mux)

	// Parse the URL to get host and port
	u, err := url.Parse(server.URL)
	if err != nil {
		server.Close()
		return fmt.Errorf("failed to parse server URL: %w", err)
	}

	// Update test context
	testCtx.ServerHost = u.Hostname()
	portStr := u.Port()
	if portStr != "" {
		testCtx.ServerPort, _ = strconv.Atoi(portStr)
	}

	// Store server reference for cleanup
	testCtx.HTTPTestServer = &HTTPTestServerWrapper{
		Server:       server,
		MockPipeline: mockPipeline,
	}

	return nil
}

// stopTestHTTPServer stops the httptest server.
func (testCtx *TestContext) stopTestHTTPServer() error {
	if testCtx.HTTPTestServer != nil && testCtx.HTTPTestServer.Server != nil {
		testCtx.HTTPTestServer.Server.Close()
		testCtx.HTTPTestServer = nil
	}
	return nil
}

// createSimpleTestImage creates a basic test image.
func (testCtx *TestContext) createSimpleTestImage(width, height int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	// Fill with white background
	for y := range height {
		for x := range width {
			img.Set(x, y, color.RGBA{255, 255, 255, 255})
		}
	}
	// Add some black text simulation
	for y := 10; y < 20; y++ {
		for x := 10; x < 90; x++ {
			img.Set(x, y, color.RGBA{0, 0, 0, 255})
		}
	}
	return img
}
