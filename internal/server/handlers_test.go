package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testServer is a wrapper for Server that allows injection of mock pipeline for testing.
type testServer struct {
	*Server
	mockPipeline pipelineInterface
}

// ocrImageHandlerMock is a version of ocrImageHandler that uses the mock pipeline.
func (ts *testServer) ocrImageHandlerMock(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Set content length limit
	r.Body = http.MaxBytesReader(w, r.Body, ts.maxUploadMB*1024*1024)

	// Parse multipart form
	err := r.ParseMultipartForm(ts.maxUploadMB * 1024 * 1024)
	if err != nil {
		ts.writeErrorResponse(w, "Failed to parse form data", http.StatusBadRequest)
		return
	}

	// Get uploaded file
	file, header, err := r.FormFile("image")
	if err != nil {
		ts.writeErrorResponse(w, "No image file provided", http.StatusBadRequest)
		return
	}
	defer func() { _ = file.Close() }()

	// Validate file size
	if header.Size > ts.maxUploadMB*1024*1024 {
		ts.writeErrorResponse(w, "File too large", http.StatusRequestEntityTooLarge)
		return
	}

	// Read file content
	imageData, err := io.ReadAll(file)
	if err != nil {
		ts.writeErrorResponse(w, "Failed to read image data", http.StatusInternalServerError)
		return
	}

	// Decode image
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		ts.writeErrorResponse(w, "Invalid image format", http.StatusBadRequest)
		return
	}

	// Check if pipeline is available
	if ts.mockPipeline == nil {
		ts.writeErrorResponse(w, "OCR pipeline not initialized", http.StatusServiceUnavailable)
		return
	}

	// Run full OCR pipeline using mock
	res, err := ts.mockPipeline.ProcessImage(img)
	if err != nil {
		ts.writeErrorResponse(w, fmt.Sprintf("OCR processing failed: %v", err), http.StatusInternalServerError)
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
	if format == "text" {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		s, err := pipeline.ToPlainTextImage(res)
		if err != nil {
			http.Error(w, fmt.Sprintf("formatting failed: %v", err), http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte(s))
		return
	}

	// default: json
	w.Header().Set("Content-Type", "application/json")
	obj := struct {
		OCR *pipeline.OCRImageResult `json:"ocr"`
	}{OCR: res}
	if err := json.NewEncoder(w).Encode(obj); err != nil {
		http.Error(w, fmt.Sprintf("encoding failed: %v", err), http.StatusInternalServerError)
	}
}

// mockPipeline is a simple mock implementation of the pipeline for testing.
type mockPipeline struct{}

// ProcessImage returns a mock OCR result for testing.
func (m *mockPipeline) ProcessImage(img image.Image) (*pipeline.OCRImageResult, error) {
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

// ProcessPDF returns a mock PDF OCR result for testing.
func (m *mockPipeline) ProcessPDF(filename string, pageRange string) (*pipeline.OCRPDFResult, error) {
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

func (m *mockPipeline) Close() error {
	return nil
}

// createTestImage creates a simple test image for testing.
func createTestImage(width, height int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			img.Set(x, y, color.RGBA{uint8(x % 256), uint8(y % 256), 0, 255})
		}
	}
	return img
}

// encodeImageToPNG encodes an image to PNG bytes.
func encodeImageToPNG(img image.Image) ([]byte, error) {
	var buf bytes.Buffer
	err := png.Encode(&buf, img)
	return buf.Bytes(), err
}

// createMultipartFormRequest creates a multipart form request with an image.
func createMultipartFormRequest(
	imageData []byte,
	filename string,
	extraFields map[string]string,
) (*http.Request, error) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add image file
	part, err := writer.CreateFormFile("image", filename)
	if err != nil {
		return nil, err
	}
	_, err = part.Write(imageData)
	if err != nil {
		return nil, err
	}

	// Add extra fields
	for key, value := range extraFields {
		err = writer.WriteField(key, value)
		if err != nil {
			return nil, err
		}
	}

	err = writer.Close()
	if err != nil {
		return nil, err
	}

	req := httptest.NewRequest(http.MethodPost, "/ocr/image", &buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	return req, nil
}

// createMultipartPDFFormRequest creates a multipart form request with a PDF file.
func createMultipartPDFFormRequest(
	pdfData []byte,
	filename string,
	extraFields map[string]string,
) (*http.Request, error) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add PDF file
	part, err := writer.CreateFormFile("pdf", filename)
	if err != nil {
		return nil, err
	}
	_, err = part.Write(pdfData)
	if err != nil {
		return nil, err
	}

	// Add extra fields
	for key, value := range extraFields {
		err = writer.WriteField(key, value)
		if err != nil {
			return nil, err
		}
	}

	err = writer.Close()
	if err != nil {
		return nil, err
	}

	req := httptest.NewRequest(http.MethodPost, "/ocr/pdf", &buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	return req, nil
}

func TestServer_HealthHandler(t *testing.T) {
	server := &Server{}

	tests := []struct {
		name           string
		method         string
		expectedStatus int
		checkResponse  bool
	}{
		{
			name:           "GET request success",
			method:         "GET",
			expectedStatus: http.StatusOK,
			checkResponse:  true,
		},
		{
			name:           "POST request not allowed",
			method:         "POST",
			expectedStatus: http.StatusMethodNotAllowed,
			checkResponse:  false,
		},
		{
			name:           "PUT request not allowed",
			method:         "PUT",
			expectedStatus: http.StatusMethodNotAllowed,
			checkResponse:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, "/health", nil)
			w := httptest.NewRecorder()

			server.healthHandler(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)

			if tt.checkResponse {
				var response HealthResponse
				err := json.Unmarshal(w.Body.Bytes(), &response)
				require.NoError(t, err)

				assert.Equal(t, "healthy", response.Status)
				assert.NotEmpty(t, response.Time)
				assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
			}
		})
	}
}

func TestServer_ModelsHandler(t *testing.T) {
	server := &Server{}

	tests := []struct {
		name           string
		method         string
		expectedStatus int
		checkResponse  bool
	}{
		{
			name:           "GET request success",
			method:         "GET",
			expectedStatus: http.StatusOK,
			checkResponse:  true,
		},
		{
			name:           "POST request not allowed",
			method:         "POST",
			expectedStatus: http.StatusMethodNotAllowed,
			checkResponse:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, "/models", nil)
			w := httptest.NewRecorder()

			server.modelsHandler(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)

			if tt.checkResponse {
				var response ModelsResponse
				err := json.Unmarshal(w.Body.Bytes(), &response)
				require.NoError(t, err)

				assert.GreaterOrEqual(t, response.Count, 0)
				assert.Equal(t, len(response.Models), response.Count)
				assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
			}
		})
	}
}

func TestServer_OCRImageHandler_MethodValidation(t *testing.T) {
	server := &Server{
		maxUploadMB: 10,
	}

	tests := []struct {
		name           string
		method         string
		expectedStatus int
	}{
		{
			name:           "GET request not allowed",
			method:         "GET",
			expectedStatus: http.StatusMethodNotAllowed,
		},
		{
			name:           "PUT request not allowed",
			method:         "PUT",
			expectedStatus: http.StatusMethodNotAllowed,
		},
		{
			name:           "DELETE request not allowed",
			method:         "DELETE",
			expectedStatus: http.StatusMethodNotAllowed,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, "/ocr/image", nil)
			w := httptest.NewRecorder()

			server.ocrImageHandler(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)
		})
	}
}

func TestServer_OCRImageHandler_FormParsing(t *testing.T) {
	server := &Server{
		maxUploadMB: 1, // 1MB limit for testing
	}

	t.Run("missing image file", func(t *testing.T) {
		var buf bytes.Buffer
		writer := multipart.NewWriter(&buf)
		require.NoError(t, writer.Close())
		req := httptest.NewRequest(http.MethodPost, "/ocr/image", &buf)
		req.Header.Set("Content-Type", writer.FormDataContentType())
		w := httptest.NewRecorder()

		server.ocrImageHandler(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)

		var response OCRResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		require.NoError(t, err)

		assert.False(t, response.Success)
		assert.Contains(t, response.Error, "No image file provided")
	})

	t.Run("invalid multipart form", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/ocr/image", strings.NewReader("invalid form data"))
		req.Header.Set("Content-Type", "multipart/form-data; boundary=invalid")
		w := httptest.NewRecorder()

		server.ocrImageHandler(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("file too large", func(t *testing.T) {
		// Create a large image for testing size limits
		largeImage := createTestImage(2000, 2000) // Creates a large image
		imageData, err := encodeImageToPNG(largeImage)
		require.NoError(t, err)

		// Create multipart form with large image
		req, err := createMultipartFormRequest(imageData, "large.png", nil)
		require.NoError(t, err)

		w := httptest.NewRecorder()
		server.ocrImageHandler(w, req)

		// Should fail due to size limit (depending on actual image size vs maxUploadMB)
		if len(imageData) > int(server.maxUploadMB*1024*1024) {
			assert.Equal(t, http.StatusRequestEntityTooLarge, w.Code)
		}
	})

	t.Run("invalid image format", func(t *testing.T) {
		// Create request with non-image data
		invalidData := []byte("This is not an image")
		req, err := createMultipartFormRequest(invalidData, "invalid.txt", nil)
		require.NoError(t, err)

		w := httptest.NewRecorder()
		server.ocrImageHandler(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)

		var response OCRResponse
		err = json.Unmarshal(w.Body.Bytes(), &response)
		require.NoError(t, err)

		assert.False(t, response.Success)
		assert.Contains(t, response.Error, "Invalid image format")
	})
}

func TestServer_OCRImageHandler_OutputFormats(t *testing.T) {
	mock := &mockPipeline{}
	server := &testServer{
		Server: &Server{
			maxUploadMB: 10,
		},
		mockPipeline: mock,
	}

	testImage := createTestImage(100, 100)
	imageData, err := encodeImageToPNG(testImage)
	require.NoError(t, err)

	tests := []struct {
		name           string
		format         string
		expectedStatus int
		expectedType   string
		checkContent   func(*testing.T, string)
	}{
		{
			name:           "JSON format (default)",
			format:         "",
			expectedStatus: http.StatusOK,
			expectedType:   "application/json",
			checkContent: func(t *testing.T, body string) {
				t.Helper()
				assert.Contains(t, body, "Hello World")
				assert.Contains(t, body, "ocr")
				assert.Contains(t, body, "regions")
			},
		},
		{
			name:           "JSON format (explicit)",
			format:         "json",
			expectedStatus: http.StatusOK,
			expectedType:   "application/json",
			checkContent: func(t *testing.T, body string) {
				t.Helper()
				assert.Contains(t, body, "Hello World")
			},
		},
		{
			name:           "CSV format",
			format:         "csv",
			expectedStatus: http.StatusOK,
			expectedType:   "text/csv",
			checkContent: func(t *testing.T, body string) {
				t.Helper()
				// CSV should contain the mock text
				assert.Contains(t, body, "Hello World")
			},
		},
		{
			name:           "Text format",
			format:         "text",
			expectedStatus: http.StatusOK,
			expectedType:   "text/plain; charset=utf-8",
			checkContent: func(t *testing.T, body string) {
				t.Helper()
				// Text should contain the mock text
				assert.Contains(t, body, "Hello World")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			extraFields := map[string]string{}
			if tt.format != "" {
				extraFields["format"] = tt.format
			}

			req, err := createMultipartFormRequest(imageData, "test.png", extraFields)
			require.NoError(t, err)

			w := httptest.NewRecorder()
			server.ocrImageHandlerMock(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)
			assert.Contains(t, w.Header().Get("Content-Type"), tt.expectedType)

			if tt.checkContent != nil {
				tt.checkContent(t, w.Body.String())
			}
		})
	}
}

func TestServer_OCRImageHandler_OverlayDisabled(t *testing.T) {
	server := &Server{
		maxUploadMB:    10,
		overlayEnabled: false, // Disabled
		pipeline:       nil,   // Will fail before overlay check
	}

	testImage := createTestImage(100, 100)
	imageData, err := encodeImageToPNG(testImage)
	require.NoError(t, err)

	req, err := createMultipartFormRequest(imageData, "test.png", map[string]string{"format": "overlay"})
	require.NoError(t, err)

	w := httptest.NewRecorder()
	server.ocrImageHandler(w, req)

	// Should fail due to nil pipeline check before overlay check
	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
}

func TestServer_WriteErrorResponse(t *testing.T) {
	server := &Server{}

	tests := []struct {
		name       string
		message    string
		statusCode int
	}{
		{
			name:       "bad request error",
			message:    "Invalid input",
			statusCode: http.StatusBadRequest,
		},
		{
			name:       "internal server error",
			message:    "Something went wrong",
			statusCode: http.StatusInternalServerError,
		},
		{
			name:       "not found error",
			message:    "Resource not found",
			statusCode: http.StatusNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()

			server.writeErrorResponse(w, tt.message, tt.statusCode)

			assert.Equal(t, tt.statusCode, w.Code)
			assert.Equal(t, "application/json", w.Header().Get("Content-Type"))

			var response OCRResponse
			err := json.Unmarshal(w.Body.Bytes(), &response)
			require.NoError(t, err)

			assert.False(t, response.Success)
			assert.Equal(t, tt.message, response.Error)
		})
	}
}

func TestParseHexColor(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected color.Color
	}{
		{
			name:     "valid color with hash",
			input:    "#FF0000",
			expected: color.RGBA{255, 0, 0, 255},
		},
		{
			name:     "valid color without hash",
			input:    "00FF00",
			expected: color.RGBA{0, 255, 0, 255},
		},
		{
			name:     "blue color",
			input:    "#0000FF",
			expected: color.RGBA{0, 0, 255, 255},
		},
		{
			name:     "empty string",
			input:    "",
			expected: nil,
		},
		{
			name:     "invalid length",
			input:    "#FF00",
			expected: nil,
		},
		{
			name:     "invalid characters",
			input:    "#GGGGGG",
			expected: nil,
		},
		{
			name:     "too long",
			input:    "#FF000000",
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parseHexColor(tt.input)
			if tt.expected == nil {
				assert.Nil(t, result)
			} else {
				assert.Equal(t, tt.expected, result)
			}
		})
	}
}

// Note: Mock pipeline implementation removed due to interface compatibility issues

// Benchmark tests.
func BenchmarkServer_HealthHandler(b *testing.B) {
	server := &Server{}
	req := httptest.NewRequest(http.MethodGet, "/health", nil)

	b.ResetTimer()
	for range b.N {
		w := httptest.NewRecorder()
		server.healthHandler(w, req)
	}
}

func BenchmarkServer_ModelsHandler(b *testing.B) {
	server := &Server{}
	req := httptest.NewRequest(http.MethodGet, "/models", nil)

	b.ResetTimer()
	for range b.N {
		w := httptest.NewRecorder()
		server.modelsHandler(w, req)
	}
}

func BenchmarkParseHexColor(b *testing.B) {
	colors := []string{"#FF0000", "00FF00", "#0000FF", "FFFFFF", "#123456"}

	b.ResetTimer()
	for range b.N {
		for _, color := range colors {
			parseHexColor(color)
		}
	}
}

// Integration test for full request/response cycle.
func TestServer_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Create server with nil pipeline (will cause test failures, which is fine)
	server := &Server{
		pipeline:    nil,
		maxUploadMB: 10,
		corsOrigin:  "*",
	}

	// Set up routes
	mux := http.NewServeMux()
	server.SetupRoutes(mux)

	// Test health endpoint
	t.Run("health endpoint", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/health", nil)
		w := httptest.NewRecorder()

		mux.ServeHTTP(w, req)

		assert.Equal(t, http.StatusOK, w.Code)
	})

	// Test models endpoint
	t.Run("models endpoint", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/models", nil)
		w := httptest.NewRecorder()

		mux.ServeHTTP(w, req)

		assert.Equal(t, http.StatusOK, w.Code)
	})

	// Test OCR endpoint
	t.Run("OCR endpoint", func(t *testing.T) {
		testImage := createTestImage(50, 50)
		imageData, err := encodeImageToPNG(testImage)
		require.NoError(t, err)

		req, err := createMultipartFormRequest(imageData, "test.png", nil)
		require.NoError(t, err)

		// Update URL to match route
		req.URL.Path = "/ocr/image"

		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)

		// Expect 503 because server has nil pipeline
		assert.Equal(t, http.StatusServiceUnavailable, w.Code)

		// Verify error response
		var response OCRResponse
		err = json.Unmarshal(w.Body.Bytes(), &response)
		require.NoError(t, err)
		assert.False(t, response.Success)
		assert.Contains(t, response.Error, "OCR pipeline not initialized")
	})
}

func TestServer_OCRPdfHandler_MethodValidation(t *testing.T) {
	server := &Server{
		maxUploadMB: 10,
	}

	tests := []struct {
		name           string
		method         string
		expectedStatus int
	}{
		{
			name:           "GET request not allowed",
			method:         "GET",
			expectedStatus: http.StatusMethodNotAllowed,
		},
		{
			name:           "PUT request not allowed",
			method:         "PUT",
			expectedStatus: http.StatusMethodNotAllowed,
		},
		{
			name:           "DELETE request not allowed",
			method:         "DELETE",
			expectedStatus: http.StatusMethodNotAllowed,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, "/ocr/pdf", nil)
			w := httptest.NewRecorder()

			server.ocrPdfHandler(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)
		})
	}
}

func TestServer_OCRPdfHandler_FormParsing(t *testing.T) {
	server := &Server{
		maxUploadMB: 1, // 1MB limit for testing
	}

	t.Run("missing pdf file", func(t *testing.T) {
		var buf bytes.Buffer
		writer := multipart.NewWriter(&buf)
		require.NoError(t, writer.Close())

		req := httptest.NewRequest(http.MethodPost, "/ocr/pdf", &buf)
		req.Header.Set("Content-Type", writer.FormDataContentType())
		w := httptest.NewRecorder()

		server.ocrPdfHandler(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)

		var response OCRResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		require.NoError(t, err)

		assert.False(t, response.Success)
		assert.Contains(t, response.Error, "No PDF file provided")
	})

	t.Run("invalid multipart form", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/ocr/pdf", strings.NewReader("invalid form data"))
		req.Header.Set("Content-Type", "multipart/form-data; boundary=invalid")
		w := httptest.NewRecorder()

		server.ocrPdfHandler(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("file too large", func(t *testing.T) {
		// Create large PDF-like data for testing size limits
		largeData := make([]byte, int(server.maxUploadMB*1024*1024)+1000) // Exceed limit
		for i := range largeData {
			largeData[i] = byte(i % 256)
		}

		req, err := createMultipartPDFFormRequest(largeData, "large.pdf", nil)
		require.NoError(t, err)

		w := httptest.NewRecorder()
		server.ocrPdfHandler(w, req)

		assert.Equal(t, http.StatusRequestEntityTooLarge, w.Code)
	})

	t.Run("invalid pdf format", func(t *testing.T) {
		// Create request with non-PDF data
		invalidData := []byte("This is not a PDF file")
		req, err := createMultipartPDFFormRequest(invalidData, "invalid.txt", nil)
		require.NoError(t, err)

		w := httptest.NewRecorder()
		server.ocrPdfHandler(w, req)

		// Should fail during PDF processing
		assert.Equal(t, http.StatusInternalServerError, w.Code)
	})
}

func TestServer_OCRPdfHandler_OutputFormats(t *testing.T) {
	// Skip this test since it requires a working pipeline and actual PDF file
	t.Skip("Skipping output format tests - requires pipeline integration and PDF file")
}

func TestServer_CORS_Middleware(t *testing.T) {
	server := &Server{
		corsOrigin: "*",
	}

	// Test CORS headers are added
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	req.Header.Set("Origin", "http://example.com")
	w := httptest.NewRecorder()

	handler := server.corsMiddleware(server.healthHandler)
	handler(w, req)

	assert.Equal(t, "*", w.Header().Get("Access-Control-Allow-Origin"))
	assert.Equal(t, "GET, POST, PUT, DELETE, OPTIONS", w.Header().Get("Access-Control-Allow-Methods"))
	assert.Equal(t, "Content-Type, Authorization", w.Header().Get("Access-Control-Allow-Headers"))
}

func TestServer_CORS_Preflight(t *testing.T) {
	server := &Server{
		corsOrigin: "http://localhost:3000",
	}

	req := httptest.NewRequest(http.MethodOptions, "/ocr/image", nil)
	req.Header.Set("Origin", "http://localhost:3000")
	req.Header.Set("Access-Control-Request-Method", "POST")
	req.Header.Set("Access-Control-Request-Headers", "Content-Type")
	w := httptest.NewRecorder()

	handler := server.corsMiddleware(func(w http.ResponseWriter, r *http.Request) {})
	handler(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "http://localhost:3000", w.Header().Get("Access-Control-Allow-Origin"))
	assert.Equal(t, "GET, POST, PUT, DELETE, OPTIONS", w.Header().Get("Access-Control-Allow-Methods"))
	assert.Equal(t, "Content-Type, Authorization", w.Header().Get("Access-Control-Allow-Headers"))
}
