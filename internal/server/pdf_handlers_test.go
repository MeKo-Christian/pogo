package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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

func TestServer_ocrPdfHandler_Success(t *testing.T) {
	// Create a mock pipeline that returns a successful PDF result
	mockPipeline := &mockPipelineForTesting{
		processPDFResult: &pipeline.OCRPDFResult{
			Filename:   "test.pdf",
			TotalPages: 1,
			Pages: []pipeline.OCRPDFPageResult{
				{
					PageNumber: 1,
					Width:      100,
					Height:     200,
					Images: []pipeline.OCRPDFImageResult{
						{
							ImageIndex: 0,
							Width:      100,
							Height:     200,
							Regions: []pipeline.OCRRegionResult{
								{
									Text:          "Hello World",
									RecConfidence: 0.95,
									DetConfidence: 0.90,
									Box: struct{ X, Y, W, H int }{
										X: 10, Y: 10, W: 80, H: 20,
									},
								},
							},
							Confidence: 0.90,
						},
					},
					Processing: struct {
						TotalNs int64 `json:"total_ns"`
					}{TotalNs: 1000000},
				},
			},
			Processing: struct {
				ExtractionNs int64 `json:"extraction_ns"`
				TotalNs      int64 `json:"total_ns"`
			}{ExtractionNs: 500000, TotalNs: 1500000},
		},
		processPDFError: nil,
	}

	server := &Server{
		pipeline:    mockPipeline,
		maxUploadMB: 10,
	}

	// Create a multipart form with a mock PDF file
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// Add PDF file
	fileWriter, err := writer.CreateFormFile("pdf", "test.pdf")
	require.NoError(t, err)
	_, err = fileWriter.Write([]byte("mock pdf content"))
	require.NoError(t, err)

	// Add page range
	err = writer.WriteField("pages", "1")
	require.NoError(t, err)

	err = writer.Close()
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodPost, "/ocr/pdf", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	w := httptest.NewRecorder()

	server.ocrPdfHandler(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))

	var response map[string]interface{}
	err = json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)

	ocr, ok := response["ocr"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "test.pdf", ocr["filename"])
	assert.Equal(t, float64(1), ocr["total_pages"])
}

func TestServer_ocrPdfHandler_PageRange(t *testing.T) {
	// Create a mock pipeline that returns a successful PDF result with multiple pages
	mockPipeline := &mockPipelineForTesting{
		processPDFResult: &pipeline.OCRPDFResult{
			Filename:   "test.pdf",
			TotalPages: 2,
			Pages: []pipeline.OCRPDFPageResult{
				{
					PageNumber: 1,
					Width:      100,
					Height:     200,
					Images: []pipeline.OCRPDFImageResult{
						{
							ImageIndex: 0,
							Width:      100,
							Height:     200,
							Regions: []pipeline.OCRRegionResult{
								{
									Text:          "Page 1",
									RecConfidence: 0.95,
									DetConfidence: 0.90,
								},
							},
							Confidence: 0.90,
						},
					},
				},
				{
					PageNumber: 2,
					Width:      100,
					Height:     200,
					Images: []pipeline.OCRPDFImageResult{
						{
							ImageIndex: 0,
							Width:      100,
							Height:     200,
							Regions: []pipeline.OCRRegionResult{
								{
									Text:          "Page 2",
									RecConfidence: 0.95,
									DetConfidence: 0.90,
								},
							},
							Confidence: 0.90,
						},
					},
				},
			},
		},
		processPDFError: nil,
	}

	server := &Server{
		pipeline:    mockPipeline,
		maxUploadMB: 10,
	}

	// Create a multipart form with page range "1-2"
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	fileWriter, err := writer.CreateFormFile("pdf", "test.pdf")
	require.NoError(t, err)
	_, err = fileWriter.Write([]byte("mock pdf content"))
	require.NoError(t, err)

	err = writer.WriteField("pages", "1-2")
	require.NoError(t, err)

	err = writer.Close()
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodPost, "/ocr/pdf", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	w := httptest.NewRecorder()

	server.ocrPdfHandler(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	err = json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)

	ocr, ok := response["ocr"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, float64(2), ocr["total_pages"])
}

func TestServer_ocrPdfHandler_TextFormat(t *testing.T) {
	mockPipeline := &mockPipelineForTesting{
		processPDFResult: &pipeline.OCRPDFResult{
			Filename:   "test.pdf",
			TotalPages: 1,
			Pages: []pipeline.OCRPDFPageResult{
				{
					PageNumber: 1,
					Width:      100,
					Height:     200,
					Images: []pipeline.OCRPDFImageResult{
						{
							ImageIndex: 0,
							Width:      100,
							Height:     200,
							Regions: []pipeline.OCRRegionResult{
								{
									Text:          "Hello World",
									RecConfidence: 0.95,
									DetConfidence: 0.90,
									Box: struct{ X, Y, W, H int }{
										X: 10, Y: 10, W: 80, H: 20,
									},
								},
							},
							Confidence: 0.90,
						},
					},
				},
			},
		},
		processPDFError: nil,
	}

	server := &Server{
		pipeline:    mockPipeline,
		maxUploadMB: 10,
	}

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	fileWriter, err := writer.CreateFormFile("pdf", "test.pdf")
	require.NoError(t, err)
	_, err = fileWriter.Write([]byte("mock pdf content"))
	require.NoError(t, err)

	err = writer.WriteField("format", "text")
	require.NoError(t, err)

	err = writer.Close()
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodPost, "/ocr/pdf", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	w := httptest.NewRecorder()

	server.ocrPdfHandler(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "text/plain; charset=utf-8", w.Header().Get("Content-Type"))
	assert.Contains(t, w.Body.String(), "File: test.pdf")
	assert.Contains(t, w.Body.String(), "Hello World")
}

func TestServer_ocrPdfHandler_NoPipeline(t *testing.T) {
	server := &Server{
		pipeline:    nil,
		maxUploadMB: 10,
	}

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	fileWriter, err := writer.CreateFormFile("pdf", "test.pdf")
	require.NoError(t, err)
	_, err = fileWriter.Write([]byte("mock pdf content"))
	require.NoError(t, err)

	err = writer.Close()
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodPost, "/ocr/pdf", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	w := httptest.NewRecorder()

	server.ocrPdfHandler(w, req)

	assert.Equal(t, http.StatusInternalServerError, w.Code)

	var response map[string]interface{}
	err = json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)

	assert.Equal(t, false, response["success"])
	assert.Contains(t, response["error"], "OCR pipeline not initialized")
}

func TestServer_ocrPdfHandler_NoPDFFile(t *testing.T) {
	server := &Server{
		pipeline:    &mockPipelineForTesting{},
		maxUploadMB: 10,
	}

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	err := writer.Close()
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodPost, "/ocr/pdf", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	w := httptest.NewRecorder()

	server.ocrPdfHandler(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)

	var response map[string]interface{}
	err = json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)

	assert.Equal(t, false, response["success"])
	assert.Contains(t, response["error"], "No PDF file provided")
}

func TestServer_ocrPdfHandler_MethodNotAllowed(t *testing.T) {
	server := &Server{
		pipeline:    &mockPipelineForTesting{},
		maxUploadMB: 10,
	}

	req := httptest.NewRequest(http.MethodGet, "/ocr/pdf", nil)
	w := httptest.NewRecorder()

	server.ocrPdfHandler(w, req)

	assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
	assert.Equal(t, "Method not allowed\n", w.Body.String())
}

func TestServer_ocrPdfHandler_PipelineError(t *testing.T) {
	mockPipeline := &mockPipelineForTesting{
		processPDFResult: nil,
		processPDFError:  fmt.Errorf("PDF processing failed"),
	}

	server := &Server{
		pipeline:    mockPipeline,
		maxUploadMB: 10,
	}

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	fileWriter, err := writer.CreateFormFile("pdf", "test.pdf")
	require.NoError(t, err)
	_, err = fileWriter.Write([]byte("mock pdf content"))
	require.NoError(t, err)

	err = writer.Close()
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodPost, "/ocr/pdf", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	w := httptest.NewRecorder()

	server.ocrPdfHandler(w, req)

	assert.Equal(t, http.StatusInternalServerError, w.Code)

	var response map[string]interface{}
	err = json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)

	assert.Equal(t, false, response["success"])
	assert.Contains(t, response["error"], "OCR processing failed")
}
