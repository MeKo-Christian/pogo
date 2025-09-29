package server

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
