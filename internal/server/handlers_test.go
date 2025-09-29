package server

import (
	"encoding/json"
	"image/color"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
