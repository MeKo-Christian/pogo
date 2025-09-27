package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestServer_CORSMiddleware(t *testing.T) {
	tests := []struct {
		name           string
		corsOrigin     string
		method         string
		expectedCORS   string
		expectedStatus int
		shouldCallNext bool
	}{
		{
			name:           "GET request with CORS headers",
			corsOrigin:     "*",
			method:         "GET",
			expectedCORS:   "*",
			expectedStatus: http.StatusOK,
			shouldCallNext: true,
		},
		{
			name:           "POST request with specific origin",
			corsOrigin:     "https://example.com",
			method:         "POST",
			expectedCORS:   "https://example.com",
			expectedStatus: http.StatusOK,
			shouldCallNext: true,
		},
		{
			name:           "OPTIONS request (preflight)",
			corsOrigin:     "*",
			method:         "OPTIONS",
			expectedCORS:   "*",
			expectedStatus: http.StatusOK,
			shouldCallNext: false,
		},
		{
			name:           "PUT request with CORS",
			corsOrigin:     "http://localhost:3000",
			method:         "PUT",
			expectedCORS:   "http://localhost:3000",
			expectedStatus: http.StatusOK,
			shouldCallNext: true,
		},
		{
			name:           "empty CORS origin",
			corsOrigin:     "",
			method:         "GET",
			expectedCORS:   "",
			expectedStatus: http.StatusOK,
			shouldCallNext: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := &Server{
				corsOrigin: tt.corsOrigin,
			}

			// Track if the next handler was called
			nextCalled := false
			nextHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				nextCalled = true
				w.WriteHeader(http.StatusOK)
			})

			// Create the CORS middleware
			corsHandler := server.corsMiddleware(nextHandler)

			// Create request and response recorder
			req := httptest.NewRequest(tt.method, "/test", nil)
			w := httptest.NewRecorder()

			// Execute the middleware
			corsHandler(w, req)

			// Check status code
			assert.Equal(t, tt.expectedStatus, w.Code)

			// Check CORS headers
			assert.Equal(t, tt.expectedCORS, w.Header().Get("Access-Control-Allow-Origin"))
			assert.Equal(t, "GET, POST, PUT, DELETE, OPTIONS", w.Header().Get("Access-Control-Allow-Methods"))
			assert.Equal(t, "Content-Type, Authorization", w.Header().Get("Access-Control-Allow-Headers"))

			// Check if next handler was called
			assert.Equal(t, tt.shouldCallNext, nextCalled)
		})
	}
}

func TestServer_CORSMiddleware_HeadersSet(t *testing.T) {
	server := &Server{
		corsOrigin: "https://myapp.com",
	}

	nextHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Next handler should see CORS headers already set
		assert.Equal(t, "https://myapp.com", w.Header().Get("Access-Control-Allow-Origin"))
		assert.Equal(t, "GET, POST, PUT, DELETE, OPTIONS", w.Header().Get("Access-Control-Allow-Methods"))
		assert.Equal(t, "Content-Type, Authorization", w.Header().Get("Access-Control-Allow-Headers"))
		w.WriteHeader(http.StatusOK)
	})

	corsHandler := server.corsMiddleware(nextHandler)

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	w := httptest.NewRecorder()

	corsHandler(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestServer_CORSMiddleware_OptionsOnly(t *testing.T) {
	server := &Server{
		corsOrigin: "*",
	}

	// Next handler should NOT be called for OPTIONS
	nextHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("Next handler should not be called for OPTIONS request")
	})

	corsHandler := server.corsMiddleware(nextHandler)

	req := httptest.NewRequest(http.MethodOptions, "/test", nil)
	w := httptest.NewRecorder()

	corsHandler(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	// Verify CORS headers are still set for OPTIONS
	assert.Equal(t, "*", w.Header().Get("Access-Control-Allow-Origin"))
	assert.Equal(t, "GET, POST, PUT, DELETE, OPTIONS", w.Header().Get("Access-Control-Allow-Methods"))
	assert.Equal(t, "Content-Type, Authorization", w.Header().Get("Access-Control-Allow-Headers"))
}

func TestServer_CORSMiddleware_ErrorInNext(t *testing.T) {
	server := &Server{
		corsOrigin: "*",
	}

	// Next handler returns an error status
	nextHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})

	corsHandler := server.corsMiddleware(nextHandler)

	req := httptest.NewRequest(http.MethodPost, "/test", nil)
	w := httptest.NewRecorder()

	corsHandler(w, req)

	// Even with error, CORS headers should still be present
	assert.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Equal(t, "*", w.Header().Get("Access-Control-Allow-Origin"))
}

func TestServer_CORSMiddleware_MultipleOrigins(t *testing.T) {
	// Test with multiple different origins
	origins := []string{
		"*",
		"https://example.com",
		"http://localhost:3000",
		"https://api.myapp.com",
		"",
	}

	for _, origin := range origins {
		t.Run("origin_"+origin, func(t *testing.T) {
			server := &Server{
				corsOrigin: origin,
			}

			nextHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
			})

			corsHandler := server.corsMiddleware(nextHandler)

			req := httptest.NewRequest(http.MethodGet, "/test", nil)
			w := httptest.NewRecorder()

			corsHandler(w, req)

			assert.Equal(t, origin, w.Header().Get("Access-Control-Allow-Origin"))
		})
	}
}

// Test middleware chaining.
func TestServer_CORSMiddleware_Chaining(t *testing.T) {
	server := &Server{
		corsOrigin: "https://test.com",
	}

	// Create a chain of middleware
	var callOrder []string

	finalHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callOrder = append(callOrder, "final")
		w.WriteHeader(http.StatusOK)
	})

	// Another middleware to test chaining
	testMiddleware := func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			callOrder = append(callOrder, "test")
			next(w, r)
		}
	}

	// Chain: CORS -> Test -> Final
	handler := server.corsMiddleware(testMiddleware(finalHandler))

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	w := httptest.NewRecorder()

	handler(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, []string{"test", "final"}, callOrder)
	assert.Equal(t, "https://test.com", w.Header().Get("Access-Control-Allow-Origin"))
}

// Benchmark the CORS middleware.
func BenchmarkServer_CORSMiddleware(b *testing.B) {
	server := &Server{
		corsOrigin: "*",
	}

	nextHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	corsHandler := server.corsMiddleware(nextHandler)
	req := httptest.NewRequest(http.MethodGet, "/test", nil)

	b.ResetTimer()
	for range b.N {
		w := httptest.NewRecorder()
		corsHandler(w, req)
	}
}

func BenchmarkServer_CORSMiddleware_OPTIONS(b *testing.B) {
	server := &Server{
		corsOrigin: "*",
	}

	nextHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	corsHandler := server.corsMiddleware(nextHandler)
	req := httptest.NewRequest(http.MethodOptions, "/test", nil)

	b.ResetTimer()
	for range b.N {
		w := httptest.NewRecorder()
		corsHandler(w, req)
	}
}
