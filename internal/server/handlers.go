package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/prometheus/client_golang/prometheus/promhttp"
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

// metricsHandler exposes Prometheus metrics.
func (s *Server) metricsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Import promhttp inline to avoid import issues
	promhttp.Handler().ServeHTTP(w, r)
}
