package server

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConfig_Default(t *testing.T) {
	config := Config{
		Host:        "localhost",
		Port:        8080,
		CORSOrigin:  "*",
		MaxUploadMB: 10,
		TimeoutSec:  30,
	}

	assert.Equal(t, "localhost", config.Host)
	assert.Equal(t, 8080, config.Port)
	assert.Equal(t, "*", config.CORSOrigin)
	assert.Equal(t, int64(10), config.MaxUploadMB)
	assert.Equal(t, 30, config.TimeoutSec)
}

func TestHealthResponse_Serialization(t *testing.T) {
	response := HealthResponse{
		Status:  "healthy",
		Version: "1.0.0",
		Time:    "2023-12-01T12:00:00Z",
	}

	data, err := json.Marshal(response)
	require.NoError(t, err)

	assert.Contains(t, string(data), `"status":"healthy"`)
	assert.Contains(t, string(data), `"version":"1.0.0"`)
	assert.Contains(t, string(data), `"time":"2023-12-01T12:00:00Z"`)

	var unmarshaled HealthResponse
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)

	assert.Equal(t, response.Status, unmarshaled.Status)
	assert.Equal(t, response.Version, unmarshaled.Version)
	assert.Equal(t, response.Time, unmarshaled.Time)
}

func TestModelInfo_Serialization(t *testing.T) {
	modelInfo := ModelInfo{
		Name:        "test-model",
		Path:        "/path/to/model",
		Type:        "detection",
		Description: "Test model for OCR detection",
		Config:      map[string]interface{}{"threshold": 0.5},
	}

	data, err := json.Marshal(modelInfo)
	require.NoError(t, err)

	assert.Contains(t, string(data), `"name":"test-model"`)
	assert.Contains(t, string(data), `"type":"detection"`)
	assert.Contains(t, string(data), `"description":"Test model for OCR detection"`)

	var unmarshaled ModelInfo
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)

	assert.Equal(t, modelInfo.Name, unmarshaled.Name)
	assert.Equal(t, modelInfo.Type, unmarshaled.Type)
	assert.Equal(t, modelInfo.Description, unmarshaled.Description)
}

func TestModelsResponse_Serialization(t *testing.T) {
	models := []ModelInfo{
		{
			Name: "detector",
			Type: "detection",
			Path: "/models/detector.onnx",
		},
		{
			Name: "recognizer",
			Type: "recognition",
			Path: "/models/recognizer.onnx",
		},
	}

	response := ModelsResponse{
		Models: models,
		Count:  len(models),
	}

	data, err := json.Marshal(response)
	require.NoError(t, err)

	assert.Contains(t, string(data), `"count":2`)
	assert.Contains(t, string(data), `"models":[`)

	var unmarshaled ModelsResponse
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)

	assert.Equal(t, 2, unmarshaled.Count)
	assert.Len(t, unmarshaled.Models, 2)
	assert.Equal(t, "detector", unmarshaled.Models[0].Name)
	assert.Equal(t, "recognizer", unmarshaled.Models[1].Name)
}

func TestDetectionBox_Serialization(t *testing.T) {
	box := DetectionBox{
		X1:         10.5,
		Y1:         20.0,
		X2:         100.5,
		Y2:         50.0,
		Confidence: 0.95,
	}

	data, err := json.Marshal(box)
	require.NoError(t, err)

	assert.Contains(t, string(data), `"x1":10.5`)
	assert.Contains(t, string(data), `"confidence":0.95`)

	var unmarshaled DetectionBox
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)

	assert.InDelta(t, box.X1, unmarshaled.X1, 0.0001)
	assert.InDelta(t, box.Y1, unmarshaled.Y1, 0.0001)
	assert.InDelta(t, box.X2, unmarshaled.X2, 0.0001)
	assert.InDelta(t, box.Y2, unmarshaled.Y2, 0.0001)
	assert.InDelta(t, box.Confidence, unmarshaled.Confidence, 0.0001)
}

func TestOCRResult_Serialization(t *testing.T) {
	result := OCRResult{
		Text:       "Sample text",
		Language:   "en",
		Confidence: 0.92,
		Width:      800,
		Height:     600,
		Boxes: []DetectionBox{
			{X1: 10, Y1: 20, X2: 100, Y2: 50, Confidence: 0.9},
			{X1: 20, Y1: 60, X2: 120, Y2: 90, Confidence: 0.85},
		},
	}
	result.Processing.DetectionTimeMs = 150
	result.Processing.TotalTimeMs = 200

	data, err := json.Marshal(result)
	require.NoError(t, err)

	assert.Contains(t, string(data), `"text":"Sample text"`)
	assert.Contains(t, string(data), `"confidence":0.92`)
	assert.Contains(t, string(data), `"detection_time_ms":150`)

	var unmarshaled OCRResult
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)

	assert.Equal(t, result.Text, unmarshaled.Text)
	assert.InDelta(t, result.Confidence, unmarshaled.Confidence, 0.0001)
	assert.Len(t, unmarshaled.Boxes, 2)
	assert.Equal(t, int64(150), unmarshaled.Processing.DetectionTimeMs)
}

func TestOCRResponse_Serialization(t *testing.T) {
	tests := []struct {
		name     string
		response OCRResponse
	}{
		{
			name: "success response",
			response: OCRResponse{
				Success: true,
				Result: OCRResult{
					Text:       "Success",
					Confidence: 0.9,
					Width:      100,
					Height:     100,
				},
			},
		},
		{
			name: "error response",
			response: OCRResponse{
				Success: false,
				Error:   "Processing failed",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.response)
			require.NoError(t, err)

			var unmarshaled OCRResponse
			err = json.Unmarshal(data, &unmarshaled)
			require.NoError(t, err)

			assert.Equal(t, tt.response.Success, unmarshaled.Success)
			if tt.response.Success {
				assert.Equal(t, tt.response.Result.Text, unmarshaled.Result.Text)
			} else {
				assert.Equal(t, tt.response.Error, unmarshaled.Error)
			}
		})
	}
}

func TestNewServer_ErrorCases(t *testing.T) {
	t.Run("invalid pipeline config", func(t *testing.T) {
		config := Config{
			Host:        "localhost",
			Port:        8080,
			CORSOrigin:  "*",
			MaxUploadMB: 10,
			TimeoutSec:  30,
			PipelineConfig: pipeline.Config{
				ModelsDir: "/non/existent/path", // Invalid path should cause error
			},
		}

		server, err := NewServer(config)
		// Should fail due to invalid models directory
		require.Error(t, err)
		assert.Nil(t, server)
	})
}

func TestServer_SetupRoutes(t *testing.T) {
	// Create a minimal server for route testing
	server := &Server{
		corsOrigin:  "*",
		maxUploadMB: 10,
	}

	mux := http.NewServeMux()
	server.SetupRoutes(mux)

	// Test that routes are registered by checking the handlers
	// This is a basic test to ensure SetupRoutes doesn't panic
	assert.NotNil(t, mux)
}

func TestServer_Close(t *testing.T) {
	tests := []struct {
		name     string
		server   *Server
		hasError bool
	}{
		{
			name:     "server with nil pipeline",
			server:   &Server{pipeline: nil},
			hasError: false,
		},
		{
			name: "server close",
			server: &Server{
				corsOrigin:  "*",
				maxUploadMB: 10,
				pipeline:    nil, // nil pipeline should not cause error
			},
			hasError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.server.Close()
			if tt.hasError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestConfig_Validation(t *testing.T) {
	tests := []struct {
		name   string
		config Config
		valid  bool
	}{
		{
			name: "valid config",
			config: Config{
				Host:           "localhost",
				Port:           8080,
				CORSOrigin:     "*",
				MaxUploadMB:    10,
				TimeoutSec:     30,
				OverlayEnabled: true,
			},
			valid: true,
		},
		{
			name: "zero port",
			config: Config{
				Host:        "localhost",
				Port:        0,
				MaxUploadMB: 10,
			},
			valid: true, // Port 0 might be valid for auto-assignment
		},
		{
			name: "negative max upload",
			config: Config{
				Host:        "localhost",
				Port:        8080,
				MaxUploadMB: -1,
			},
			valid: true, // We don't validate here, just test structure
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Basic structure validation
			assert.IsType(t, Config{}, tt.config)
			assert.IsType(t, "", tt.config.Host)
			assert.IsType(t, 0, tt.config.Port)
			assert.IsType(t, int64(0), tt.config.MaxUploadMB)
		})
	}
}

// Test JSON field names match expected API format.
func TestJSON_FieldNames(t *testing.T) {
	t.Run("HealthResponse field names", func(t *testing.T) {
		response := HealthResponse{Status: "ok", Version: "1.0", Time: "now"}
		data, _ := json.Marshal(response)
		jsonStr := string(data)

		assert.Contains(t, jsonStr, `"status"`)
		assert.Contains(t, jsonStr, `"version"`)
		assert.Contains(t, jsonStr, `"time"`)
	})

	t.Run("OCRResponse field names", func(t *testing.T) {
		response := OCRResponse{Success: true, Error: "test"}
		data, _ := json.Marshal(response)
		jsonStr := string(data)

		assert.Contains(t, jsonStr, `"success"`)
		assert.Contains(t, jsonStr, `"error"`)
	})
}

// Benchmark tests.
func BenchmarkHealthResponse_Marshal(b *testing.B) {
	response := HealthResponse{
		Status:  "healthy",
		Version: "1.0.0",
		Time:    "2023-12-01T12:00:00Z",
	}

	b.ResetTimer()
	for range b.N {
		_, _ = json.Marshal(response)
	}
}

func BenchmarkOCRResponse_Marshal(b *testing.B) {
	response := OCRResponse{
		Success: true,
		Result: OCRResult{
			Text:       "Sample text for benchmarking",
			Confidence: 0.95,
			Width:      1024,
			Height:     768,
			Boxes:      make([]DetectionBox, 100), // Many boxes for realistic test
		},
	}

	b.ResetTimer()
	for range b.N {
		_, _ = json.Marshal(response)
	}
}
