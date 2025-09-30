package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/websocket"
)

// WebSocket upgrader with reasonable defaults.
var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		// Allow connections from any origin in development
		// In production, you should check against allowed origins
		return true
	},
}

// WebSocketMessage represents a message sent over WebSocket.
type WebSocketMessage struct {
	Type    string      `json:"type"`
	Payload interface{} `json:"payload,omitempty"`
}

// WebSocketOCRRequest represents an OCR request via WebSocket.
type WebSocketOCRRequest struct {
	Type     string                 `json:"type"` // "image" or "pdf"
	Image    []byte                 `json:"image,omitempty"`
	Filename string                 `json:"filename,omitempty"`
	Pages    string                 `json:"pages,omitempty"`
	Format   string                 `json:"format,omitempty"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

// WebSocketConnWriter is an interface for writing WebSocket messages.
type WebSocketConnWriter interface {
	WriteMessage(messageType int, data []byte) error
}

// WebSocketOCRResponse represents an OCR response via WebSocket.
type WebSocketOCRResponse struct {
	Type      string      `json:"type"`
	Status    string      `json:"status"` // "processing", "completed", "error"
	Progress  float64     `json:"progress,omitempty"`
	Result    interface{} `json:"result,omitempty"`
	Error     string      `json:"error,omitempty"`
	ErrorType string      `json:"error_type,omitempty"`
	RequestID string      `json:"request_id,omitempty"`
}

// ocrWebSocketHandler handles WebSocket connections for real-time OCR.
func (s *Server) ocrWebSocketHandler(w http.ResponseWriter, r *http.Request) {
	// Upgrade HTTP connection to WebSocket
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		slog.Error("Failed to upgrade connection to WebSocket", "error", err)
		return
	}
	defer func() {
		_ = conn.Close()
	}()

	// Increment active connections metric
	websocketConnections.Inc()
	defer websocketConnections.Dec()

	slog.Info("WebSocket connection established", "remote_addr", r.RemoteAddr)

	// Handle the WebSocket connection
	s.handleWebSocketConnection(conn)
}

// handleWebSocketConnection processes messages from a WebSocket connection.
func (s *Server) handleWebSocketConnection(conn *websocket.Conn) {
	// Set read deadline to prevent hanging connections
	_ = conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	conn.SetPongHandler(func(string) error {
		_ = conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	// Send ping messages to keep connection alive
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			if err := conn.WriteControl(websocket.PingMessage, []byte{}, time.Now().Add(10*time.Second)); err != nil {
				return
			}
		}
	}()

	for {
		// Read message from client
		messageType, data, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				slog.Error("WebSocket error", "error", err)
			}
			break
		}

		// Record message metric
		websocketMessagesTotal.WithLabelValues("received").Inc()

		if messageType == websocket.TextMessage {
			s.handleWebSocketMessage(conn, data)
		}
	}
}

// handleWebSocketMessage processes a WebSocket message.
func (s *Server) handleWebSocketMessage(conn *websocket.Conn, data []byte) {
	var req WebSocketOCRRequest
	if err := json.Unmarshal(data, &req); err != nil {
		s.sendWebSocketError(conn, "invalid_request", fmt.Sprintf("Failed to parse request: %v", err))
		return
	}

	// Generate a request ID for tracking
	requestID := strconv.FormatInt(time.Now().UnixNano(), 10)

	// Send processing start message
	s.sendWebSocketResponse(conn, WebSocketOCRResponse{
		Type:      "ocr_response",
		Status:    "processing",
		Progress:  0.0,
		RequestID: requestID,
	})

	// Process the request based on type
	switch req.Type {
	case "image":
		s.processWebSocketImage(conn, req, requestID)
	case "pdf":
		s.processWebSocketPDF(conn, req, requestID)
	default:
		s.sendWebSocketError(conn, "invalid_request", "Unsupported request type: "+req.Type)
	}
}

// processWebSocketImage processes an image OCR request via WebSocket.
func (s *Server) processWebSocketImage(conn *websocket.Conn, req WebSocketOCRRequest, requestID string) {
	if len(req.Image) == 0 {
		s.sendWebSocketError(conn, "invalid_request", "No image data provided")
		return
	}

	// Decode image
	img, _, err := image.Decode(bytes.NewReader(req.Image))
	if err != nil {
		s.sendWebSocketError(conn, "processing_error", fmt.Sprintf("Failed to decode image: %v", err))
		return
	}

	// Extract request configuration from options
	reqConfig := s.extractWebSocketConfig(req.Options)

	// Get pipeline for this request
	pipeline, err := s.getPipelineForRequest(reqConfig)
	if err != nil {
		s.sendWebSocketError(conn, "processing_error", fmt.Sprintf("Failed to create pipeline: %v", err))
		return
	}

	// Send progress update
	s.sendWebSocketResponse(conn, WebSocketOCRResponse{
		Type:      "ocr_response",
		Status:    "processing",
		Progress:  0.5,
		RequestID: requestID,
	})

	// Process image
	start := time.Now()
	res, err := pipeline.ProcessImage(img)
	duration := time.Since(start)

	if err != nil {
		ocrRequestsTotal.WithLabelValues("websocket_image", "error").Inc()
		s.sendWebSocketError(conn, "processing_error", fmt.Sprintf("OCR processing failed: %v", err))
		return
	}

	// Record metrics
	ocrRequestsTotal.WithLabelValues("websocket_image", "success").Inc()
	ocrProcessingDuration.WithLabelValues("websocket_image").Observe(duration.Seconds())

	var totalTextLength int
	for _, region := range res.Regions {
		totalTextLength += len(region.Text)
	}
	ocrTextLength.WithLabelValues("websocket_image").Observe(float64(totalTextLength))
	ocrRegionsDetected.WithLabelValues("websocket_image").Observe(float64(len(res.Regions)))

	// Send completion response
	s.sendWebSocketResponse(conn, WebSocketOCRResponse{
		Type:      "ocr_response",
		Status:    "completed",
		Progress:  1.0,
		Result:    res,
		RequestID: requestID,
	})
}

// processWebSocketPDF processes a PDF OCR request via WebSocket.
func (s *Server) processWebSocketPDF(conn *websocket.Conn, req WebSocketOCRRequest, requestID string) {
	if req.Filename == "" {
		s.sendWebSocketError(conn, "invalid_request", "No PDF filename provided")
		return
	}

	// Send progress update
	s.sendWebSocketResponse(conn, WebSocketOCRResponse{
		Type:      "ocr_response",
		Status:    "processing",
		Progress:  0.2,
		RequestID: requestID,
	})

	// Extract request configuration from options
	reqConfig := s.extractWebSocketConfig(req.Options)

	// Get pipeline for this request
	pipeline, err := s.getPipelineForRequest(reqConfig)
	if err != nil {
		s.sendWebSocketError(conn, "processing_error", fmt.Sprintf("Failed to create pipeline: %v", err))
		return
	}

	// Process PDF
	start := time.Now()
	res, err := pipeline.ProcessPDF(req.Filename, req.Pages)
	duration := time.Since(start)

	if err != nil {
		ocrRequestsTotal.WithLabelValues("websocket_pdf", "error").Inc()
		s.sendWebSocketError(conn, "processing_error", fmt.Sprintf("PDF OCR processing failed: %v", err))
		return
	}

	// Record metrics
	ocrRequestsTotal.WithLabelValues("websocket_pdf", "success").Inc()
	ocrProcessingDuration.WithLabelValues("websocket_pdf").Observe(duration.Seconds())

	var totalTextLength, totalRegions int
	for _, page := range res.Pages {
		for _, img := range page.Images {
			totalRegions += len(img.Regions)
			for _, region := range img.Regions {
				totalTextLength += len(region.Text)
			}
		}
	}
	ocrTextLength.WithLabelValues("websocket_pdf").Observe(float64(totalTextLength))
	ocrRegionsDetected.WithLabelValues("websocket_pdf").Observe(float64(totalRegions))

	// Send completion response
	s.sendWebSocketResponse(conn, WebSocketOCRResponse{
		Type:      "ocr_response",
		Status:    "completed",
		Progress:  1.0,
		Result:    res,
		RequestID: requestID,
	})
}

// sendWebSocketResponse sends a response message over WebSocket.
func (s *Server) sendWebSocketResponse(conn WebSocketConnWriter, response WebSocketOCRResponse) {
	data, err := json.Marshal(response)
	if err != nil {
		slog.Error("Failed to marshal WebSocket response", "error", err)
		return
	}

	if err := conn.WriteMessage(websocket.TextMessage, data); err != nil {
		slog.Error("Failed to send WebSocket message", "error", err)
		return
	}

	websocketMessagesTotal.WithLabelValues("sent").Inc()
}

// extractWebSocketConfig extracts RequestConfig from WebSocket options.
func (s *Server) extractWebSocketConfig(options map[string]interface{}) *RequestConfig {
	config := &RequestConfig{}

	if options == nil {
		return config
	}

	// Extract string values
	if val, ok := options["language"].(string); ok {
		config.Language = val
	}
	if val, ok := options["dict"].(string); ok {
		config.DictPath = val
	}
	if val, ok := options["det-model"].(string); ok {
		config.DetModel = val
	}
	if val, ok := options["rec-model"].(string); ok {
		config.RecModel = val
	}

	// Extract dict-langs as string or []string
	if val, ok := options["dict-langs"].(string); ok {
		config.DictLangs = strings.Split(val, ",")
		for i, lang := range config.DictLangs {
			config.DictLangs[i] = strings.TrimSpace(lang)
		}
	} else if val, ok := options["dict-langs"].([]interface{}); ok {
		config.DictLangs = make([]string, 0, len(val))
		for _, lang := range val {
			if langStr, ok := lang.(string); ok {
				config.DictLangs = append(config.DictLangs, strings.TrimSpace(langStr))
			}
		}
	}

	return config
}

// sendWebSocketError sends an error message over WebSocket.
func (s *Server) sendWebSocketError(conn WebSocketConnWriter, errorType, message string) {
	response := WebSocketOCRResponse{
		Type:      "error",
		Status:    "error",
		Error:     message,
		ErrorType: errorType,
	}

	data, err := json.Marshal(response)
	if err != nil {
		slog.Error("Failed to marshal WebSocket error response", "error", err)
		return
	}

	if err := conn.WriteMessage(websocket.TextMessage, data); err != nil {
		slog.Error("Failed to send WebSocket error message", "error", err)
		return
	}

	websocketMessagesTotal.WithLabelValues("sent").Inc()
}
