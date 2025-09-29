package server

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// HTTP request metrics
	httpRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "pogo_http_requests_total",
			Help: "Total number of HTTP requests",
		},
		[]string{"method", "endpoint", "status"},
	)

	httpRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "pogo_http_request_duration_seconds",
			Help:    "HTTP request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "endpoint"},
	)

	// OCR processing metrics
	ocrRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "pogo_ocr_requests_total",
			Help: "Total number of OCR requests",
		},
		[]string{"type", "status"}, // type: image, pdf, batch
	)

	ocrProcessingDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "pogo_ocr_processing_duration_seconds",
			Help:    "OCR processing duration in seconds",
			Buckets: []float64{.1, .25, .5, 1, 2.5, 5, 10, 25, 50, 100},
		},
		[]string{"type"}, // type: image, pdf, batch
	)

	ocrTextLength = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "pogo_ocr_text_length",
			Help:    "Length of extracted text",
			Buckets: []float64{0, 10, 50, 100, 500, 1000, 5000, 10000, 50000},
		},
		[]string{"type"},
	)

	ocrRegionsDetected = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "pogo_ocr_regions_detected",
			Help:    "Number of text regions detected",
			Buckets: []float64{0, 1, 5, 10, 25, 50, 100, 250, 500},
		},
		[]string{"type"},
	)

	// Rate limiting metrics
	rateLimitHits = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "pogo_rate_limit_hits_total",
			Help: "Total number of rate limit hits",
		},
		[]string{"type"}, // type: requests_per_minute, requests_per_hour, max_requests_per_day, max_data_per_day
	)

	// File upload metrics
	uploadSizeBytes = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "pogo_upload_size_bytes",
			Help:    "Size of uploaded files in bytes",
			Buckets: []float64{1024, 10 * 1024, 100 * 1024, 1024 * 1024, 10 * 1024 * 1024, 50 * 1024 * 1024, 100 * 1024 * 1024},
		},
	)

	// WebSocket metrics
	websocketConnections = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "pogo_websocket_active_connections",
			Help: "Number of active WebSocket connections",
		},
	)

	websocketMessagesTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "pogo_websocket_messages_total",
			Help: "Total number of WebSocket messages",
		},
		[]string{"direction"}, // direction: sent, received
	)
)
