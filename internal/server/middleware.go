package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// responseWriter wraps http.ResponseWriter to capture status code.
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// corsMiddleware adds CORS headers to responses.
func (s *Server) corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", s.corsOrigin)
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		// Cache preflight results for a day to reduce OPTIONS traffic
		w.Header().Set("Access-Control-Max-Age", "86400")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		// Wrap response writer to capture status code
		rw := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		start := time.Now()
		next(rw, r)
		duration := time.Since(start)

		// Record metrics
		httpRequestsTotal.WithLabelValues(r.Method, r.URL.Path, http.StatusText(rw.statusCode)).Inc()
		httpRequestDuration.WithLabelValues(r.Method, r.URL.Path).Observe(duration.Seconds())
	}
}

// rateLimitMiddleware enforces rate limiting and quotas.
func (s *Server) rateLimitMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Skip rate limiting if not configured
		if s.rateLimiter == nil {
			next(w, r)
			return
		}

		// Get user identifier (IP address for now, could be extended to use API keys)
		userID := getClientIP(r)

		// Get content length for data quota checking
		var dataSize int64
		if r.ContentLength > 0 {
			dataSize = r.ContentLength
		}

		// Check rate limits
		if err := s.rateLimiter.CheckRateLimit(userID, dataSize); err != nil {
			// Record rate limit hit
			{
				var e *RateLimitError
				var e1 *QuotaExceededError
				switch {
				case errors.As(err, &e):
					rateLimitHits.WithLabelValues(e.Type).Inc()
				case errors.As(err, &e1):
					rateLimitHits.WithLabelValues(e1.Type).Inc()
				}
			}
			s.handleRateLimitError(w, err)
			return
		}

		next(w, r)
	}
}

// handleRateLimitError handles rate limit and quota errors.
func (s *Server) handleRateLimitError(w http.ResponseWriter, err error) {
	w.Header().Set("Content-Type", "application/json")

	{
		var e *RateLimitError
		var e1 *QuotaExceededError
		switch {
		case errors.As(err, &e):
			w.Header().Set("X-RateLimit-Type", e.Type)
			w.Header().Set("X-RateLimit-Limit", strconv.Itoa(e.Limit))
			w.Header().Set("Retry-After", fmt.Sprintf("%.0f", e.RetryAfter.Seconds()))
			w.WriteHeader(http.StatusTooManyRequests)
			response := map[string]interface{}{"error": "rate_limit_exceeded", "type": e.Type, "limit": e.Limit, "retry_after": e.RetryAfter.Seconds(), "message": e.Error()}
			if err := json.NewEncoder(w).Encode(response); err != nil {
				slog.Error("Failed to encode rate limit response", "error", err)
			}
		case errors.As(err, &e1):
			w.Header().Set("X-Quota-Type", e1.Type)
			w.Header().Set("X-Quota-Limit", strconv.FormatInt(e1.Limit, 10))
			w.Header().Set("X-Quota-Used", strconv.FormatInt(e1.Used, 10))
			w.Header().Set("X-Quota-Resets", e1.Resets.Format(http.TimeFormat))
			w.WriteHeader(http.StatusTooManyRequests)
			response := map[string]interface{}{"error": "quota_exceeded", "type": e1.Type, "limit": e1.Limit, "used": e1.Used, "resets": e1.Resets.Format(time.RFC3339), "message": e1.Error()}
			if err := json.NewEncoder(w).Encode(response); err != nil {
				slog.Error("Failed to encode quota exceeded response", "error", err)
			}
		default:
			w.WriteHeader(http.StatusInternalServerError)
			if err := json.NewEncoder(w).Encode(map[string]string{"error": "internal_error", "message": "Rate limiting check failed"}); err != nil {
				slog.Error("Failed to encode internal error response", "error", err)
			}
		}
	}
}

// getClientIP extracts the client IP address from the request.
func getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header first (for proxies/load balancers)
	xff := r.Header.Get("X-Forwarded-For")
	if xff != "" {
		// X-Forwarded-For can contain multiple IPs, take the first one
		if idx := strings.Index(xff, ","); idx > 0 {
			return strings.TrimSpace(xff[:idx])
		}
		return strings.TrimSpace(xff)
	}

	// Check X-Real-IP header
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return strings.TrimSpace(xri)
	}

	// Fall back to RemoteAddr
	ip, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return ip
}
