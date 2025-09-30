package server

import (
	"fmt"
	"sync"
	"time"
)

// RateLimiter manages request rate limiting and quotas.
type RateLimiter struct {
	mu sync.RWMutex

	// Request rate limiting
	requestsPerMinute int
	requestsPerHour   int

	// User quotas
	maxRequestsPerDay int
	maxDataPerDay     int64 // in bytes

	// Storage for tracking usage
	userRequests map[string]*UserUsage
}

// UserUsage tracks usage for a specific user/IP.
type UserUsage struct {
	// Request counts
	requestsLastMinute int
	requestsLastHour   int
	requestsToday      int

	// Data usage
	dataToday int64 // bytes uploaded today

	// Timestamps
	lastRequestTime time.Time
	dayStartTime    time.Time
}

// NewRateLimiter creates a new rate limiter with the given limits.
func NewRateLimiter(requestsPerMinute, requestsPerHour, maxRequestsPerDay int, maxDataPerDay int64) *RateLimiter {
	return &RateLimiter{
		requestsPerMinute: requestsPerMinute,
		requestsPerHour:   requestsPerHour,
		maxRequestsPerDay: maxRequestsPerDay,
		maxDataPerDay:     maxDataPerDay,
		userRequests:      make(map[string]*UserUsage),
	}
}

// CheckRateLimit checks if a request from the given user/IP is allowed.
func (rl *RateLimiter) CheckRateLimit(userID string, dataSize int64) error {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	usage := rl.getOrCreateUserUsage(userID, now)

	rl.resetCountersIfNeeded(usage, now)

	// Check rate limits
	if err := rl.checkRateLimits(usage, now); err != nil {
		return err
	}

	// Check daily quotas
	if err := rl.checkDailyQuotas(usage, dataSize, now); err != nil {
		return err
	}

	// Update usage counters
	rl.updateUsageCounters(usage, dataSize, now)

	return nil
}

// resetCountersIfNeeded resets usage counters when time periods change.
func (rl *RateLimiter) resetCountersIfNeeded(usage *UserUsage, now time.Time) {
	// Reset counters if day has changed
	if now.Day() != usage.dayStartTime.Day() || now.Month() != usage.dayStartTime.Month() {
		usage.requestsToday = 0
		usage.dataToday = 0
		usage.dayStartTime = now
	}

	// Reset minute/hour counters if needed
	if now.Sub(usage.lastRequestTime) >= time.Minute {
		usage.requestsLastMinute = 0
	}
	if now.Sub(usage.lastRequestTime) >= time.Hour {
		usage.requestsLastHour = 0
	}
}

// checkRateLimits checks minute and hour rate limits.
func (rl *RateLimiter) checkRateLimits(usage *UserUsage, now time.Time) error {
	if rl.requestsPerMinute > 0 && usage.requestsLastMinute >= rl.requestsPerMinute {
		return &RateLimitError{
			Type:       "minute",
			Limit:      rl.requestsPerMinute,
			RetryAfter: time.Minute - now.Sub(usage.lastRequestTime),
		}
	}

	if rl.requestsPerHour > 0 && usage.requestsLastHour >= rl.requestsPerHour {
		return &RateLimitError{
			Type:       "hour",
			Limit:      rl.requestsPerHour,
			RetryAfter: time.Hour - now.Sub(usage.lastRequestTime),
		}
	}

	return nil
}

// checkDailyQuotas checks daily request and data quotas.
func (rl *RateLimiter) checkDailyQuotas(usage *UserUsage, dataSize int64, now time.Time) error {
	if rl.maxRequestsPerDay > 0 && usage.requestsToday >= rl.maxRequestsPerDay {
		return &QuotaExceededError{
			Type:   "requests",
			Limit:  int64(rl.maxRequestsPerDay),
			Used:   int64(usage.requestsToday),
			Resets: time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, now.Location()),
		}
	}

	if rl.maxDataPerDay > 0 && usage.dataToday+dataSize > rl.maxDataPerDay {
		return &QuotaExceededError{
			Type:   "data",
			Limit:  rl.maxDataPerDay,
			Used:   usage.dataToday,
			Resets: time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, now.Location()),
		}
	}

	return nil
}

// updateUsageCounters increments usage counters after a successful request.
func (rl *RateLimiter) updateUsageCounters(usage *UserUsage, dataSize int64, now time.Time) {
	usage.requestsLastMinute++
	usage.requestsLastHour++
	usage.requestsToday++
	usage.dataToday += dataSize
	usage.lastRequestTime = now
}

// getOrCreateUserUsage gets or creates usage tracking for a user.
func (rl *RateLimiter) getOrCreateUserUsage(userID string, now time.Time) *UserUsage {
	usage, exists := rl.userRequests[userID]
	if !exists {
		usage = &UserUsage{
			lastRequestTime: now,
			dayStartTime:    now,
		}
		rl.userRequests[userID] = usage
	}
	return usage
}

// GetUsage returns current usage statistics for a user.
func (rl *RateLimiter) GetUsage(userID string) *UserUsage {
	rl.mu.RLock()
	defer rl.mu.RUnlock()

	if usage, exists := rl.userRequests[userID]; exists {
		// Return a copy to avoid race conditions
		return &UserUsage{
			requestsLastMinute: usage.requestsLastMinute,
			requestsLastHour:   usage.requestsLastHour,
			requestsToday:      usage.requestsToday,
			dataToday:          usage.dataToday,
			lastRequestTime:    usage.lastRequestTime,
			dayStartTime:       usage.dayStartTime,
		}
	}
	return &UserUsage{}
}

// RateLimitError represents a rate limit violation.
type RateLimitError struct {
	Type       string        // "minute" or "hour"
	Limit      int           // the limit that was exceeded
	RetryAfter time.Duration // how long to wait before retrying
}

func (e *RateLimitError) Error() string {
	return fmt.Sprintf("rate limit exceeded for %s (limit: %d, retry after: %v)", e.Type, e.Limit, e.RetryAfter)
}

// QuotaExceededError represents a quota violation.
type QuotaExceededError struct {
	Type   string    // "requests" or "data"
	Limit  int64     // the limit that was exceeded
	Used   int64     // current usage
	Resets time.Time // when the quota resets
}

func (e *QuotaExceededError) Error() string {
	return fmt.Sprintf("quota exceeded for %s (used: %d, limit: %d, resets: %s)",
		e.Type, e.Used, e.Limit, e.Resets.Format(time.RFC3339))
}
