package server

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewRateLimiter(t *testing.T) {
	rl := NewRateLimiter(10, 100, 1000, 1024*1024)

	assert.NotNil(t, rl)
	assert.Equal(t, 10, rl.requestsPerMinute)
	assert.Equal(t, 100, rl.requestsPerHour)
	assert.Equal(t, 1000, rl.maxRequestsPerDay)
	assert.Equal(t, int64(1024*1024), rl.maxDataPerDay)
	assert.NotNil(t, rl.userRequests)
}

func TestRateLimiter_CheckRateLimit_NoLimits(t *testing.T) {
	rl := NewRateLimiter(0, 0, 0, 0) // No limits

	err := rl.CheckRateLimit("user1", 100)
	assert.NoError(t, err)

	usage := rl.GetUsage("user1")
	assert.Equal(t, 1, usage.requestsToday)
	assert.Equal(t, int64(100), usage.dataToday)
}

func TestRateLimiter_CheckRateLimit_RequestsPerMinute(t *testing.T) {
	rl := NewRateLimiter(2, 0, 0, 0) // 2 requests per minute

	userID := "user1"

	// First request should succeed
	err := rl.CheckRateLimit(userID, 0)
	assert.NoError(t, err)

	// Second request should succeed
	err = rl.CheckRateLimit(userID, 0)
	assert.NoError(t, err)

	// Third request should fail
	err = rl.CheckRateLimit(userID, 0)
	assert.Error(t, err)

	rateLimitErr := &RateLimitError{}
	ok := errors.As(err, &rateLimitErr)
	require.True(t, ok)
	assert.Equal(t, "minute", rateLimitErr.Type)
	assert.Equal(t, 2, rateLimitErr.Limit)
	assert.Positive(t, rateLimitErr.RetryAfter)
}

func TestRateLimiter_CheckRateLimit_RequestsPerHour(t *testing.T) {
	rl := NewRateLimiter(0, 3, 0, 0) // 3 requests per hour

	userID := "user1"

	// Make 3 requests
	for range 3 {
		err := rl.CheckRateLimit(userID, 0)
		assert.NoError(t, err)
	}

	// Fourth request should fail
	err := rl.CheckRateLimit(userID, 0)
	assert.Error(t, err)

	rateLimitErr := &RateLimitError{}
	ok := errors.As(err, &rateLimitErr)
	require.True(t, ok)
	assert.Equal(t, "hour", rateLimitErr.Type)
	assert.Equal(t, 3, rateLimitErr.Limit)
}

func TestRateLimiter_CheckRateLimit_MaxRequestsPerDay(t *testing.T) {
	rl := NewRateLimiter(0, 0, 2, 0) // 2 requests per day

	userID := "user1"

	// First request should succeed
	err := rl.CheckRateLimit(userID, 0)
	assert.NoError(t, err)

	// Second request should succeed
	err = rl.CheckRateLimit(userID, 0)
	assert.NoError(t, err)

	// Third request should fail
	err = rl.CheckRateLimit(userID, 0)
	assert.Error(t, err)

	quotaErr := &QuotaExceededError{}
	ok := errors.As(err, &quotaErr)
	require.True(t, ok)
	assert.Equal(t, "requests", quotaErr.Type)
	assert.Equal(t, int64(2), quotaErr.Limit)
	assert.Equal(t, int64(2), quotaErr.Used)
	assert.True(t, quotaErr.Resets.After(time.Now()))
}

func TestRateLimiter_CheckRateLimit_MaxDataPerDay(t *testing.T) {
	rl := NewRateLimiter(0, 0, 0, 1000) // 1000 bytes per day

	userID := "user1"

	// First request with 500 bytes should succeed
	err := rl.CheckRateLimit(userID, 500)
	assert.NoError(t, err)

	// Second request with 400 bytes should succeed
	err = rl.CheckRateLimit(userID, 400)
	assert.NoError(t, err)

	// Third request with 200 bytes should fail
	err = rl.CheckRateLimit(userID, 200)
	assert.Error(t, err)

	quotaErr := &QuotaExceededError{}
	ok := errors.As(err, &quotaErr)
	require.True(t, ok)
	assert.Equal(t, "data", quotaErr.Type)
	assert.Equal(t, int64(1000), quotaErr.Limit)
	assert.Equal(t, int64(900), quotaErr.Used)
}

func TestRateLimiter_CheckRateLimit_TimeReset(t *testing.T) {
	rl := NewRateLimiter(1, 0, 0, 0) // 1 request per minute

	userID := "user1"

	// First request
	err := rl.CheckRateLimit(userID, 0)
	assert.NoError(t, err)

	// Second request should fail
	err = rl.CheckRateLimit(userID, 0)
	assert.Error(t, err)

	// Manually reset the last request time to more than a minute ago
	rl.mu.Lock()
	if usage, exists := rl.userRequests[userID]; exists {
		usage.lastRequestTime = time.Now().Add(-2 * time.Minute)
	}
	rl.mu.Unlock()

	// Third request should succeed after time reset
	err = rl.CheckRateLimit(userID, 0)
	assert.NoError(t, err)
}

func TestRateLimiter_CheckRateLimit_DayReset(t *testing.T) {
	rl := NewRateLimiter(0, 0, 1, 0) // 1 request per day

	userID := "user1"

	// First request
	err := rl.CheckRateLimit(userID, 0)
	assert.NoError(t, err)

	// Second request should fail
	err = rl.CheckRateLimit(userID, 0)
	assert.Error(t, err)

	// Manually reset the day start time to yesterday
	rl.mu.Lock()
	if usage, exists := rl.userRequests[userID]; exists {
		usage.dayStartTime = time.Now().AddDate(0, 0, -1)
	}
	rl.mu.Unlock()

	// Third request should succeed after day reset
	err = rl.CheckRateLimit(userID, 0)
	assert.NoError(t, err)
}

func TestRateLimiter_GetUsage(t *testing.T) {
	rl := NewRateLimiter(10, 100, 1000, 10000)

	userID := "user1"

	// No usage initially
	usage := rl.GetUsage(userID)
	assert.Equal(t, 0, usage.requestsLastMinute)
	assert.Equal(t, 0, usage.requestsLastHour)
	assert.Equal(t, 0, usage.requestsToday)
	assert.Equal(t, int64(0), usage.dataToday)

	// Make some requests
	err := rl.CheckRateLimit(userID, 500)
	assert.NoError(t, err)

	err = rl.CheckRateLimit(userID, 300)
	assert.NoError(t, err)

	// Check usage
	usage = rl.GetUsage(userID)
	assert.Equal(t, 2, usage.requestsLastMinute)
	assert.Equal(t, 2, usage.requestsLastHour)
	assert.Equal(t, 2, usage.requestsToday)
	assert.Equal(t, int64(800), usage.dataToday)
	assert.True(t, usage.lastRequestTime.After(time.Now().Add(-time.Minute)))
}

func TestRateLimiter_GetUsage_NonExistentUser(t *testing.T) {
	rl := NewRateLimiter(10, 100, 1000, 10000)

	usage := rl.GetUsage("nonexistent")
	assert.Equal(t, 0, usage.requestsLastMinute)
	assert.Equal(t, 0, usage.requestsLastHour)
	assert.Equal(t, 0, usage.requestsToday)
	assert.Equal(t, int64(0), usage.dataToday)
	assert.True(t, usage.lastRequestTime.IsZero())
	assert.True(t, usage.dayStartTime.IsZero())
}

func TestRateLimiter_MultipleUsers(t *testing.T) {
	rl := NewRateLimiter(2, 0, 0, 0) // 2 requests per minute

	user1 := "user1"
	user2 := "user2"

	// User1 makes 2 requests
	err := rl.CheckRateLimit(user1, 0)
	assert.NoError(t, err)
	err = rl.CheckRateLimit(user1, 0)
	assert.NoError(t, err)

	// User1's third request should fail
	err = rl.CheckRateLimit(user1, 0)
	assert.Error(t, err)

	// User2 should still be able to make requests
	err = rl.CheckRateLimit(user2, 0)
	assert.NoError(t, err)
	err = rl.CheckRateLimit(user2, 0)
	assert.NoError(t, err)

	// User2's third request should also fail
	err = rl.CheckRateLimit(user2, 0)
	assert.Error(t, err)
}

func TestRateLimitError_Error(t *testing.T) {
	err := &RateLimitError{
		Type:       "minute",
		Limit:      10,
		RetryAfter: time.Minute * 5,
	}

	expected := "rate limit exceeded for minute (limit: 10, retry after: 5m0s)"
	assert.Equal(t, expected, err.Error())
}

func TestQuotaExceededError_Error(t *testing.T) {
	resetTime := time.Date(2024, 1, 2, 0, 0, 0, 0, time.UTC)
	err := &QuotaExceededError{
		Type:   "data",
		Limit:  1000,
		Used:   950,
		Resets: resetTime,
	}

	expected := "quota exceeded for data (used: 950, limit: 1000, resets: 2024-01-02T00:00:00Z)"
	assert.Equal(t, expected, err.Error())
}

// Benchmark tests.
func BenchmarkRateLimiter_CheckRateLimit(b *testing.B) {
	rl := NewRateLimiter(100, 1000, 10000, 1024*1024)

	b.ResetTimer()
	for range b.N {
		_ = rl.CheckRateLimit("benchuser", 100)
	}
}

func BenchmarkRateLimiter_GetUsage(b *testing.B) {
	rl := NewRateLimiter(100, 1000, 10000, 1024*1024)
	_ = rl.CheckRateLimit("benchuser", 100) // Initialize usage

	b.ResetTimer()
	for range b.N {
		_ = rl.GetUsage("benchuser")
	}
}
