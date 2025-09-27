package pipeline

import (
	"bytes"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestNoOpProgressCallback(t *testing.T) {
	// Should not panic or cause issues
	callback := NoOpProgressCallback{}
	callback.OnStart(10)
	callback.OnProgress(5, 10)
	callback.OnComplete()
	callback.OnError(3, assert.AnError)
}

func TestConsoleProgressCallback(t *testing.T) {
	var buf bytes.Buffer
	callback := NewConsoleProgressCallback(&buf, "Test: ")

	// Test start
	callback.OnStart(10)
	output := buf.String()
	assert.Contains(t, output, "Test: 0/10 (0.0%)")

	// Test progress
	buf.Reset()
	callback.OnProgress(5, 10)
	output = buf.String()
	assert.Contains(t, output, "Test: ")
	assert.Contains(t, output, "5/10")
	assert.Contains(t, output, "50.0%")

	// Test completion
	buf.Reset()
	callback.OnComplete()
	output = buf.String()
	assert.Contains(t, output, "Test: Completed")

	// Test error
	buf.Reset()
	callback.OnError(3, assert.AnError)
	output = buf.String()
	assert.Contains(t, output, "Test: Error at item 3")
}

func TestConsoleProgressCallback_WithOptions(t *testing.T) {
	var buf bytes.Buffer
	callback := NewConsoleProgressCallback(&buf, "Test: ").
		WithWidth(20).
		WithUpdateInterval(time.Millisecond).
		WithOptions(true, true)

	callback.OnStart(10)

	// Allow some time to pass for rate calculation
	time.Sleep(10 * time.Millisecond)

	buf.Reset()
	callback.OnProgress(5, 10)
	output := buf.String()

	// Should include progress bar with custom width
	assert.Contains(t, output, "Test: ")
	// Should have rate and ETA (though ETA might be very small)
	assert.Contains(t, output, "/s") // Rate indicator
}

func TestConsoleProgressCallback_UpdateThrottling(t *testing.T) {
	var buf bytes.Buffer
	callback := NewConsoleProgressCallback(&buf, "Test: ").
		WithUpdateInterval(100 * time.Millisecond)

	callback.OnStart(10)
	buf.Reset()

	// Multiple rapid updates should be throttled
	callback.OnProgress(1, 10)
	firstOutput := buf.String()

	buf.Reset()
	callback.OnProgress(2, 10) // Should be throttled
	secondOutput := buf.String()

	// Second update should be empty due to throttling
	assert.NotEmpty(t, firstOutput)
	assert.Empty(t, secondOutput)

	// But final update should always go through
	buf.Reset()
	callback.OnProgress(10, 10)
	finalOutput := buf.String()
	assert.NotEmpty(t, finalOutput)
}

func TestLogProgressCallback(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	}))

	callback := NewLogProgressCallback(logger, slog.LevelInfo, "Test: ").
		WithInterval(2)

	// Test start
	callback.OnStart(10)
	output := buf.String()
	assert.Contains(t, output, "Test: Starting processing")
	assert.Contains(t, output, "total=10")

	// Test progress (should not log yet due to interval)
	buf.Reset()
	callback.OnProgress(1, 10)
	output = buf.String()
	assert.Empty(t, output) // Should be empty due to interval

	// Test progress at interval
	buf.Reset()
	callback.OnProgress(2, 10)
	output = buf.String()
	assert.Contains(t, output, "Test: Progress update")
	assert.Contains(t, output, "current=2")
	assert.Contains(t, output, "total=10")

	// Test completion
	buf.Reset()
	callback.OnComplete()
	output = buf.String()
	assert.Contains(t, output, "Test: Processing completed")

	// Test error
	buf.Reset()
	callback.OnError(5, assert.AnError)
	output = buf.String()
	assert.Contains(t, output, "Test: Processing error")
	assert.Contains(t, output, "current=5")
}

func TestMultiProgressCallback(t *testing.T) {
	var buf1, buf2 bytes.Buffer

	callback1 := NewConsoleProgressCallback(&buf1, "Console: ")

	logger := slog.New(slog.NewTextHandler(&buf2, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	callback2 := NewLogProgressCallback(logger, slog.LevelInfo, "Log: ")

	multi := NewMultiProgressCallback(callback1, callback2)

	// Test that all callbacks are called
	multi.OnStart(5)

	console := buf1.String()
	log := buf2.String()

	assert.Contains(t, console, "Console: 0/5")
	assert.Contains(t, log, "Log: Starting processing")
	assert.Contains(t, log, "total=5")

	// Test adding another callback
	var buf3 bytes.Buffer
	callback3 := NewConsoleProgressCallback(&buf3, "Added: ")
	multi.Add(callback3)

	multi.OnProgress(3, 5)

	added := buf3.String()
	assert.Contains(t, added, "Added: ")
}

func TestThrottledProgressCallback(t *testing.T) {
	var buf bytes.Buffer
	base := NewConsoleProgressCallback(&buf, "Test: ").
		WithUpdateInterval(0) // No throttling from the base callback
	throttled := NewThrottledProgressCallback(base, 50*time.Millisecond)

	throttled.OnStart(10)
	buf.Reset()

	// First update should go through
	throttled.OnProgress(1, 10)
	first := buf.String()
	assert.NotEmpty(t, first)

	// Immediate second update should be throttled
	buf.Reset()
	throttled.OnProgress(2, 10)
	second := buf.String()
	assert.Empty(t, second)

	// After interval, update should go through
	time.Sleep(60 * time.Millisecond)
	buf.Reset()
	throttled.OnProgress(3, 10)
	third := buf.String()
	assert.NotEmpty(t, third)

	// Final update should always go through regardless of throttling
	buf.Reset()
	throttled.OnProgress(10, 10)
	final := buf.String()
	assert.NotEmpty(t, final)
}

func TestProgressTracker(t *testing.T) {
	tracker := NewProgressTracker(10)

	// Initial state
	stats := tracker.GetStats()
	assert.Equal(t, 10, stats.Total)
	assert.Equal(t, 0, stats.Current)
	assert.Equal(t, 0, stats.Completed)
	assert.Equal(t, 0, stats.Failed)
	assert.Equal(t, 0.0, stats.Rate)
	assert.Equal(t, 0.0, tracker.PercentComplete())

	// Update progress
	time.Sleep(10 * time.Millisecond) // Ensure some time passes
	tracker.Update(5, 4, 1)

	stats = tracker.GetStats()
	assert.Equal(t, 5, stats.Current)
	assert.Equal(t, 4, stats.Completed)
	assert.Equal(t, 1, stats.Failed)
	assert.Greater(t, stats.Rate, 0.0) // Should have positive rate
	assert.Greater(t, stats.Elapsed, time.Duration(0))
	assert.InDelta(t, 50.0, tracker.PercentComplete(), 0.1)

	// Complete processing
	tracker.Update(10, 9, 1)
	assert.InDelta(t, 100.0, tracker.PercentComplete(), 0.1)
}

func TestProgressTracker_ConcurrentAccess(t *testing.T) {
	tracker := NewProgressTracker(100)

	// Multiple goroutines updating concurrently
	done := make(chan bool, 10)

	for i := range 10 {
		go func(id int) {
			defer func() { done <- true }()

			for j := range 10 {
				tracker.Update(id*10+j, id*10+j, 0)
				_ = tracker.GetStats()
				_ = tracker.PercentComplete()
			}
		}(i)
	}

	// Wait for all goroutines
	for range 10 {
		<-done
	}

	// Should not panic and should have consistent final state
	stats := tracker.GetStats()
	assert.GreaterOrEqual(t, stats.Current, 0)
	assert.GreaterOrEqual(t, stats.Completed, 0)
	assert.GreaterOrEqual(t, stats.Elapsed, time.Duration(0))
}

func TestProgressTracker_ZeroTotal(t *testing.T) {
	tracker := NewProgressTracker(0)

	tracker.Update(0, 0, 0)
	assert.Equal(t, 0.0, tracker.PercentComplete())

	stats := tracker.GetStats()
	assert.Equal(t, 0, stats.Total)
}

func TestProgressTracker_RateCalculation(t *testing.T) {
	tracker := NewProgressTracker(100)

	// Wait a bit to ensure time passes
	time.Sleep(50 * time.Millisecond)

	// Update with partial progress to ensure remaining work exists
	tracker.Update(25, 25, 0)
	stats := tracker.GetStats()

	// Rate should be positive
	assert.Greater(t, stats.Rate, 0.0)

	// Estimated total should be calculated when there's remaining work
	if stats.Rate > 0 && stats.Current < stats.Total {
		assert.Greater(t, stats.EstimatedTotal, time.Duration(0))
	}
}

// Helper function to capture console output for testing.
func captureConsoleOutput(callback *ConsoleProgressCallback, fn func()) string {
	// This is a simplified version - in real tests you might want to
	// use a more sophisticated output capture mechanism
	var buf bytes.Buffer
	callback.writer = &buf
	fn()
	return buf.String()
}
