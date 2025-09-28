package common

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestGetMemoryStats(t *testing.T) {
	stats := GetMemoryStats()
	assert.Positive(t, stats.Alloc)
	assert.Positive(t, stats.TotalAlloc)
	assert.Positive(t, stats.Sys)

	str := stats.String()
	assert.Contains(t, str, "Alloc:")
	assert.Contains(t, str, "KB")
}

func TestBenchmarkResult(t *testing.T) {
	// Test successful result
	result := BenchmarkResult{
		Name:         "test_result",
		Duration:     100 * time.Millisecond,
		Iterations:   10,
		MemoryBefore: MemoryStats{Alloc: 1000},
		MemoryAfter:  MemoryStats{Alloc: 2000},
	}

	str := result.String()
	assert.Contains(t, str, "test_result")
	assert.Contains(t, str, "10 iterations")
	assert.Contains(t, str, "10ms")  // avg duration
	assert.Contains(t, str, "100ms") // total duration

	// Test error result
	errorResult := BenchmarkResult{
		Name:  "error_result",
		Error: errors.New("test error"),
	}

	str = errorResult.String()
	assert.Contains(t, str, "error_result")
	assert.Contains(t, str, "ERROR")
	assert.Contains(t, str, "test error")
}

func BenchmarkMemoryStatsRetrieval(b *testing.B) {
	for range b.N {
		GetMemoryStats()
	}
}