package benchmark

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTimer(t *testing.T) {
	timer := NewTimer("test_timer")
	assert.Equal(t, "test_timer", timer.name)

	// Sleep for a short duration
	time.Sleep(10 * time.Millisecond)

	duration := timer.Stop()
	assert.GreaterOrEqual(t, duration, 10*time.Millisecond)
	assert.Equal(t, duration, timer.Duration())

	str := timer.String()
	assert.Contains(t, str, "test_timer")
	assert.Contains(t, str, "ms")
}

func TestGetMemoryStats(t *testing.T) {
	stats := GetMemoryStats()
	assert.Positive(t, stats.AllocBytes)
	assert.Positive(t, stats.TotalAllocBytes)
	assert.Positive(t, stats.SysBytes)

	str := stats.String()
	assert.Contains(t, str, "Alloc:")
	assert.Contains(t, str, "KB")
}

func TestBenchmarkSuite(t *testing.T) {
	suite := NewBenchmarkSuite()
	assert.NotNil(t, suite)
	assert.Empty(t, suite.benchmarks)

	// Add a simple benchmark
	suite.Add("test_benchmark", func() error {
		time.Sleep(1 * time.Millisecond)
		return nil
	})

	assert.Len(t, suite.benchmarks, 1)
	assert.Equal(t, "test_benchmark", suite.benchmarks[0].Name)
}

func TestBenchmarkSuiteRun(t *testing.T) {
	suite := NewBenchmarkSuite()

	// Add a successful benchmark
	suite.Add("success_test", func() error {
		time.Sleep(1 * time.Millisecond)
		return nil
	})

	// Add a failing benchmark
	suite.Add("error_test", func() error {
		return errors.New("test error")
	})

	// Run successful benchmark
	result := suite.Run("success_test", 5)
	assert.Equal(t, "success_test", result.Name)
	assert.Equal(t, 5, result.Iterations)
	require.NoError(t, result.Error)
	assert.Positive(t, result.Duration)

	// Run failing benchmark
	result = suite.Run("error_test", 3)
	assert.Equal(t, "error_test", result.Name)
	require.Error(t, result.Error)
	assert.Contains(t, result.Error.Error(), "test error")

	// Run non-existent benchmark
	result = suite.Run("non_existent", 1)
	require.Error(t, result.Error)
	assert.Contains(t, result.Error.Error(), "not found")
}

func TestBenchmarkSuiteRunAll(t *testing.T) {
	suite := NewBenchmarkSuite()

	// Add multiple benchmarks
	suite.Add("fast_test", func() error {
		time.Sleep(1 * time.Millisecond)
		return nil
	})

	suite.Add("slow_test", func() error {
		time.Sleep(5 * time.Millisecond)
		return nil
	})

	// Run all benchmarks
	results := suite.RunAll(3)
	require.Len(t, results, 2)

	// Check that results are stored
	storedResults := suite.Results()
	assert.Equal(t, results, storedResults)

	// Verify results
	fastResult := results[0]
	slowResult := results[1]

	assert.Equal(t, "fast_test", fastResult.Name)
	assert.Equal(t, "slow_test", slowResult.Name)
	assert.Equal(t, 3, fastResult.Iterations)
	assert.Equal(t, 3, slowResult.Iterations)
	assert.NoError(t, fastResult.Error)
	assert.NoError(t, slowResult.Error)

	// Slow test should take longer than fast test
	assert.Greater(t, slowResult.Duration, fastResult.Duration)
}

func TestBenchmarkResult(t *testing.T) {
	// Test successful result
	result := BenchmarkResult{
		Name:         "test_result",
		Duration:     100 * time.Millisecond,
		Iterations:   10,
		MemoryBefore: MemoryStats{AllocBytes: 1000},
		MemoryAfter:  MemoryStats{AllocBytes: 2000},
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

func TestOCRPipelineBenchmark(t *testing.T) {
	ocr := NewOCRPipelineBenchmark()
	assert.NotNil(t, ocr)
	assert.NotNil(t, ocr.BenchmarkSuite)

	// Add different types of benchmarks
	ocr.AddImageProcessingBenchmark("resize", func() error {
		time.Sleep(1 * time.Millisecond)
		return nil
	})

	ocr.AddDetectionBenchmark("text_detection", func() error {
		time.Sleep(2 * time.Millisecond)
		return nil
	})

	ocr.AddRecognitionBenchmark("text_recognition", func() error {
		time.Sleep(3 * time.Millisecond)
		return nil
	})

	ocr.AddPipelineBenchmark("full_pipeline", func() error {
		time.Sleep(5 * time.Millisecond)
		return nil
	})

	assert.Len(t, ocr.benchmarks, 4)

	// Check that names are prefixed correctly
	names := make([]string, len(ocr.benchmarks))
	for i, b := range ocr.benchmarks {
		names[i] = b.Name
	}

	assert.Contains(t, names, "ImageProcessing_resize")
	assert.Contains(t, names, "Detection_text_detection")
	assert.Contains(t, names, "Recognition_text_recognition")
	assert.Contains(t, names, "Pipeline_full_pipeline")
}

// Example benchmark test that shows how to use the framework.
func TestExampleBenchmarkUsage(t *testing.T) {
	// Create a benchmark suite
	suite := NewBenchmarkSuite()

	// Add some example operations
	suite.Add("string_concat", func() error {
		var result string
		for range 1000 {
			result += "a"
		}
		return nil
	})

	suite.Add("slice_append", func() error {
		var slice []int
		for i := range 1000 {
			slice = append(slice, i)
		}
		_ = slice // result intentionally unused in benchmark
		return nil
	})

	// Run benchmarks
	results := suite.RunAll(10)
	require.Len(t, results, 2)

	// Print results for demonstration
	t.Log("Example benchmark results:")
	for _, result := range results {
		t.Log(result.String())
	}

	// All should succeed
	for _, result := range results {
		require.NoError(t, result.Error)
		assert.Positive(t, result.Duration)
	}
}

// Benchmark function for Go's testing framework.
func BenchmarkTimerCreation(b *testing.B) {
	for range b.N {
		timer := NewTimer("benchmark_test")
		timer.Stop()
	}
}

func BenchmarkMemoryStatsRetrieval(b *testing.B) {
	for range b.N {
		GetMemoryStats()
	}
}
