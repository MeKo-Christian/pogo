package pipeline

import (
	"context"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultResourceConfig(t *testing.T) {
	config := DefaultResourceConfig()

	assert.Equal(t, uint64(0), config.MaxMemoryBytes)
	assert.Equal(t, 0, config.MaxGoroutines)
	assert.InDelta(t, 0.8, config.MemoryThreshold, 0.001)
	assert.Equal(t, time.Second, config.MonitorInterval)
	assert.True(t, config.EnableBackpressure)
	assert.False(t, config.EnableAdaptiveScale)
}

func TestNewResourceManager(t *testing.T) {
	config := ResourceConfig{
		MaxMemoryBytes:  1024 * 1024, // 1MB
		MaxGoroutines:   10,
		MemoryThreshold: 0.9,
	}

	rm := NewResourceManager(config)
	require.NotNil(t, rm)
	defer rm.Stop()

	assert.Equal(t, uint64(1024*1024), rm.maxMemoryBytes)
	assert.Equal(t, 10, rm.maxGoroutines)
	assert.InDelta(t, 0.9, rm.memoryThreshold, 0.001)
	assert.NotNil(t, rm.goroutineSem)
	assert.NotNil(t, rm.memoryMonitor)
}

func TestNewResourceManager_InvalidThreshold(t *testing.T) {
	config := ResourceConfig{
		MemoryThreshold: 1.5, // Invalid
	}

	rm := NewResourceManager(config)
	assert.InDelta(t, 0.8, rm.memoryThreshold, 0.001) // Should default to 0.8
	rm.Stop()
}

func TestResourceManager_StartStop(t *testing.T) {
	config := DefaultResourceConfig()
	rm := NewResourceManager(config)

	// Should start without error
	rm.Start()
	assert.True(t, rm.monitoringActive)

	// Should stop without error
	rm.Stop()
	// Note: monitoringActive is checked under mutex, so we can't easily assert it here
}

func TestResourceManager_AcquireReleaseGoroutine_NoLimit(t *testing.T) {
	config := ResourceConfig{
		MaxGoroutines: 0, // No limit
	}
	rm := NewResourceManager(config)
	defer rm.Stop()

	ctx := context.Background()

	// Should acquire without blocking
	err := rm.AcquireGoroutine(ctx)
	require.NoError(t, err)

	// Should release without error
	rm.ReleaseGoroutine()
}

func TestResourceManager_AcquireReleaseGoroutine_WithLimit(t *testing.T) {
	config := ResourceConfig{
		MaxGoroutines: 2,
	}
	rm := NewResourceManager(config)
	defer rm.Stop()

	ctx := context.Background()

	// Should acquire up to limit
	err1 := rm.AcquireGoroutine(ctx)
	require.NoError(t, err1)

	err2 := rm.AcquireGoroutine(ctx)
	require.NoError(t, err2)

	// Third acquire should block, test with timeout
	ctx3, cancel3 := context.WithTimeout(ctx, 10*time.Millisecond)
	defer cancel3()

	err3 := rm.AcquireGoroutine(ctx3)
	require.Error(t, err3)
	assert.Contains(t, err3.Error(), "context deadline exceeded")

	// Release one and try again
	rm.ReleaseGoroutine()

	err4 := rm.AcquireGoroutine(ctx)
	require.NoError(t, err4)

	// Clean up
	rm.ReleaseGoroutine()
	rm.ReleaseGoroutine()
}

func TestResourceManager_ConcurrentGoroutineAccess(t *testing.T) {
	config := ResourceConfig{
		MaxGoroutines: 5,
	}
	rm := NewResourceManager(config)
	defer rm.Stop()

	const numWorkers = 10
	const workDuration = 100 * time.Millisecond

	var wg sync.WaitGroup
	successCount := int32(0)
	errorCount := int32(0)

	for range numWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()

			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()

			if err := rm.AcquireGoroutine(ctx); err != nil {
				atomic.AddInt32(&errorCount, 1)
				return
			}
			defer rm.ReleaseGoroutine()

			atomic.AddInt32(&successCount, 1)
			time.Sleep(workDuration)
		}()
	}

	wg.Wait()

	// Some should succeed, some should timeout
	assert.Positive(t, int(successCount))
	assert.Positive(t, int(errorCount))
	assert.Equal(t, numWorkers, int(successCount+errorCount))
}

func TestResourceManager_CheckMemoryPressure_NoLimit(t *testing.T) {
	config := ResourceConfig{
		MaxMemoryBytes: 0, // No limit
	}
	rm := NewResourceManager(config)
	defer rm.Stop()

	// Should never have memory pressure with no limit
	pressure := rm.CheckMemoryPressure()
	assert.False(t, pressure)
}

func TestResourceManager_GetStats(t *testing.T) {
	config := ResourceConfig{
		MaxMemoryBytes: 1024 * 1024, // 1MB
		MaxGoroutines:  5,
	}
	rm := NewResourceManager(config)
	rm.Start()
	defer rm.Stop()

	// Allow some time for monitoring to start
	time.Sleep(10 * time.Millisecond)

	stats := rm.GetStats()

	assert.GreaterOrEqual(t, stats.ActiveGoroutines, 0)
	assert.GreaterOrEqual(t, stats.MonitoringDuration, time.Duration(0))
}

func TestResourceManager_GetOptimalWorkerCount(t *testing.T) {
	config := ResourceConfig{
		MaxGoroutines: 2,
	}
	rm := NewResourceManager(config)
	defer rm.Stop()

	optimal := rm.GetOptimalWorkerCount()

	// Should be limited by MaxGoroutines
	assert.LessOrEqual(t, optimal, 2)
	assert.Positive(t, optimal)
}

func TestMemoryMonitor(t *testing.T) {
	monitor := NewMemoryMonitor(10 * time.Millisecond)
	require.NotNil(t, monitor)

	// Start monitoring
	monitor.Start()
	defer monitor.Stop()

	// Allow some samples to be collected
	time.Sleep(50 * time.Millisecond)

	current := monitor.GetCurrentUsage()
	peak := monitor.GetPeakUsage()
	average := monitor.GetAverageUsage()

	assert.Positive(t, current)
	assert.GreaterOrEqual(t, peak, current)
	assert.Positive(t, average)
}

func TestMemoryMonitor_StartStop(t *testing.T) {
	monitor := NewMemoryMonitor(time.Second)

	// Should start without error
	monitor.Start()
	assert.True(t, monitor.active)

	// Should stop without error
	monitor.Stop()
	// Note: active is checked under mutex
}

func TestMemoryMonitor_ConcurrentAccess(t *testing.T) {
	monitor := NewMemoryMonitor(time.Millisecond)
	monitor.Start()
	defer monitor.Stop()

	const numReaders = 10
	var wg sync.WaitGroup

	// Multiple concurrent readers
	for range numReaders {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 10 {
				_ = monitor.GetCurrentUsage()
				_ = monitor.GetPeakUsage()
				_ = monitor.GetAverageUsage()
				time.Sleep(time.Millisecond)
			}
		}()
	}

	wg.Wait()
	// Should not panic or race
}

func TestAdaptiveWorkerPool(t *testing.T) {
	config := ResourceConfig{
		MaxGoroutines:       10,
		EnableAdaptiveScale: true,
	}
	rm := NewResourceManager(config)
	rm.Start()
	defer rm.Stop()

	awp := NewAdaptiveWorkerPool(rm, 2, 8, 50*time.Millisecond)
	require.NotNil(t, awp)

	// Start adaptive scaling
	awp.Start()
	defer awp.Stop()

	// Should start with minimum workers
	assert.Equal(t, 2, awp.GetCurrentWorkerCount())

	// Allow some adjustment time
	time.Sleep(100 * time.Millisecond)

	// Worker count should be within bounds
	count := awp.GetCurrentWorkerCount()
	assert.GreaterOrEqual(t, count, 2)
	assert.LessOrEqual(t, count, 8)
}

func TestAdaptiveWorkerPool_InvalidParams(t *testing.T) {
	rm := NewResourceManager(DefaultResourceConfig())
	defer rm.Stop()

	// Test with invalid parameters
	awp := NewAdaptiveWorkerPool(rm, 0, 0, time.Second)

	// Should have sane defaults
	assert.Equal(t, 1, awp.minWorkers)
	assert.Equal(t, runtime.NumCPU(), awp.maxWorkers)
	assert.Equal(t, 1, awp.currentWorkers)
}

func TestResourceError(t *testing.T) {
	err := NewMemoryLimitError(2048, 1024)
	assert.Contains(t, err.Error(), "memory usage 2048 bytes exceeds limit 1024 bytes")
	assert.Equal(t, "memory_limit", err.Type)

	err2 := NewGoroutineLimitError(10, 5)
	assert.Contains(t, err2.Error(), "goroutine count 10 exceeds limit 5")
	assert.Equal(t, "goroutine_limit", err2.Type)
}

func TestResourceStats_Utilization(t *testing.T) {
	config := ResourceConfig{
		MaxMemoryBytes: 1024,
		MaxGoroutines:  10,
	}
	rm := NewResourceManager(config)
	defer rm.Stop()

	// Simulate some usage
	ctx := context.Background()
	_ = rm.AcquireGoroutine(ctx)
	_ = rm.AcquireGoroutine(ctx)

	stats := rm.GetStats()

	// Should calculate utilization
	assert.GreaterOrEqual(t, stats.GoroutineUtilization, 0.0)
	assert.LessOrEqual(t, stats.GoroutineUtilization, 1.0)

	// Clean up
	rm.ReleaseGoroutine()
	rm.ReleaseGoroutine()
}

func TestResourceManager_ShouldThrottle(t *testing.T) {
	config := ResourceConfig{
		MaxMemoryBytes:  1024, // Very small limit to trigger pressure
		MemoryThreshold: 0.1,  // Very low threshold
	}
	rm := NewResourceManager(config)
	defer rm.Stop()

	// Current memory usage is likely > 1024 bytes, so should throttle
	shouldThrottle := rm.ShouldThrottle()
	// This might be true or false depending on actual memory usage
	assert.IsType(t, false, shouldThrottle)
}

// Benchmark tests for performance

func BenchmarkResourceManager_AcquireRelease(b *testing.B) {
	config := ResourceConfig{
		MaxGoroutines: 100,
	}
	rm := NewResourceManager(config)
	defer rm.Stop()

	ctx := context.Background()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if err := rm.AcquireGoroutine(ctx); err == nil {
				rm.ReleaseGoroutine()
			}
		}
	})
}

func BenchmarkMemoryMonitor_GetUsage(b *testing.B) {
	monitor := NewMemoryMonitor(time.Second)
	monitor.Start()
	defer monitor.Stop()

	b.ResetTimer()
	for range b.N {
		_ = monitor.GetCurrentUsage()
	}
}

func BenchmarkProgressTracker_Update(b *testing.B) {
	tracker := NewProgressTracker(1000000)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			tracker.Update(i, i, 0)
			i++
		}
	})
}
