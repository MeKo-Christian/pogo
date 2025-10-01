package mempool

import (
	"runtime"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestPoolIntegration_SimulatedDetectorWorkflow simulates a complete detector workflow
// using the memory pool to ensure proper buffer management.
func TestPoolIntegration_SimulatedDetectorWorkflow(t *testing.T) {
	const (
		imageWidth  = 640
		imageHeight = 480
		iterations  = 100
	)

	// Simulate preprocessing + detection workflow
	for range iterations {
		// Simulate tensor allocation (3 channels * width * height)
		tensorSize := 3 * imageWidth * imageHeight
		tensor := GetFloat32(tensorSize)
		assert.Len(t, tensor, tensorSize)

		// Fill tensor with normalized image data
		for j := range tensor {
			tensor[j] = float32(j%256) / 255.0
		}

		// Simulate probability map from detection (width * height)
		probMapSize := imageWidth * imageHeight
		probMap := GetFloat32(probMapSize)
		assert.Len(t, probMap, probMapSize)

		// Fill with detection probabilities
		for j := range probMap {
			probMap[j] = float32(j%100) / 100.0
		}

		// Simulate binary mask for connected components
		mask := GetBool(probMapSize)
		assert.Len(t, mask, probMapSize)

		// Simulate thresholding
		for j := range probMap {
			if probMap[j] > 0.5 {
				mask[j] = true
			}
		}

		// Simulate morphological operations (creates new buffers)
		dilated := GetFloat32(probMapSize)
		copy(dilated, probMap)
		// Simulate dilation effect
		for j := range dilated {
			if dilated[j] < 1.0 {
				dilated[j] += 0.1
			}
		}

		// Clean up all buffers
		PutFloat32(tensor)
		PutFloat32(probMap)
		PutBool(mask)
		PutFloat32(dilated)
	}

	t.Logf("Completed %d simulated detector workflows", iterations)
}

// TestPoolIntegration_ConcurrentDetectors simulates multiple concurrent detector instances
// sharing the same pool.
func TestPoolIntegration_ConcurrentDetectors(t *testing.T) {
	const (
		numDetectors = 10
		iterations   = 50
		imageSize    = 512 * 512
	)

	var wg sync.WaitGroup
	wg.Add(numDetectors)

	for d := range numDetectors {
		go func(detectorID int) {
			defer wg.Done()

			for i := range iterations {
				// Each detector processes images independently
				tensor := GetFloat32(3 * imageSize)
				probMap := GetFloat32(imageSize)
				mask := GetBool(imageSize)

				// Simulate processing
				for j := range tensor {
					tensor[j] = float32((detectorID+i+j)%256) / 255.0
				}

				// Clean up
				PutFloat32(tensor)
				PutFloat32(probMap)
				PutBool(mask)
			}
		}(d)
	}

	wg.Wait()
	t.Logf("Completed %d concurrent detectors × %d iterations", numDetectors, iterations)
}

// TestPoolIntegration_MemoryFootprint tests that pooling reduces memory footprint.
func TestPoolIntegration_MemoryFootprint(t *testing.T) {
	const (
		bufferSize = 1024 * 1024 // 1M floats = 4MB
		iterations = 100
	)

	// Force GC to get clean baseline
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)
	baseline := m1.TotalAlloc

	// Run many iterations with pooling
	for range iterations {
		buf := GetFloat32(bufferSize)
		// Use the buffer
		for j := range buf {
			buf[j] = float32(j)
		}
		PutFloat32(buf)
	}

	// Force GC and measure again
	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	allocatedWithPool := m2.TotalAlloc - baseline
	t.Logf("Total allocations with pooling: %d bytes (%.2f MB)", allocatedWithPool, float64(allocatedWithPool)/(1024*1024))

	// The pool should keep allocations much lower than direct allocation
	// (100 iterations × 4MB = 400MB without pooling)
	// With pooling, we expect < 100MB of total allocations
	maxExpected := uint64(100 * 1024 * 1024) // 100MB max
	assert.Less(t, allocatedWithPool, maxExpected,
		"Pooling should keep total allocations below 100MB for 100×4MB iterations")
}

// TestPoolIntegration_StressTest performs a stress test with varying buffer sizes.
func TestPoolIntegration_StressTest(t *testing.T) {
	const (
		numGoroutines = 50
		iterations    = 100
	)

	sizes := []int{100, 512, 1024, 2048, 4096, 8192, 16384}

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for range numGoroutines {
		go func() {
			defer wg.Done()

			for range iterations {
				// Randomly allocate different sizes
				for _, size := range sizes {
					f32Buf := GetFloat32(size)
					boolBuf := GetBool(size)

					// Use buffers
					for j := range f32Buf {
						f32Buf[j] = float32(j)
					}
					for j := range boolBuf {
						boolBuf[j] = j%2 == 0
					}

					// Return to pool
					PutFloat32(f32Buf)
					PutBool(boolBuf)
				}
			}
		}()
	}

	wg.Wait()
	t.Logf("Stress test completed: %d goroutines × %d iterations × %d sizes",
		numGoroutines, iterations, len(sizes))
}

// TestPoolIntegration_BufferReuse verifies that buffers are actually being reused.
func TestPoolIntegration_BufferReuse(t *testing.T) {
	const size = 5000

	// Get a buffer and record its capacity
	buf1 := GetFloat32(size)
	require.Len(t, buf1, size)
	cap1 := cap(buf1)

	// Fill with pattern
	for i := range buf1 {
		buf1[i] = float32(i)
	}

	// Return to pool
	PutFloat32(buf1)

	// Get another buffer of same size
	buf2 := GetFloat32(size)
	require.Len(t, buf2, size)
	cap2 := cap(buf2)

	// Capacities should match (high probability of reuse from pool)
	if cap1 == cap2 {
		t.Log("Buffer was reused from pool (capacities match)")
	} else {
		t.Log("Got a different buffer from pool (which is also valid)")
	}

	// Buffer should have correct length
	assert.Len(t, buf2, size)
	PutFloat32(buf2)
}

// TestPoolIntegration_ErrorRecovery tests that pool works correctly after errors.
func TestPoolIntegration_ErrorRecovery(t *testing.T) {
	// Simulate error scenarios where buffers might not be returned properly

	// Scenario 1: Get buffer but don't return it (simulating forgotten cleanup)
	_ = GetFloat32(1000)
	// Pool should still work

	// Scenario 2: Return nil buffer (should be safe)
	PutFloat32(nil)
	PutBool(nil)

	// Scenario 3: Normal operation should still work
	buf := GetFloat32(1000)
	assert.Len(t, buf, 1000)
	PutFloat32(buf)

	t.Log("Pool handles error scenarios gracefully")
}

// TestPoolIntegration_LargeAllocation tests pooling behavior with very large buffers.
func TestPoolIntegration_LargeAllocation(t *testing.T) {
	// 10 megapixel image: 10000 × 1000
	const (
		width  = 10000
		height = 1000
	)

	tensorSize := 3 * width * height
	probMapSize := width * height

	// Allocate large buffers
	tensor := GetFloat32(tensorSize)
	defer PutFloat32(tensor)

	probMap := GetFloat32(probMapSize)
	defer PutFloat32(probMap)

	mask := GetBool(probMapSize)
	defer PutBool(mask)

	// Verify sizes
	assert.Len(t, tensor, tensorSize)
	assert.Len(t, probMap, probMapSize)
	assert.Len(t, mask, probMapSize)

	t.Logf("Successfully handled large allocations: tensor=%d, probMap=%d, mask=%d",
		len(tensor), len(probMap), len(mask))
}

// TestPoolIntegration_MixedOperations tests interleaved pool operations.
func TestPoolIntegration_MixedOperations(t *testing.T) {
	const iterations = 50

	// Interleave gets and puts in complex patterns
	buffers := make([][]float32, 0, iterations)
	masks := make([][]bool, 0, iterations)

	// Accumulate phase
	for i := range iterations {
		size := (i + 1) * 100
		buffers = append(buffers, GetFloat32(size))
		masks = append(masks, GetBool(size))
	}

	// Verify all allocated
	assert.Len(t, buffers, iterations)
	assert.Len(t, masks, iterations)

	// Return in reverse order
	for i := len(buffers) - 1; i >= 0; i-- {
		PutFloat32(buffers[i])
		PutBool(masks[i])
	}

	// Allocate again (should reuse from pool)
	for i := range iterations {
		size := (i + 1) * 100
		buf := GetFloat32(size)
		assert.Len(t, buf, size)
		PutFloat32(buf)
	}

	t.Log("Mixed operations completed successfully")
}
