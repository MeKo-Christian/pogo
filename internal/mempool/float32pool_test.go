package mempool

import (
	"fmt"
	"runtime"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSizeClass(t *testing.T) {
	tests := []struct {
		name     string
		input    int
		expected int
	}{
		{
			name:     "small size gets minimum",
			input:    1,
			expected: 1024,
		},
		{
			name:     "exactly 1024",
			input:    1024,
			expected: 1024,
		},
		{
			name:     "just over 1024",
			input:    1025,
			expected: 2048,
		},
		{
			name:     "exact multiple of 1024",
			input:    2048,
			expected: 2048,
		},
		{
			name:     "odd number",
			input:    1500,
			expected: 2048,
		},
		{
			name:     "large size",
			input:    10000,
			expected: 10240,
		},
		{
			name:     "zero size",
			input:    0,
			expected: 1024,
		},
		{
			name:     "negative size",
			input:    -1,
			expected: 1024,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := sizeClass(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestGetFloat32_BasicFunctionality(t *testing.T) {
	tests := []struct {
		name        string
		requestSize int
		expectedLen int
		minCap      int
	}{
		{
			name:        "small buffer",
			requestSize: 100,
			expectedLen: 100,
			minCap:      100,
		},
		{
			name:        "exactly 1024",
			requestSize: 1024,
			expectedLen: 1024,
			minCap:      1024,
		},
		{
			name:        "large buffer",
			requestSize: 5000,
			expectedLen: 5000,
			minCap:      5000,
		},
		{
			name:        "zero size",
			requestSize: 0,
			expectedLen: 0,
			minCap:      0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := GetFloat32(tt.requestSize)

			assert.Len(t, buf, tt.expectedLen)
			assert.GreaterOrEqual(t, cap(buf), tt.minCap)

			// Verify we can write to the buffer
			if len(buf) > 0 {
				buf[0] = 42.0
				assert.InDelta(t, float32(42.0), buf[0], 0.0001)
			}
		})
	}
}

func TestPutFloat32_BasicFunctionality(t *testing.T) {
	t.Run("put valid buffer", func(t *testing.T) {
		buf := GetFloat32(1000)
		require.NotNil(t, buf)

		// This should not panic
		PutFloat32(buf)
	})

	t.Run("put nil buffer", func(t *testing.T) {
		// This should not panic
		PutFloat32(nil)
	})

	t.Run("put empty buffer", func(t *testing.T) {
		buf := make([]float32, 0)
		// This should not panic
		PutFloat32(buf)
	})
}

func TestMemoryPoolReuse(t *testing.T) {
	// Test that buffers are actually reused
	size := 2000

	// Get a buffer and modify it
	buf1 := GetFloat32(size)
	require.Len(t, buf1, size)

	// Fill with a pattern
	for i := range buf1 {
		buf1[i] = float32(i)
	}

	// Put it back
	PutFloat32(buf1)

	// Get another buffer of the same size
	buf2 := GetFloat32(size)
	require.Len(t, buf2, size)

	// The buffers might be the same (reused) or different (new allocation)
	// Both are valid behaviors for a pool
	assert.GreaterOrEqual(t, cap(buf2), size)
}

func TestConcurrentAccess(t *testing.T) {
	const numGoroutines = 100
	const numIterations = 100
	const bufferSize = 1500

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Test concurrent gets and puts
	for range numGoroutines {
		go func() {
			defer wg.Done()

			for range numIterations {
				// Get a buffer
				buf := GetFloat32(bufferSize)
				assert.Len(t, buf, bufferSize)
				assert.GreaterOrEqual(t, cap(buf), bufferSize)

				// Use the buffer
				for k := 0; k < len(buf); k++ {
					buf[k] = float32(k)
				}

				// Put it back
				PutFloat32(buf)
			}
		}()
	}

	wg.Wait()
}

func TestDifferentSizeClasses(t *testing.T) {
	// Test that different size classes don't interfere
	sizes := []int{100, 1500, 3000, 10000}
	buffers := make([][]float32, len(sizes))

	// Get buffers of different sizes
	for i, size := range sizes {
		buffers[i] = GetFloat32(size)
		assert.Len(t, buffers[i], size)

		// Fill with unique pattern
		for j := range buffers[i] {
			buffers[i][j] = float32(i*1000 + j)
		}
	}

	// Put them all back
	for _, buf := range buffers {
		PutFloat32(buf)
	}

	// Get them again and verify independence
	for _, size := range sizes {
		newBuf := GetFloat32(size)
		assert.Len(t, newBuf, size)
		// The pool doesn't guarantee clearing, so we don't check contents
	}
}

func TestSizeClassBoundaries(t *testing.T) {
	// Test behavior around size class boundaries
	testCases := []struct {
		size          int
		expectedClass int
	}{
		{1023, 1024},
		{1024, 1024},
		{1025, 2048},
		{2047, 2048},
		{2048, 2048},
		{2049, 3072},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("size_%d", tc.size), func(t *testing.T) {
			buf := GetFloat32(tc.size)
			assert.Len(t, buf, tc.size)
			// Capacity should be at least the size class
			expectedCap := sizeClass(tc.size)
			assert.GreaterOrEqual(t, cap(buf), expectedCap)
			PutFloat32(buf)
		})
	}
}

func TestPoolGrowth(t *testing.T) {
	// Test that the pool can handle growing demands
	const maxSize = 10000
	var buffers [][]float32

	// Get increasingly large buffers
	for size := 1000; size <= maxSize; size += 1000 {
		buf := GetFloat32(size)
		assert.Len(t, buf, size)
		buffers = append(buffers, buf)
	}

	// Put them all back
	for _, buf := range buffers {
		PutFloat32(buf)
	}

	// Verify we can still get buffers
	for size := 1000; size <= maxSize; size += 1000 {
		buf := GetFloat32(size)
		assert.Len(t, buf, size)
		PutFloat32(buf)
	}
}

func TestMemoryBehavior(t *testing.T) {
	// Test that using the pool doesn't cause obvious memory leaks
	const iterations = 1000
	const bufferSize = 5000

	// Force GC before starting
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	// Perform many allocations through the pool
	for range iterations {
		buf := GetFloat32(bufferSize)

		// Use the buffer
		for j := 0; j < len(buf); j++ {
			buf[j] = float32(j)
		}

		PutFloat32(buf)
	}

	// Force GC after operations
	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	// We can't make strong assertions about memory usage since pools
	// may retain some buffers, but this test helps detect obvious leaks
	t.Logf("Memory before: %d bytes, after: %d bytes", m1.Alloc, m2.Alloc)
}

// Edge case tests.
func TestEdgeCases(t *testing.T) {
	t.Run("very large buffer", func(t *testing.T) {
		size := 1000000 // 1M floats = 4MB
		buf := GetFloat32(size)
		assert.Len(t, buf, size)
		assert.GreaterOrEqual(t, cap(buf), size)
		PutFloat32(buf)
	})

	t.Run("buffer capacity vs length", func(t *testing.T) {
		buf := GetFloat32(100)
		originalCap := cap(buf)

		// Extend the slice within capacity
		if originalCap > 100 {
			extended := buf[:originalCap]
			PutFloat32(extended)
		}

		PutFloat32(buf)
	})

	t.Run("repeated get/put cycles", func(t *testing.T) {
		size := 2000
		for range 100 {
			buf := GetFloat32(size)
			assert.Len(t, buf, size)
			PutFloat32(buf)
		}
	})
}

// Benchmark tests.
func BenchmarkGetFloat32_Small(b *testing.B) {
	for range b.N {
		buf := GetFloat32(100)
		PutFloat32(buf)
	}
}

func BenchmarkGetFloat32_Medium(b *testing.B) {
	for range b.N {
		buf := GetFloat32(2000)
		PutFloat32(buf)
	}
}

func BenchmarkGetFloat32_Large(b *testing.B) {
	for range b.N {
		buf := GetFloat32(10000)
		PutFloat32(buf)
	}
}

func BenchmarkDirectAllocation_Medium(b *testing.B) {
	// Compare with direct allocation
	for range b.N {
		_ = make([]float32, 2000)
	}
}

func BenchmarkConcurrentAccess(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			buf := GetFloat32(1500)
			// Simulate some work
			for i := range buf {
				buf[i] = float32(i)
			}
			PutFloat32(buf)
		}
	})
}

func BenchmarkSizeClass(b *testing.B) {
	sizes := []int{100, 1024, 1500, 5000, 10000}

	for range b.N {
		for _, size := range sizes {
			_ = sizeClass(size)
		}
	}
}

// Bool pool tests.
func TestGetBool_BasicFunctionality(t *testing.T) {
	tests := []struct {
		name        string
		requestSize int
		expectedLen int
		minCap      int
	}{
		{
			name:        "small buffer",
			requestSize: 100,
			expectedLen: 100,
			minCap:      100,
		},
		{
			name:        "exactly 1024",
			requestSize: 1024,
			expectedLen: 1024,
			minCap:      1024,
		},
		{
			name:        "large buffer",
			requestSize: 5000,
			expectedLen: 5000,
			minCap:      5000,
		},
		{
			name:        "zero size",
			requestSize: 0,
			expectedLen: 0,
			minCap:      0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := GetBool(tt.requestSize)

			assert.Len(t, buf, tt.expectedLen)
			assert.GreaterOrEqual(t, cap(buf), tt.minCap)

			// Verify buffer is zeroed
			for i, v := range buf {
				assert.False(t, v, "buffer element %d should be false", i)
			}

			// Verify we can write to the buffer
			if len(buf) > 0 {
				buf[0] = true
				assert.True(t, buf[0])
			}
		})
	}
}

func TestPutBool_BasicFunctionality(t *testing.T) {
	t.Run("put valid buffer", func(t *testing.T) {
		buf := GetBool(1000)
		require.NotNil(t, buf)

		// Modify buffer
		for i := range buf {
			buf[i] = i%2 == 0
		}

		// This should not panic
		PutBool(buf)
	})

	t.Run("put nil buffer", func(t *testing.T) {
		// This should not panic
		PutBool(nil)
	})

	t.Run("put empty buffer", func(t *testing.T) {
		buf := make([]bool, 0)
		// This should not panic
		PutBool(buf)
	})
}

func TestBoolPoolReuse(t *testing.T) {
	// Test that buffers are actually reused and zeroed
	size := 2000

	// Get a buffer and modify it
	buf1 := GetBool(size)
	require.Len(t, buf1, size)

	// Fill with pattern
	for i := range buf1 {
		buf1[i] = i%3 == 0
	}

	// Put it back
	PutBool(buf1)

	// Get another buffer of the same size
	buf2 := GetBool(size)
	require.Len(t, buf2, size)

	// Buffer should be zeroed
	for i, v := range buf2 {
		assert.False(t, v, "buffer element %d should be false after retrieval from pool", i)
	}
}

func TestBoolConcurrentAccess(t *testing.T) {
	const numGoroutines = 100
	const numIterations = 100
	const bufferSize = 1500

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Test concurrent gets and puts
	for range numGoroutines {
		go func() {
			defer wg.Done()

			for range numIterations {
				// Get a buffer
				buf := GetBool(bufferSize)
				assert.Len(t, buf, bufferSize)
				assert.GreaterOrEqual(t, cap(buf), bufferSize)

				// Use the buffer
				for k := 0; k < len(buf); k++ {
					buf[k] = k%2 == 0
				}

				// Put it back
				PutBool(buf)
			}
		}()
	}

	wg.Wait()
}

func TestBoolPoolZeroingBehavior(t *testing.T) {
	// Test that GetBool always returns a zeroed buffer
	size := 1000

	for i := range 10 {
		buf := GetBool(size)
		assert.Len(t, buf, size)

		// Verify all elements are false
		for j, v := range buf {
			assert.False(t, v, "iteration %d: element %d should be false", i, j)
		}

		// Modify buffer
		for j := range buf {
			buf[j] = true
		}

		// Return to pool
		PutBool(buf)
	}
}

func TestGetFloat32Multiple(t *testing.T) {
	t.Run("get multiple buffers", func(t *testing.T) {
		sizes := []int{100, 500, 1000, 2000}
		buffers := GetFloat32Multiple(sizes)

		assert.Len(t, buffers, len(sizes))
		for i, buf := range buffers {
			assert.Len(t, buf, sizes[i])
			assert.GreaterOrEqual(t, cap(buf), sizes[i])
		}

		// Clean up
		PutFloat32Multiple(buffers)
	})

	t.Run("empty sizes", func(t *testing.T) {
		buffers := GetFloat32Multiple([]int{})
		assert.Nil(t, buffers)
	})

	t.Run("nil sizes", func(t *testing.T) {
		buffers := GetFloat32Multiple(nil)
		assert.Nil(t, buffers)
	})
}

func TestPutFloat32Multiple(t *testing.T) {
	t.Run("put multiple buffers", func(t *testing.T) {
		sizes := []int{100, 500, 1000}
		buffers := GetFloat32Multiple(sizes)

		// Modify buffers
		for i, buf := range buffers {
			for j := range buf {
				buf[j] = float32(i*1000 + j)
			}
		}

		// This should not panic
		PutFloat32Multiple(buffers)
	})

	t.Run("put nil array", func(t *testing.T) {
		// This should not panic
		PutFloat32Multiple(nil)
	})

	t.Run("put array with nil elements", func(t *testing.T) {
		buffers := [][]float32{nil, GetFloat32(100), nil}
		// This should not panic
		PutFloat32Multiple(buffers)
	})
}

func TestMixedPoolUsage(t *testing.T) {
	// Test that float32 and bool pools don't interfere with each other
	const size = 5000

	// Get buffers from both pools
	f32Buf := GetFloat32(size)
	boolBuf := GetBool(size)

	assert.Len(t, f32Buf, size)
	assert.Len(t, boolBuf, size)

	// Modify both
	for i := range f32Buf {
		f32Buf[i] = float32(i)
	}
	for i := range boolBuf {
		boolBuf[i] = i%2 == 0
	}

	// Return both
	PutFloat32(f32Buf)
	PutBool(boolBuf)

	// Get again and verify independence
	f32Buf2 := GetFloat32(size)
	boolBuf2 := GetBool(size)

	assert.Len(t, f32Buf2, size)
	assert.Len(t, boolBuf2, size)

	// Bool buffer should be zeroed
	for i, v := range boolBuf2 {
		assert.False(t, v, "bool buffer element %d should be false", i)
	}

	// Clean up
	PutFloat32(f32Buf2)
	PutBool(boolBuf2)
}

// Benchmark tests for bool pools.
func BenchmarkGetBool_Small(b *testing.B) {
	for range b.N {
		buf := GetBool(100)
		PutBool(buf)
	}
}

func BenchmarkGetBool_Medium(b *testing.B) {
	for range b.N {
		buf := GetBool(2000)
		PutBool(buf)
	}
}

func BenchmarkGetBool_Large(b *testing.B) {
	for range b.N {
		buf := GetBool(10000)
		PutBool(buf)
	}
}

func BenchmarkDirectBoolAllocation_Medium(b *testing.B) {
	// Compare with direct allocation
	for range b.N {
		_ = make([]bool, 2000)
	}
}

func BenchmarkGetFloat32Multiple_Small(b *testing.B) {
	sizes := []int{100, 200, 300}
	for range b.N {
		buffers := GetFloat32Multiple(sizes)
		PutFloat32Multiple(buffers)
	}
}

func BenchmarkGetFloat32Multiple_Large(b *testing.B) {
	sizes := []int{1000, 2000, 5000, 10000}
	for range b.N {
		buffers := GetFloat32Multiple(sizes)
		PutFloat32Multiple(buffers)
	}
}
