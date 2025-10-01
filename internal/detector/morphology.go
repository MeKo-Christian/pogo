package detector

import (
	"github.com/MeKo-Tech/pogo/internal/mempool"
)

// MorphologicalOp represents the type of morphological operation to perform.
type MorphologicalOp int

const (
	MorphNone MorphologicalOp = iota
	MorphDilate
	MorphErode
	MorphOpening // Erode then Dilate - removes small noise
	MorphClosing // Dilate then Erode - fills gaps
	MorphSmooth  // Gaussian-like smoothing
)

// MorphConfig holds configuration for morphological operations.
type MorphConfig struct {
	Operation  MorphologicalOp
	KernelSize int // Size of the morphological kernel (e.g., 3 for 3x3)
	Iterations int // Number of times to apply the operation
}

// DefaultMorphConfig returns default morphological operation configuration.
func DefaultMorphConfig() MorphConfig {
	return MorphConfig{
		Operation:  MorphNone,
		KernelSize: 3,
		Iterations: 1,
	}
}

// ApplyMorphologicalOperation applies morphological operations to a probability map.
// It uses memory pooling to reduce allocations.
func ApplyMorphologicalOperation(probMap []float32, width, height int, config MorphConfig) []float32 {
	if config.Operation == MorphNone || config.KernelSize <= 0 || config.Iterations <= 0 {
		return probMap
	}

	result := mempool.GetFloat32(len(probMap))
	copy(result, probMap)

	for range config.Iterations {
		switch config.Operation {
		case MorphDilate:
			result = dilateFloat32Pooled(result, width, height, config.KernelSize)
		case MorphErode:
			result = erodeFloat32Pooled(result, width, height, config.KernelSize)
		case MorphOpening:
			result = erodeFloat32Pooled(result, width, height, config.KernelSize)
			result = dilateFloat32Pooled(result, width, height, config.KernelSize)
		case MorphClosing:
			result = dilateFloat32Pooled(result, width, height, config.KernelSize)
			result = erodeFloat32Pooled(result, width, height, config.KernelSize)
		case MorphSmooth:
			result = smoothFloat32Pooled(result, width, height, config.KernelSize)
		}
	}

	return result
}

// dilateFloat32 performs dilation on a float32 probability map.
// Dilation expands bright regions (high probability areas).
func dilateFloat32(probMap []float32, width, height, kernelSize int) []float32 {
	if kernelSize <= 1 {
		return probMap
	}

	result := make([]float32, len(probMap))
	half := kernelSize / 2

	for y := range height {
		for x := range width {
			maxVal := float32(0.0)
			applyKernel(probMap, width, height, x, y, half, func(idx int) {
				if probMap[idx] > maxVal {
					maxVal = probMap[idx]
				}
			})
			result[y*width+x] = maxVal
		}
	}

	return result
}

// erodeFloat32 performs erosion on a float32 probability map.
// Erosion shrinks bright regions (high probability areas).
func erodeFloat32(probMap []float32, width, height, kernelSize int) []float32 {
	if kernelSize <= 1 {
		return probMap
	}

	result := make([]float32, len(probMap))
	half := kernelSize / 2

	for y := range height {
		for x := range width {
			minVal := float32(1.0)
			applyKernel(probMap, width, height, x, y, half, func(idx int) {
				if probMap[idx] < minVal {
					minVal = probMap[idx]
				}
			})
			result[y*width+x] = minVal
		}
	}

	return result
}

// smoothFloat32 performs Gaussian-like smoothing on a float32 probability map.
// This helps reduce noise while preserving overall text structure.
func smoothFloat32(probMap []float32, width, height, kernelSize int) []float32 {
	if kernelSize <= 1 {
		return probMap
	}

	result := make([]float32, len(probMap))
	half := kernelSize / 2

	// Simple box filter for smoothing (could be improved with Gaussian weights)

	for y := range height {
		for x := range width {
			sum := float32(0.0)
			count := float32(0.0)
			applyKernel(probMap, width, height, x, y, half, func(idx int) {
				sum += probMap[idx]
				count += 1.0
			})

			if count > 0 {
				result[y*width+x] = sum / count
			} else {
				result[y*width+x] = probMap[y*width+x]
			}
		}
	}

	return result
}

// applyKernel applies a kernel operation to neighboring pixels within bounds.
func applyKernel(_ []float32, width, height, x, y, half int, operation func(idx int)) {
	for ky := -half; ky <= half; ky++ {
		for kx := -half; kx <= half; kx++ {
			nx, ny := x+kx, y+ky
			if nx >= 0 && nx < width && ny >= 0 && ny < height {
				idx := ny*width + nx
				operation(idx)
			}
		}
	}
}

// dilateFloat32Pooled performs dilation using memory pooling.
// The input buffer is returned to the pool and a new buffer is returned.
func dilateFloat32Pooled(probMap []float32, width, height, kernelSize int) []float32 {
	if kernelSize <= 1 {
		return probMap
	}

	result := mempool.GetFloat32(len(probMap))
	half := kernelSize / 2

	for y := range height {
		for x := range width {
			maxVal := float32(0.0)
			applyKernel(probMap, width, height, x, y, half, func(idx int) {
				if probMap[idx] > maxVal {
					maxVal = probMap[idx]
				}
			})
			result[y*width+x] = maxVal
		}
	}

	// Return input buffer to pool
	mempool.PutFloat32(probMap)
	return result
}

// erodeFloat32Pooled performs erosion using memory pooling.
// The input buffer is returned to the pool and a new buffer is returned.
func erodeFloat32Pooled(probMap []float32, width, height, kernelSize int) []float32 {
	if kernelSize <= 1 {
		return probMap
	}

	result := mempool.GetFloat32(len(probMap))
	half := kernelSize / 2

	for y := range height {
		for x := range width {
			minVal := float32(1.0)
			applyKernel(probMap, width, height, x, y, half, func(idx int) {
				if probMap[idx] < minVal {
					minVal = probMap[idx]
				}
			})
			result[y*width+x] = minVal
		}
	}

	// Return input buffer to pool
	mempool.PutFloat32(probMap)
	return result
}

// smoothFloat32Pooled performs smoothing using memory pooling.
// The input buffer is returned to the pool and a new buffer is returned.
func smoothFloat32Pooled(probMap []float32, width, height, kernelSize int) []float32 {
	if kernelSize <= 1 {
		return probMap
	}

	result := mempool.GetFloat32(len(probMap))
	half := kernelSize / 2

	for y := range height {
		for x := range width {
			sum := float32(0.0)
			count := float32(0.0)
			applyKernel(probMap, width, height, x, y, half, func(idx int) {
				sum += probMap[idx]
				count += 1.0
			})

			if count > 0 {
				result[y*width+x] = sum / count
			} else {
				result[y*width+x] = probMap[y*width+x]
			}
		}
	}

	// Return input buffer to pool
	mempool.PutFloat32(probMap)
	return result
}
