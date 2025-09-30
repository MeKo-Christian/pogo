package detector

import (
	"testing"

	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
)

// TestApplyMorphologicalOperation_BoundsPreservation verifies values stay in [0, 1].
func TestApplyMorphologicalOperation_BoundsPreservation(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("morphological operations preserve value bounds [0, 1]", prop.ForAll(
		func(width, height, kernelSize, iterations int, op MorphologicalOp) bool {
			if width < 5 || height < 5 || width > 50 || height > 50 {
				return true
			}
			if kernelSize < 3 || kernelSize > 7 {
				return true
			}
			if iterations < 1 || iterations > 3 {
				return true
			}

			// Generate probability map
			probMap := make([]float32, width*height)
			for i := range probMap {
				probMap[i] = float32(i%100) / 100.0
			}

			config := MorphConfig{
				Operation:  op,
				KernelSize: kernelSize,
				Iterations: iterations,
			}

			result := ApplyMorphologicalOperation(probMap, width, height, config)

			// Check all values are in [0, 1]
			for _, val := range result {
				if val < 0.0 || val > 1.0 {
					return false
				}
			}
			return true
		},
		gen.IntRange(5, 50),
		gen.IntRange(5, 50),
		gen.IntRange(3, 7),
		gen.IntRange(1, 3),
		gen.OneConstOf(MorphDilate, MorphErode, MorphOpening, MorphClosing, MorphSmooth),
	))

	properties.TestingRun(t)
}

// TestApplyMorphologicalOperation_LengthPreservation verifies output length equals input.
func TestApplyMorphologicalOperation_LengthPreservation(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("morphological operations preserve map dimensions", prop.ForAll(
		func(width, height int, op MorphologicalOp) bool {
			if width < 5 || height < 5 || width > 50 || height > 50 {
				return true
			}

			probMap := make([]float32, width*height)
			for i := range probMap {
				probMap[i] = 0.5
			}

			config := MorphConfig{
				Operation:  op,
				KernelSize: 3,
				Iterations: 1,
			}

			result := ApplyMorphologicalOperation(probMap, width, height, config)
			return len(result) == len(probMap)
		},
		gen.IntRange(5, 50),
		gen.IntRange(5, 50),
		gen.OneConstOf(MorphDilate, MorphErode, MorphOpening, MorphClosing, MorphSmooth),
	))

	properties.TestingRun(t)
}

// TestDilateFloat32_MonotonicIncrease verifies dilation increases or maintains values.
func TestDilateFloat32_MonotonicIncrease(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("dilation increases or maintains probability values", prop.ForAll(
		func(width, height int) bool {
			if width < 5 || height < 5 || width > 30 || height > 30 {
				return true
			}

			// Create a sparse probability map
			probMap := make([]float32, width*height)
			for i := range probMap {
				if i%(width+1) == 0 {
					probMap[i] = 0.8
				} else {
					probMap[i] = 0.2
				}
			}

			dilated := dilateFloat32(probMap, width, height, 3)

			// Check that dilation increased or maintained values
			increased := 0
			for i := range probMap {
				if dilated[i] > probMap[i] {
					increased++
				}
				// Dilated value should never be less than original
				if dilated[i] < probMap[i] {
					return false
				}
			}

			// At least some values should have increased
			return increased > 0
		},
		gen.IntRange(5, 30),
		gen.IntRange(5, 30),
	))

	properties.TestingRun(t)
}

// TestErodeFloat32_MonotonicDecrease verifies erosion decreases or maintains values.
func TestErodeFloat32_MonotonicDecrease(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("erosion decreases or maintains probability values", prop.ForAll(
		func(width, height int) bool {
			if width < 5 || height < 5 || width > 30 || height > 30 {
				return true
			}

			// Create a dense probability map
			probMap := make([]float32, width*height)
			for i := range probMap {
				probMap[i] = 0.8
			}
			// Add some low-probability pixels
			for i := 0; i < len(probMap); i += (width + 1) {
				probMap[i] = 0.2
			}

			eroded := erodeFloat32(probMap, width, height, 3)

			// Check that erosion decreased or maintained values
			decreased := 0
			for i := range probMap {
				if eroded[i] < probMap[i] {
					decreased++
				}
				// Eroded value should never be greater than original
				if eroded[i] > probMap[i] {
					return false
				}
			}

			// At least some values should have decreased
			return decreased > 0
		},
		gen.IntRange(5, 30),
		gen.IntRange(5, 30),
	))

	properties.TestingRun(t)
}

// TestMorphOpening_Idempotence verifies opening is idempotent.
func TestMorphOpening_Idempotence(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("morphological opening is idempotent", prop.ForAll(
		func(width, height int) bool {
			if width < 10 || height < 10 || width > 30 || height > 30 {
				return true
			}

			probMap := make([]float32, width*height)
			for i := range probMap {
				probMap[i] = float32(i%2) * 0.8
			}

			config := MorphConfig{
				Operation:  MorphOpening,
				KernelSize: 3,
				Iterations: 1,
			}

			result1 := ApplyMorphologicalOperation(probMap, width, height, config)
			result2 := ApplyMorphologicalOperation(result1, width, height, config)

			// Second application should not change result significantly
			for i := range result1 {
				if result1[i] != result2[i] {
					// Allow small floating point differences
					diff := result1[i] - result2[i]
					if diff < -1e-6 || diff > 1e-6 {
						return false
					}
				}
			}
			return true
		},
		gen.IntRange(10, 30),
		gen.IntRange(10, 30),
	))

	properties.TestingRun(t)
}

// TestMorphClosing_Idempotence verifies closing is idempotent.
func TestMorphClosing_Idempotence(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("morphological closing is idempotent", prop.ForAll(
		func(width, height int) bool {
			if width < 10 || height < 10 || width > 30 || height > 30 {
				return true
			}

			probMap := make([]float32, width*height)
			for i := range probMap {
				probMap[i] = float32(i%2) * 0.8
			}

			config := MorphConfig{
				Operation:  MorphClosing,
				KernelSize: 3,
				Iterations: 1,
			}

			result1 := ApplyMorphologicalOperation(probMap, width, height, config)
			result2 := ApplyMorphologicalOperation(result1, width, height, config)

			// Second application should not change result significantly
			for i := range result1 {
				if result1[i] != result2[i] {
					// Allow small floating point differences
					diff := result1[i] - result2[i]
					if diff < -1e-6 || diff > 1e-6 {
						return false
					}
				}
			}
			return true
		},
		gen.IntRange(10, 30),
		gen.IntRange(10, 30),
	))

	properties.TestingRun(t)
}

// TestSmoothFloat32_ReducesVariance verifies smoothing reduces variance.
func TestSmoothFloat32_ReducesVariance(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("smoothing reduces or maintains variance", prop.ForAll(
		func(width, height int) bool {
			if width < 10 || height < 10 || width > 30 || height > 30 {
				return true
			}

			// Create noisy probability map
			probMap := make([]float32, width*height)
			for i := range probMap {
				probMap[i] = float32(i%3) / 3.0
			}

			smoothed := smoothFloat32(probMap, width, height, 3)

			// Calculate variance of original
			var sum, sumSq float32
			for _, val := range probMap {
				sum += val
				sumSq += val * val
			}
			mean := sum / float32(len(probMap))
			variance := (sumSq / float32(len(probMap))) - mean*mean

			// Calculate variance of smoothed
			var sumSmooth, sumSqSmooth float32
			for _, val := range smoothed {
				sumSmooth += val
				sumSqSmooth += val * val
			}
			meanSmooth := sumSmooth / float32(len(smoothed))
			varianceSmooth := (sumSqSmooth / float32(len(smoothed))) - meanSmooth*meanSmooth

			// Smoothed variance should be less than or equal to original
			return varianceSmooth <= variance+1e-5 // small tolerance for floating point
		},
		gen.IntRange(10, 30),
		gen.IntRange(10, 30),
	))

	properties.TestingRun(t)
}

// TestMorphOpening_RemovesSmallNoise verifies opening removes small bright regions.
func TestMorphOpening_RemovesSmallNoise(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("morphological opening removes small bright regions", prop.ForAll(
		func(width, height int) bool {
			if width < 15 || height < 15 || width > 30 || height > 30 {
				return true
			}

			// Create map with small isolated high-probability pixels (noise)
			probMap := make([]float32, width*height)
			for i := range probMap {
				probMap[i] = 0.1
			}
			// Add isolated bright pixels
			probMap[width*5+5] = 0.9
			probMap[width*10+15] = 0.9
			probMap[width*20+10] = 0.9

			config := MorphConfig{
				Operation:  MorphOpening,
				KernelSize: 3,
				Iterations: 1,
			}

			result := ApplyMorphologicalOperation(probMap, width, height, config)

			// Isolated pixels should be removed (values should be reduced)
			removed := 0
			for i := range probMap {
				if probMap[i] > 0.8 && result[i] < 0.8 {
					removed++
				}
			}

			// At least some bright pixels should have been removed
			return removed > 0
		},
		gen.IntRange(15, 30),
		gen.IntRange(15, 30),
	))

	properties.TestingRun(t)
}

// TestMorphClosing_FillsGaps verifies closing fills small gaps.
func TestMorphClosing_FillsGaps(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("morphological closing fills small dark gaps", prop.ForAll(
		func(width, height int) bool {
			if width < 15 || height < 15 || width > 30 || height > 30 {
				return true
			}

			// Create map with bright region and small gaps
			probMap := make([]float32, width*height)
			for y := 5; y < height-5; y++ {
				for x := 5; x < width-5; x++ {
					probMap[y*width+x] = 0.9
				}
			}
			// Add small gaps
			probMap[10*width+10] = 0.1
			probMap[15*width+15] = 0.1
			probMap[20*width+10] = 0.1

			config := MorphConfig{
				Operation:  MorphClosing,
				KernelSize: 3,
				Iterations: 1,
			}

			result := ApplyMorphologicalOperation(probMap, width, height, config)

			// Gaps should be filled (values should be increased)
			filled := 0
			for i := range probMap {
				if probMap[i] < 0.2 && result[i] > 0.2 {
					filled++
				}
			}

			// At least some gaps should have been filled
			return filled > 0
		},
		gen.IntRange(15, 30),
		gen.IntRange(15, 30),
	))

	properties.TestingRun(t)
}

// TestMorphNone_IsIdentity verifies MorphNone returns unchanged input.
func TestMorphNone_IsIdentity(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("MorphNone operation is identity", prop.ForAll(
		func(width, height int) bool {
			if width < 5 || height < 5 || width > 50 || height > 50 {
				return true
			}

			probMap := make([]float32, width*height)
			for i := range probMap {
				probMap[i] = float32(i%100) / 100.0
			}

			config := MorphConfig{
				Operation:  MorphNone,
				KernelSize: 3,
				Iterations: 1,
			}

			result := ApplyMorphologicalOperation(probMap, width, height, config)

			// Result should be unchanged
			if len(result) != len(probMap) {
				return false
			}

			for i := range probMap {
				if result[i] != probMap[i] {
					return false
				}
			}
			return true
		},
		gen.IntRange(5, 50),
		gen.IntRange(5, 50),
	))

	properties.TestingRun(t)
}

// TestKernelSize_AffectsResult verifies larger kernel size has more effect.
func TestKernelSize_AffectsResult(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("larger kernel size has stronger morphological effect", prop.ForAll(
		func(width, height int) bool {
			if width < 20 || height < 20 || width > 40 || height > 40 {
				return true
			}

			// Create a map with a small bright region
			probMap := make([]float32, width*height)
			for i := range probMap {
				probMap[i] = 0.1
			}
			// Add a 5x5 bright region
			for y := height/2 - 2; y <= height/2+2; y++ {
				for x := width/2 - 2; x <= width/2+2; x++ {
					probMap[y*width+x] = 0.9
				}
			}

			config3 := MorphConfig{Operation: MorphDilate, KernelSize: 3, Iterations: 1}
			config5 := MorphConfig{Operation: MorphDilate, KernelSize: 5, Iterations: 1}

			result3 := ApplyMorphologicalOperation(probMap, width, height, config3)
			result5 := ApplyMorphologicalOperation(probMap, width, height, config5)

			// Count pixels with high probability in each result
			count3, count5 := 0, 0
			for i := range result3 {
				if result3[i] > 0.5 {
					count3++
				}
				if result5[i] > 0.5 {
					count5++
				}
			}

			// Larger kernel should dilate more (more high-probability pixels)
			return count5 >= count3
		},
		gen.IntRange(20, 40),
		gen.IntRange(20, 40),
	))

	properties.TestingRun(t)
}
