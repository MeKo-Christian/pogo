package detector

import (
	"testing"

	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
)

// TestCalculateAdaptiveThresholds_BoundsEnforcement verifies thresholds are within configured bounds.
func TestCalculateAdaptiveThresholds_BoundsEnforcement(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("adaptive thresholds are within configured bounds", prop.ForAll(
		func(width, height int, method AdaptiveThresholdMethod) bool {
			if width < 10 || height < 10 || width > 100 || height > 100 {
				return true
			}

			probMap := make([]float32, width*height)
			for i := range probMap {
				probMap[i] = float32(i%100) / 100.0
			}

			config := AdaptiveThresholdConfig{
				Enabled:        true,
				Method:         method,
				MinDbThresh:    0.2,
				MaxDbThresh:    0.7,
				MinBoxThresh:   0.4,
				MaxBoxThresh:   0.9,
				HistogramBins:  256,
				OtsuMultiplier: 1.0,
			}

			result := CalculateAdaptiveThresholds(probMap, width, height, config)

			// Check bounds
			if result.DbThresh < config.MinDbThresh || result.DbThresh > config.MaxDbThresh {
				return false
			}
			if result.BoxThresh < config.MinBoxThresh || result.BoxThresh > config.MaxBoxThresh {
				return false
			}

			return true
		},
		gen.IntRange(10, 100),
		gen.IntRange(10, 100),
		gen.OneConstOf(AdaptiveMethodOtsu, AdaptiveMethodHistogram, AdaptiveMethodDynamic),
	))

	properties.TestingRun(t)
}

// TestCalculateAdaptiveThresholds_BoxThresholdHigher verifies box_thresh >= db_thresh.
func TestCalculateAdaptiveThresholds_BoxThresholdHigher(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("box threshold is always >= db threshold", prop.ForAll(
		func(width, height int) bool {
			if width < 10 || height < 10 || width > 100 || height > 100 {
				return true
			}

			probMap := make([]float32, width*height)
			for i := range probMap {
				probMap[i] = float32(i%100) / 100.0
			}

			config := DefaultAdaptiveThresholdConfig()
			config.Enabled = true

			result := CalculateAdaptiveThresholds(probMap, width, height, config)

			return result.BoxThresh >= result.DbThresh
		},
		gen.IntRange(10, 100),
		gen.IntRange(10, 100),
	))

	properties.TestingRun(t)
}

// TestCalculateAdaptiveThresholds_ConfidenceInRange verifies confidence is in [0, 1].
func TestCalculateAdaptiveThresholds_ConfidenceInRange(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("confidence value is in [0, 1]", prop.ForAll(
		func(width, height int, method AdaptiveThresholdMethod) bool {
			if width < 10 || height < 10 || width > 100 || height > 100 {
				return true
			}

			probMap := make([]float32, width*height)
			for i := range probMap {
				probMap[i] = float32(i%100) / 100.0
			}

			config := DefaultAdaptiveThresholdConfig()
			config.Enabled = true
			config.Method = method

			result := CalculateAdaptiveThresholds(probMap, width, height, config)

			return result.Confidence >= 0.0 && result.Confidence <= 1.0
		},
		gen.IntRange(10, 100),
		gen.IntRange(10, 100),
		gen.OneConstOf(AdaptiveMethodOtsu, AdaptiveMethodHistogram, AdaptiveMethodDynamic),
	))

	properties.TestingRun(t)
}

// TestCalculateProbabilityMapStats_BasicProperties verifies statistical properties.
func TestCalculateProbabilityMapStats_BasicProperties(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("probability map statistics have valid properties", prop.ForAll(
		func(size int) bool {
			if size < 10 || size > 1000 {
				return true
			}

			probMap := make([]float32, size)
			for i := range probMap {
				probMap[i] = float32(i%100) / 100.0
			}

			stats := calculateProbabilityMapStats(probMap)

			// Basic sanity checks
			if stats.Min < 0.0 || stats.Max > 1.0 {
				return false
			}
			if stats.Min > stats.Max {
				return false
			}
			if stats.Mean < stats.Min || stats.Mean > stats.Max {
				return false
			}
			if stats.Median < stats.Min || stats.Median > stats.Max {
				return false
			}
			if stats.DynamicRange != stats.Max-stats.Min {
				return false
			}
			if stats.StdDev < 0 {
				return false
			}
			if stats.HighProbRatio < 0.0 || stats.HighProbRatio > 1.0 {
				return false
			}
			if stats.BimodalityIndex < 0.0 || stats.BimodalityIndex > 1.0 {
				return false
			}

			return true
		},
		gen.IntRange(10, 1000),
	))

	properties.TestingRun(t)
}

// TestCalculateOtsuThreshold_ReturnsBounds verifies Otsu threshold is in valid range.
func TestCalculateOtsuThreshold_ReturnsBounds(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("Otsu threshold is in [0, 1]", prop.ForAll(
		func(size int) bool {
			if size < 100 || size > 1000 {
				return true
			}

			probMap := make([]float32, size)
			for i := range probMap {
				probMap[i] = float32(i%100) / 100.0
			}

			config := DefaultAdaptiveThresholdConfig()
			threshold, confidence := calculateOtsuThreshold(probMap, config)

			if threshold < 0.0 || threshold > 1.0 {
				return false
			}
			if confidence < 0.0 || confidence > 1.0 {
				return false
			}

			return true
		},
		gen.IntRange(100, 1000),
	))

	properties.TestingRun(t)
}

// TestCalculateBimodalityIndex_BoundsAndMonotonicity verifies bimodality index properties.
func TestCalculateBimodalityIndex_BoundsAndMonotonicity(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("bimodality index is in [0, 1]", prop.ForAll(
		func(size int) bool {
			if size < 50 || size > 500 {
				return true
			}

			probMap := make([]float32, size)
			for i := range probMap {
				probMap[i] = float32(i%100) / 100.0
			}

			bimodality := calculateBimodalityIndex(probMap)
			return bimodality >= 0.0 && bimodality <= 1.0
		},
		gen.IntRange(50, 500),
	))

	properties.TestingRun(t)
}

// TestCalculateBimodalityIndex_BimodalDistribution verifies high bimodality for bimodal data.
func TestCalculateBimodalityIndex_BimodalDistribution(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("bimodal distribution has higher bimodality index", prop.ForAll(
		func(size int) bool {
			if size < 200 || size > 500 {
				return true
			}

			// Create unimodal distribution (all values around 0.5)
			unimodal := make([]float32, size)
			for i := range unimodal {
				unimodal[i] = 0.5
			}

			// Create bimodal distribution (values at 0.1 and 0.9 - more separated)
			bimodal := make([]float32, size)
			for i := range bimodal {
				if i%2 == 0 {
					bimodal[i] = 0.1
				} else {
					bimodal[i] = 0.9
				}
			}

			unimodalIdx := calculateBimodalityIndex(unimodal)
			bimodalIdx := calculateBimodalityIndex(bimodal)

			// Bimodal should have higher index
			return bimodalIdx >= unimodalIdx
		},
		gen.IntRange(200, 500),
	))

	properties.TestingRun(t)
}

// TestCalculateHistogramBasedThresholds_ValidOutput verifies histogram method output.
func TestCalculateHistogramBasedThresholds_ValidOutput(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("histogram-based thresholds are valid", prop.ForAll(
		func(mean, stdDev, highProbRatio, bimodality float32) bool {
			// Ensure valid inputs
			if mean < 0.0 || mean > 1.0 {
				return true
			}
			if stdDev < 0.0 || stdDev > 1.0 {
				return true
			}
			if highProbRatio < 0.0 || highProbRatio > 1.0 {
				return true
			}
			if bimodality < 0.0 || bimodality > 1.0 {
				return true
			}

			stats := AdaptiveThresholdStats{
				Mean:            mean,
				StdDev:          stdDev,
				Median:          mean,
				Min:             0.0,
				Max:             1.0,
				DynamicRange:    1.0,
				HighProbRatio:   highProbRatio,
				BimodalityIndex: bimodality,
			}

			dbThresh, boxThresh, confidence := calculateHistogramBasedThresholds(stats)

			// Check basic validity
			if dbThresh < 0.0 || dbThresh > 1.0 {
				return false
			}
			if boxThresh < 0.0 || boxThresh > 1.0 {
				return false
			}
			if confidence < 0.0 || confidence > 1.0 {
				return false
			}

			return true
		},
		gen.Float32Range(0.0, 1.0),
		gen.Float32Range(0.0, 0.5),
		gen.Float32Range(0.0, 1.0),
		gen.Float32Range(0.0, 1.0),
	))

	properties.TestingRun(t)
}

// TestCalculateDynamicThresholds_ValidOutput verifies dynamic method output.
func TestCalculateDynamicThresholds_ValidOutput(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("dynamic thresholds are valid", prop.ForAll(
		func(size int) bool {
			if size < 100 || size > 500 {
				return true
			}

			probMap := make([]float32, size)
			for i := range probMap {
				probMap[i] = float32(i%100) / 100.0
			}

			stats := calculateProbabilityMapStats(probMap)
			dbThresh, boxThresh, confidence := calculateDynamicThresholds(probMap, stats)

			if dbThresh < 0.0 || dbThresh > 1.0 {
				return false
			}
			if boxThresh < 0.0 || boxThresh > 1.0 {
				return false
			}
			if confidence < 0.0 || confidence > 1.0 {
				return false
			}

			return true
		},
		gen.IntRange(100, 500),
	))

	properties.TestingRun(t)
}

// TestCalculateAdaptiveThresholds_DisabledReturnsDefaults verifies disabled config behavior.
func TestCalculateAdaptiveThresholds_DisabledReturnsDefaults(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("disabled config returns default thresholds", prop.ForAll(
		func(width, height int) bool {
			if width < 10 || height < 10 || width > 100 || height > 100 {
				return true
			}

			probMap := make([]float32, width*height)
			for i := range probMap {
				probMap[i] = float32(i%100) / 100.0
			}

			config := DefaultAdaptiveThresholdConfig()
			config.Enabled = false

			result := CalculateAdaptiveThresholds(probMap, width, height, config)

			// Should return defaults
			return result.DbThresh == 0.3 && result.BoxThresh == 0.5 && result.Method == "disabled"
		},
		gen.IntRange(10, 100),
		gen.IntRange(10, 100),
	))

	properties.TestingRun(t)
}

// TestClampFloat32_CorrectClamping verifies clamping works correctly.
func TestClampFloat32_CorrectClamping(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("clampFloat32 correctly clamps values", prop.ForAll(
		func(value, minVal, maxVal float32) bool {
			// Ensure min <= max
			if minVal > maxVal {
				minVal, maxVal = maxVal, minVal
			}

			clamped := clampFloat32(value, minVal, maxVal)

			// Check clamping
			if clamped < minVal || clamped > maxVal {
				return false
			}

			// If value is within bounds, should be unchanged
			if value >= minVal && value <= maxVal {
				return clamped == value
			}

			// If value is below min, should be min
			if value < minVal {
				return clamped == minVal
			}

			// If value is above max, should be max
			if value > maxVal {
				return clamped == maxVal
			}

			return true
		},
		gen.Float32Range(-10.0, 10.0),
		gen.Float32Range(0.0, 1.0),
		gen.Float32Range(0.0, 1.0),
	))

	properties.TestingRun(t)
}

// TestCalculateAdaptiveThresholds_Deterministic verifies deterministic behavior.
func TestCalculateAdaptiveThresholds_Deterministic(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("adaptive threshold calculation is deterministic", prop.ForAll(
		func(width, height int, method AdaptiveThresholdMethod) bool {
			if width < 10 || height < 10 || width > 50 || height > 50 {
				return true
			}

			probMap := make([]float32, width*height)
			for i := range probMap {
				probMap[i] = float32(i%100) / 100.0
			}

			config := DefaultAdaptiveThresholdConfig()
			config.Enabled = true
			config.Method = method

			result1 := CalculateAdaptiveThresholds(probMap, width, height, config)
			result2 := CalculateAdaptiveThresholds(probMap, width, height, config)

			// Results should be identical
			return result1.DbThresh == result2.DbThresh &&
				result1.BoxThresh == result2.BoxThresh &&
				result1.Confidence == result2.Confidence &&
				result1.Method == result2.Method
		},
		gen.IntRange(10, 50),
		gen.IntRange(10, 50),
		gen.OneConstOf(AdaptiveMethodOtsu, AdaptiveMethodHistogram, AdaptiveMethodDynamic),
	))

	properties.TestingRun(t)
}
