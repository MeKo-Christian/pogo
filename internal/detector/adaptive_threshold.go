package detector

import (
	"math"
	"sort"
)

// AdaptiveThresholdMethod represents different methods for calculating adaptive thresholds.
type AdaptiveThresholdMethod int

const (
	AdaptiveMethodOtsu AdaptiveThresholdMethod = iota
	AdaptiveMethodHistogram
	AdaptiveMethodDynamic
)

// AdaptiveThresholdConfig holds configuration for adaptive threshold calculation.
type AdaptiveThresholdConfig struct {
	Enabled        bool                    // Enable adaptive threshold calculation
	Method         AdaptiveThresholdMethod // Method to use for calculation
	MinDbThresh    float32                 // Minimum allowed db_thresh (default: 0.1)
	MaxDbThresh    float32                 // Maximum allowed db_thresh (default: 0.8)
	MinBoxThresh   float32                 // Minimum allowed box_thresh (default: 0.3)
	MaxBoxThresh   float32                 // Maximum allowed box_thresh (default: 0.9)
	HistogramBins  int                     // Number of bins for histogram analysis (default: 256)
	OtsuMultiplier float32                 // Multiplier for Otsu threshold (default: 1.0)
}

// DefaultAdaptiveThresholdConfig returns default adaptive threshold configuration.
func DefaultAdaptiveThresholdConfig() AdaptiveThresholdConfig {
	return AdaptiveThresholdConfig{
		Enabled:        false,
		Method:         AdaptiveMethodHistogram,
		MinDbThresh:    0.1,
		MaxDbThresh:    0.8,
		MinBoxThresh:   0.3,
		MaxBoxThresh:   0.9,
		HistogramBins:  256,
		OtsuMultiplier: 1.0,
	}
}

// AdaptiveThresholds holds the calculated adaptive thresholds.
type AdaptiveThresholds struct {
	DbThresh   float32                // Calculated db_thresh value
	BoxThresh  float32                // Calculated box_thresh value
	Method     string                 // Method used for calculation
	Confidence float32                // Confidence in the calculated thresholds (0-1)
	Statistics AdaptiveThresholdStats // Statistics used for calculation
}

// AdaptiveThresholdStats holds statistics about the probability map used for threshold calculation.
type AdaptiveThresholdStats struct {
	Mean            float32 // Mean probability value
	StdDev          float32 // Standard deviation
	Median          float32 // Median probability value
	Min             float32 // Minimum probability value
	Max             float32 // Maximum probability value
	DynamicRange    float32 // Max - Min
	HighProbRatio   float32 // Ratio of pixels above 0.5
	BimodalityIndex float32 // Measure of bimodality (0-1, higher is more bimodal)
}

// CalculateAdaptiveThresholds analyzes a probability map and calculates optimal thresholds.
func CalculateAdaptiveThresholds(probMap []float32, width, height int,
	config AdaptiveThresholdConfig) AdaptiveThresholds {
	if !config.Enabled || len(probMap) != width*height || len(probMap) == 0 {
		return AdaptiveThresholds{
			DbThresh:   0.3, // Default values
			BoxThresh:  0.5,
			Method:     "disabled",
			Confidence: 0.0,
		}
	}

	// Calculate probability map statistics
	stats := calculateProbabilityMapStats(probMap)

	var dbThresh, boxThresh float32
	var methodName string
	var confidence float32

	switch config.Method {
	case AdaptiveMethodOtsu:
		dbThresh, confidence = calculateOtsuThreshold(probMap, config)
		boxThresh = dbThresh + 0.2 // Box threshold slightly higher than db threshold
		methodName = "otsu"
	case AdaptiveMethodHistogram:
		dbThresh, boxThresh, confidence = calculateHistogramBasedThresholds(stats)
		methodName = "histogram"
	case AdaptiveMethodDynamic:
		dbThresh, boxThresh, confidence = calculateDynamicThresholds(probMap, stats)
		methodName = "dynamic"
	default:
		dbThresh, boxThresh = 0.3, 0.5
		confidence = 0.0
		methodName = "fallback"
	}

	// Clamp thresholds to configured bounds
	dbThresh = clampFloat32(dbThresh, config.MinDbThresh, config.MaxDbThresh)
	boxThresh = clampFloat32(boxThresh, config.MinBoxThresh, config.MaxBoxThresh)

	// Ensure box_thresh >= db_thresh
	if boxThresh < dbThresh {
		boxThresh = dbThresh + 0.1
		if boxThresh > config.MaxBoxThresh {
			boxThresh = config.MaxBoxThresh
		}
	}

	return AdaptiveThresholds{
		DbThresh:   dbThresh,
		BoxThresh:  boxThresh,
		Method:     methodName,
		Confidence: confidence,
		Statistics: stats,
	}
}

// calculateProbabilityMapStats computes statistical measures of the probability map.
func calculateProbabilityMapStats(probMap []float32) AdaptiveThresholdStats {
	if len(probMap) == 0 {
		return AdaptiveThresholdStats{}
	}

	// Create a copy for sorting (to find median)
	sortedProbs := make([]float32, len(probMap))
	copy(sortedProbs, probMap)
	sort.Slice(sortedProbs, func(i, j int) bool {
		return sortedProbs[i] < sortedProbs[j]
	})

	// Basic statistics
	var sum float32
	minVal := sortedProbs[0]
	maxVal := sortedProbs[len(sortedProbs)-1]
	median := sortedProbs[len(sortedProbs)/2]

	var highProbCount int
	for _, prob := range probMap {
		sum += prob
		if prob > 0.5 {
			highProbCount++
		}
	}

	mean := sum / float32(len(probMap))
	highProbRatio := float32(highProbCount) / float32(len(probMap))

	// Calculate standard deviation
	var variance float32
	for _, prob := range probMap {
		diff := prob - mean
		variance += diff * diff
	}
	variance /= float32(len(probMap))
	stdDev := float32(math.Sqrt(float64(variance)))

	// Calculate bimodality index using histogram
	bimodality := calculateBimodalityIndex(probMap)

	return AdaptiveThresholdStats{
		Mean:            mean,
		StdDev:          stdDev,
		Median:          median,
		Min:             minVal,
		Max:             maxVal,
		DynamicRange:    maxVal - minVal,
		HighProbRatio:   highProbRatio,
		BimodalityIndex: bimodality,
	}
}

// calculateBimodalityIndex calculates a measure of bimodality in the probability distribution.
func calculateBimodalityIndex(probMap []float32) float32 {
	const bins = 50
	histogram := make([]int, bins)

	// Build histogram
	for _, prob := range probMap {
		bin := int(prob * float32(bins-1))
		if bin >= bins {
			bin = bins - 1
		}
		histogram[bin]++
	}

	// Find peaks in the histogram
	peaks := 0
	for i := 1; i < bins-1; i++ {
		if histogram[i] > histogram[i-1] && histogram[i] > histogram[i+1] {
			// Only count significant peaks (at least 1% of total pixels)
			if histogram[i] > len(probMap)/100 {
				peaks++
			}
		}
	}

	// Bimodality index: closer to 1.0 means more bimodal
	if peaks >= 2 {
		return 1.0
	} else if peaks == 1 {
		return 0.5
	}
	return 0.0
}

// calculateOtsuThreshold implements Otsu's method for threshold selection.
func calculateOtsuThreshold(probMap []float32, config AdaptiveThresholdConfig) (float32, float32) {
	if len(probMap) == 0 {
		return 0.3, 0.5
	}

	const bins = 256
	histogram := make([]int, bins)
	totalPixels := len(probMap)

	// Build histogram
	for _, prob := range probMap {
		bin := int(prob * float32(bins-1))
		if bin >= bins {
			bin = bins - 1
		}
		histogram[bin]++
	}

	// Calculate Otsu threshold
	var totalMean float32
	for i := range bins {
		totalMean += float32(i) * float32(histogram[i])
	}
	totalMean /= float32(totalPixels)

	var maxVariance float32
	bestThreshold := 0
	var sumB float32
	wB := 0

	for t := range bins {
		wB += histogram[t]
		if wB == 0 {
			continue
		}

		wF := totalPixels - wB
		if wF == 0 {
			break
		}

		sumB += float32(t) * float32(histogram[t])
		meanB := sumB / float32(wB)
		meanF := (totalMean*float32(totalPixels) - sumB) / float32(wF)

		// Between-class variance
		variance := float32(wB) * float32(wF) * (meanB - meanF) * (meanB - meanF)

		if variance > maxVariance {
			maxVariance = variance
			bestThreshold = t
		}
	}

	otsuThresh := float32(bestThreshold) / float32(bins-1) * config.OtsuMultiplier
	confidence := float32(math.Min(float64(maxVariance/float32(totalPixels*totalPixels)), 1.0))

	return otsuThresh, confidence
}

// calculateHistogramBasedThresholds uses histogram analysis to find optimal thresholds.
func calculateHistogramBasedThresholds(stats AdaptiveThresholdStats) (float32, float32, float32) {
	// For histogram-based method, we use the mean and standard deviation to set thresholds

	// Base thresholds on statistical properties
	dbThresh := stats.Mean - 0.5*stats.StdDev
	boxThresh := stats.Mean + 0.2*stats.StdDev

	// Adjust based on bimodality
	if stats.BimodalityIndex > 0.7 {
		// Highly bimodal - use more aggressive thresholds
		dbThresh = stats.Mean - 0.3*stats.StdDev
		boxThresh = stats.Mean + 0.3*stats.StdDev
	} else if stats.BimodalityIndex < 0.3 {
		// Low bimodality - use more conservative thresholds
		dbThresh = stats.Mean - 0.7*stats.StdDev
		boxThresh = stats.Mean + 0.1*stats.StdDev
	}

	// Adjust based on high probability ratio
	if stats.HighProbRatio > 0.3 {
		// Many high-probability pixels - can use higher thresholds
		dbThresh += 0.1
		boxThresh += 0.1
	} else if stats.HighProbRatio < 0.05 {
		// Few high-probability pixels - use lower thresholds
		dbThresh -= 0.1
		boxThresh -= 0.05
	}

	// Confidence based on dynamic range and bimodality
	confidence := stats.DynamicRange * (0.5 + 0.5*stats.BimodalityIndex)

	return dbThresh, boxThresh, confidence
}

// calculateDynamicThresholds uses dynamic range analysis for threshold calculation.
func calculateDynamicThresholds(probMap []float32, stats AdaptiveThresholdStats) (float32, float32, float32) {
	// Dynamic method adapts to the specific characteristics of the probability map

	// Use percentiles for robust threshold estimation
	sortedProbs := make([]float32, len(probMap))
	copy(sortedProbs, probMap)
	sort.Slice(sortedProbs, func(i, j int) bool {
		return sortedProbs[i] < sortedProbs[j]
	})

	// Find percentiles
	p25 := sortedProbs[len(sortedProbs)/4]    // 25th percentile
	p75 := sortedProbs[3*len(sortedProbs)/4]  // 75th percentile
	p90 := sortedProbs[9*len(sortedProbs)/10] // 90th percentile

	// Set db_thresh based on distribution characteristics
	var dbThresh float32
	if stats.DynamicRange > 0.8 {
		// High dynamic range - use adaptive threshold
		dbThresh = p25 + 0.3*(p75-p25)
	} else {
		// Low dynamic range - use more conservative approach
		dbThresh = stats.Mean - 0.5*stats.StdDev
	}

	// Set box_thresh based on high-probability content
	var boxThresh float32
	if stats.HighProbRatio > 0.2 {
		// Sufficient high-probability content
		boxThresh = p75
	} else {
		// Limited high-probability content - lower threshold
		boxThresh = stats.Median + 0.2*(p90-stats.Median)
	}

	// Confidence based on how well-separated the distribution is
	confidence := stats.DynamicRange * float32(math.Min(float64(p75-p25), 1.0))

	return dbThresh, boxThresh, confidence
}

// clampFloat32 clamps a float32 value between min and max.
func clampFloat32(value, minVal, maxVal float32) float32 {
	if value < minVal {
		return minVal
	}
	if value > maxVal {
		return maxVal
	}
	return value
}
