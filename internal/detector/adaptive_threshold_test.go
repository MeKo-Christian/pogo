package detector

import (
	"math"
	"testing"
)

func TestDefaultAdaptiveThresholdConfig(t *testing.T) {
	config := DefaultAdaptiveThresholdConfig()

	if config.Enabled {
		t.Error("Default config should have adaptive thresholds disabled")
	}
	if config.Method != AdaptiveMethodHistogram {
		t.Errorf("Expected default method to be AdaptiveMethodHistogram, got %v", config.Method)
	}
	if config.MinDbThresh != 0.1 {
		t.Errorf("Expected MinDbThresh to be 0.1, got %f", config.MinDbThresh)
	}
	if config.MaxDbThresh != 0.8 {
		t.Errorf("Expected MaxDbThresh to be 0.8, got %f", config.MaxDbThresh)
	}
	if config.MinBoxThresh != 0.3 {
		t.Errorf("Expected MinBoxThresh to be 0.3, got %f", config.MinBoxThresh)
	}
	if config.MaxBoxThresh != 0.9 {
		t.Errorf("Expected MaxBoxThresh to be 0.9, got %f", config.MaxBoxThresh)
	}
}

func TestCalculateAdaptiveThresholds_Disabled(t *testing.T) {
	config := DefaultAdaptiveThresholdConfig()
	config.Enabled = false

	probMap := []float32{0.1, 0.2, 0.3, 0.8, 0.9}

	result := CalculateAdaptiveThresholds(probMap, 5, 1, config)

	if result.Method != "disabled" {
		t.Errorf("Expected method to be 'disabled', got %s", result.Method)
	}
	if result.DbThresh != 0.3 {
		t.Errorf("Expected default DbThresh 0.3, got %f", result.DbThresh)
	}
	if result.BoxThresh != 0.5 {
		t.Errorf("Expected default BoxThresh 0.5, got %f", result.BoxThresh)
	}
}

func TestCalculateAdaptiveThresholds_InvalidInput(t *testing.T) {
	config := DefaultAdaptiveThresholdConfig()
	config.Enabled = true

	// Empty probability map
	result := CalculateAdaptiveThresholds([]float32{}, 0, 0, config)
	if result.Method != "disabled" {
		t.Errorf("Expected method to be 'disabled' for empty input, got %s", result.Method)
	}

	// Mismatched dimensions
	result = CalculateAdaptiveThresholds([]float32{0.1, 0.2}, 3, 1, config)
	if result.Method != "disabled" {
		t.Errorf("Expected method to be 'disabled' for mismatched dimensions, got %s", result.Method)
	}
}

func TestCalculateProbabilityMapStats(t *testing.T) {
	// Test with a known distribution
	probMap := []float32{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}

	stats := calculateProbabilityMapStats(probMap)

	expectedMean := float32(0.5)
	if math.Abs(float64(stats.Mean-expectedMean)) > 0.01 {
		t.Errorf("Expected mean ~%f, got %f", expectedMean, stats.Mean)
	}

	if stats.Min != 0.0 {
		t.Errorf("Expected min 0.0, got %f", stats.Min)
	}
	if stats.Max != 1.0 {
		t.Errorf("Expected max 1.0, got %f", stats.Max)
	}
	if stats.DynamicRange != 1.0 {
		t.Errorf("Expected dynamic range 1.0, got %f", stats.DynamicRange)
	}
	if stats.Median != 0.4 && stats.Median != 0.6 {
		t.Errorf("Expected median around 0.4-0.6, got %f", stats.Median)
	}

	// Check high probability ratio (values > 0.5)
	expectedHighProbRatio := float32(2.0 / 6.0) // 0.6, 0.8, 1.0 are > 0.5, but 0.6 is not > 0.5
	// Actually: 0.6, 0.8, 1.0 are > 0.5, so ratio should be 3/6 = 0.5
	expectedHighProbRatio = float32(3.0 / 6.0)
	if math.Abs(float64(stats.HighProbRatio-expectedHighProbRatio)) > 0.01 {
		t.Errorf("Expected high prob ratio ~%f, got %f", expectedHighProbRatio, stats.HighProbRatio)
	}
}

func TestCalculateProbabilityMapStats_EmptyInput(t *testing.T) {
	stats := calculateProbabilityMapStats([]float32{})

	// Should return zero-valued stats for empty input
	if stats.Mean != 0.0 || stats.Min != 0.0 || stats.Max != 0.0 {
		t.Error("Expected zero-valued stats for empty input")
	}
}

func TestCalculateBimodalityIndex(t *testing.T) {
	// Test unimodal distribution (all values around 0.5)
	unimodal := make([]float32, 100)
	for i := range unimodal {
		unimodal[i] = 0.5 + 0.1*float32(i%3-1) // Values around 0.4-0.6
	}

	unimodalIndex := calculateBimodalityIndex(unimodal)
	if unimodalIndex > 0.6 {
		t.Errorf("Expected low bimodality for unimodal distribution, got %f", unimodalIndex)
	}

	// Test bimodal distribution (values at 0.1 and 0.9)
	bimodal := make([]float32, 100)
	for i := range bimodal {
		if i%2 == 0 {
			bimodal[i] = 0.1
		} else {
			bimodal[i] = 0.9
		}
	}

	bimodalIndex := calculateBimodalityIndex(bimodal)
	if bimodalIndex < 0.8 {
		t.Errorf("Expected high bimodality for bimodal distribution, got %f", bimodalIndex)
	}
}

func TestCalculateOtsuThreshold(t *testing.T) {
	config := DefaultAdaptiveThresholdConfig()

	// Test with a simple bimodal distribution
	probMap := make([]float32, 200)
	// First half: low values (0.1)
	for i := range 100 {
		probMap[i] = 0.1
	}
	// Second half: high values (0.9)
	for i := 100; i < 200; i++ {
		probMap[i] = 0.9
	}

	threshold, confidence := calculateOtsuThreshold(probMap, config)

	// Otsu threshold should be somewhere between 0.1 and 0.9
	if threshold < 0.2 || threshold > 0.8 {
		t.Errorf("Expected Otsu threshold between 0.2 and 0.8, got %f", threshold)
	}

	// Confidence should be reasonable for a clear bimodal distribution
	if confidence < 0.1 {
		t.Errorf("Expected reasonable confidence for bimodal distribution, got %f", confidence)
	}
}

func TestCalculateOtsuThreshold_EmptyInput(t *testing.T) {
	config := DefaultAdaptiveThresholdConfig()

	threshold, confidence := calculateOtsuThreshold([]float32{}, config)

	if threshold != 0.3 || confidence != 0.5 {
		t.Errorf("Expected fallback values for empty input, got threshold=%f, confidence=%f", threshold, confidence)
	}
}

func TestCalculateHistogramBasedThresholds(t *testing.T) {
	config := DefaultAdaptiveThresholdConfig()

	// Create test statistics
	stats := AdaptiveThresholdStats{
		Mean:            0.5,
		StdDev:          0.2,
		DynamicRange:    0.8,
		BimodalityIndex: 0.7,
		HighProbRatio:   0.3,
	}

	probMap := []float32{0.1, 0.3, 0.5, 0.7, 0.9}

	dbThresh, boxThresh, confidence := calculateHistogramBasedThresholds(probMap, stats, config)

	// DbThresh should be lower than BoxThresh
	if dbThresh >= boxThresh {
		t.Errorf("Expected DbThresh < BoxThresh, got DbThresh=%f, BoxThresh=%f", dbThresh, boxThresh)
	}

	// Confidence should be reasonable
	if confidence < 0.0 || confidence > 1.0 {
		t.Errorf("Expected confidence between 0 and 1, got %f", confidence)
	}

	// Thresholds should be in reasonable range
	if dbThresh < 0.0 || dbThresh > 1.0 {
		t.Errorf("Expected DbThresh between 0 and 1, got %f", dbThresh)
	}
	if boxThresh < 0.0 || boxThresh > 1.0 {
		t.Errorf("Expected BoxThresh between 0 and 1, got %f", boxThresh)
	}
}

func TestCalculateDynamicThresholds(t *testing.T) {
	config := DefaultAdaptiveThresholdConfig()

	// Create test statistics with high dynamic range
	stats := AdaptiveThresholdStats{
		Mean:          0.5,
		StdDev:        0.3,
		Median:        0.4,
		DynamicRange:  0.9,
		HighProbRatio: 0.25,
	}

	// Create sorted probability map for percentile calculation
	probMap := []float32{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}

	dbThresh, boxThresh, confidence := calculateDynamicThresholds(probMap, stats, config)

	// DbThresh should be lower than BoxThresh
	if dbThresh >= boxThresh {
		t.Errorf("Expected DbThresh < BoxThresh, got DbThresh=%f, BoxThresh=%f", dbThresh, boxThresh)
	}

	// Confidence should be reasonable for high dynamic range
	if confidence < 0.1 {
		t.Errorf("Expected reasonable confidence for high dynamic range, got %f", confidence)
	}

	// Thresholds should be in reasonable range
	if dbThresh < 0.0 || dbThresh > 1.0 {
		t.Errorf("Expected DbThresh between 0 and 1, got %f", dbThresh)
	}
	if boxThresh < 0.0 || boxThresh > 1.0 {
		t.Errorf("Expected BoxThresh between 0 and 1, got %f", boxThresh)
	}
}

func TestCalculateAdaptiveThresholds_AllMethods(t *testing.T) {
	// Test all three adaptive methods
	methods := []AdaptiveThresholdMethod{
		AdaptiveMethodOtsu,
		AdaptiveMethodHistogram,
		AdaptiveMethodDynamic,
	}

	expectedMethodNames := []string{"otsu", "histogram", "dynamic"}

	// Create a test probability map with bimodal distribution
	probMap := make([]float32, 100)
	for i := range 50 {
		probMap[i] = 0.2
	}
	for i := 50; i < 100; i++ {
		probMap[i] = 0.8
	}

	for i, method := range methods {
		config := DefaultAdaptiveThresholdConfig()
		config.Enabled = true
		config.Method = method

		result := CalculateAdaptiveThresholds(probMap, 10, 10, config)

		if result.Method != expectedMethodNames[i] {
			t.Errorf("Expected method name %s, got %s", expectedMethodNames[i], result.Method)
		}

		// All methods should produce reasonable thresholds
		if result.DbThresh < config.MinDbThresh || result.DbThresh > config.MaxDbThresh {
			t.Errorf("DbThresh %f outside bounds [%f, %f] for method %s",
				result.DbThresh, config.MinDbThresh, config.MaxDbThresh, result.Method)
		}

		if result.BoxThresh < config.MinBoxThresh || result.BoxThresh > config.MaxBoxThresh {
			t.Errorf("BoxThresh %f outside bounds [%f, %f] for method %s",
				result.BoxThresh, config.MinBoxThresh, config.MaxBoxThresh, result.Method)
		}

		// DbThresh should be <= BoxThresh
		if result.DbThresh > result.BoxThresh {
			t.Errorf("DbThresh %f > BoxThresh %f for method %s",
				result.DbThresh, result.BoxThresh, result.Method)
		}

		// Statistics should be populated
		if result.Statistics.Mean == 0.0 && result.Statistics.Max == 0.0 {
			t.Errorf("Statistics not populated for method %s", result.Method)
		}
	}
}

func TestClampFloat32(t *testing.T) {
	tests := []struct {
		value, min, max, expected float32
	}{
		{0.5, 0.0, 1.0, 0.5},  // Normal case
		{-0.1, 0.0, 1.0, 0.0}, // Below min
		{1.5, 0.0, 1.0, 1.0},  // Above max
		{0.3, 0.5, 1.0, 0.5},  // Below min
		{0.8, 0.0, 0.5, 0.5},  // Above max
	}

	for _, test := range tests {
		result := clampFloat32(test.value, test.min, test.max)
		if result != test.expected {
			t.Errorf("clampFloat32(%f, %f, %f) = %f, expected %f",
				test.value, test.min, test.max, result, test.expected)
		}
	}
}

func TestAdaptiveThresholds_ThresholdOrdering(t *testing.T) {
	config := DefaultAdaptiveThresholdConfig()
	config.Enabled = true
	config.MinBoxThresh = 0.2
	config.MaxBoxThresh = 0.8
	config.MinDbThresh = 0.1
	config.MaxDbThresh = 0.7

	// Create a probability map that might result in inverted thresholds
	probMap := []float32{0.9, 0.9, 0.9, 0.9, 0.9} // Very high probabilities

	result := CalculateAdaptiveThresholds(probMap, 5, 1, config)

	// BoxThresh should always be >= DbThresh
	if result.BoxThresh < result.DbThresh {
		t.Errorf("BoxThresh %f < DbThresh %f, should be corrected to ensure BoxThresh >= DbThresh",
			result.BoxThresh, result.DbThresh)
	}
}

func TestAdaptiveThresholds_BoundaryConditions(t *testing.T) {
	config := DefaultAdaptiveThresholdConfig()
	config.Enabled = true
	config.MinDbThresh = 0.4
	config.MaxDbThresh = 0.6
	config.MinBoxThresh = 0.5
	config.MaxBoxThresh = 0.7

	// Test with various probability distributions
	testCases := []struct {
		name    string
		probMap []float32
	}{
		{"All zeros", []float32{0.0, 0.0, 0.0, 0.0}},
		{"All ones", []float32{1.0, 1.0, 1.0, 1.0}},
		{"Uniform mid", []float32{0.5, 0.5, 0.5, 0.5}},
		{"Single value", []float32{0.3}},
		{"Two values", []float32{0.1, 0.9}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := CalculateAdaptiveThresholds(tc.probMap, len(tc.probMap), 1, config)

			// Thresholds should respect bounds
			if result.DbThresh < config.MinDbThresh || result.DbThresh > config.MaxDbThresh {
				t.Errorf("%s: DbThresh %f outside bounds [%f, %f]",
					tc.name, result.DbThresh, config.MinDbThresh, config.MaxDbThresh)
			}

			if result.BoxThresh < config.MinBoxThresh || result.BoxThresh > config.MaxBoxThresh {
				t.Errorf("%s: BoxThresh %f outside bounds [%f, %f]",
					tc.name, result.BoxThresh, config.MinBoxThresh, config.MaxBoxThresh)
			}

			// BoxThresh should be >= DbThresh
			if result.BoxThresh < result.DbThresh {
				t.Errorf("%s: BoxThresh %f < DbThresh %f",
					tc.name, result.BoxThresh, result.DbThresh)
			}
		})
	}
}
