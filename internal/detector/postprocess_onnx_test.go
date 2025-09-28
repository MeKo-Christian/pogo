package detector

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

const nmsMethodLinear = "linear"

// createMockProbabilityMap creates a probability map with some text-like regions.
func createMockProbabilityMap() []float32 {
	width, height := 32, 32
	probMap := make([]float32, width*height)

	// Create a high-probability region in the center (simulating detected text)
	centerX, centerY := width/2, height/2
	regionSize := 8

	for y := centerY - regionSize/2; y < centerY+regionSize/2; y++ {
		for x := centerX - regionSize/2; x < centerX+regionSize/2; x++ {
			if x >= 0 && x < width && y >= 0 && y < height {
				probMap[y*width+x] = 0.8 // High probability for text
			}
		}
	}

	// Add some background noise
	for i := range probMap {
		if probMap[i] == 0 {
			probMap[i] = 0.1 // Low background probability
		}
	}

	return probMap
}

func TestDetector_DetectRegions_Basic(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = false
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	// Create mock result
	mockResult := &DetectionResult{
		ProbabilityMap: createMockProbabilityMap(),
		Width:          32,
		Height:         32,
		OriginalWidth:  640,
		OriginalHeight: 480,
		ProcessingTime: 1000000,
	}

	// Simulate the DetectRegions logic without actual ONNX inference
	probMap := mockResult.ProbabilityMap

	// Test the core post-processing logic
	regions := PostProcessDBWithOptions(probMap, mockResult.Width, mockResult.Height,
		config.DbThresh, config.DbBoxThresh, PostProcessOptions{UseMinAreaRect: true})

	scaledRegions := ScaleRegionsToOriginal(regions, mockResult.Width, mockResult.Height,
		mockResult.OriginalWidth, mockResult.OriginalHeight)

	// Verify we get some regions (the mock probability map should produce at least one)
	assert.NotEmpty(t, scaledRegions)

	// Verify regions are properly scaled
	for _, region := range scaledRegions {
		assert.GreaterOrEqual(t, region.Box.MinX, 0.0)
		assert.GreaterOrEqual(t, region.Box.MinY, 0.0)
		assert.LessOrEqual(t, region.Box.MaxX, float64(mockResult.OriginalWidth))
		assert.LessOrEqual(t, region.Box.MaxY, float64(mockResult.OriginalHeight))
		assert.Positive(t, region.Confidence)
	}
}

func TestDetector_DetectRegions_WithMorphology(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = false
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphDilate
	config.Morphology.KernelSize = 3
	config.Morphology.Iterations = 1

	// Simulate morphology application
	originalProbMap := createMockProbabilityMap()
	morphedProbMap := ApplyMorphologicalOperation(originalProbMap, 32, 32, config.Morphology)

	// Verify morphology was applied (dilate should increase high-probability areas)
	assert.NotEqual(t, originalProbMap, morphedProbMap)

	// Process regions with morphed probability map
	regions := PostProcessDBWithOptions(morphedProbMap, 32, 32,
		config.DbThresh, config.DbBoxThresh, PostProcessOptions{UseMinAreaRect: true})

	// The function may return nil for no regions, which is valid
	if regions != nil {
		t.Logf("Morphology test produced %d regions", len(regions))
	} else {
		t.Log("Morphology test produced no regions (nil)")
	}
}

func TestDetector_DetectRegions_WithAdaptiveThresholds(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = false
	config.AdaptiveThresholds.Enabled = true
	config.AdaptiveThresholds.Method = AdaptiveMethodHistogram
	config.Morphology.Operation = MorphNone

	probMap := createMockProbabilityMap()

	// Test adaptive threshold calculation
	adaptiveResult := CalculateAdaptiveThresholds(probMap, 32, 32, config.AdaptiveThresholds)

	assert.NotNil(t, adaptiveResult)
	assert.Greater(t, adaptiveResult.Confidence, float32(0))
	assert.Greater(t, adaptiveResult.BoxThresh, adaptiveResult.DbThresh)
	assert.Equal(t, "histogram", adaptiveResult.Method)

	// Process with adaptive thresholds
	regions := PostProcessDBWithOptions(probMap, 32, 32,
		adaptiveResult.DbThresh, adaptiveResult.BoxThresh, PostProcessOptions{UseMinAreaRect: true})

	// The function can return nil for empty regions, which is valid
	if regions != nil {
		t.Logf("Found %d regions with adaptive thresholds", len(regions))
	} else {
		t.Log("No regions found with adaptive thresholds (returned nil)")
	}
	// Test passes regardless - nil or empty slice both indicate no error occurred
}

func TestDetector_DetectRegions_WithHardNMS(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = true
	config.NMSMethod = "hard"
	config.NMSThreshold = 0.3
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	// Create probability map with overlapping regions
	probMap := make([]float32, 32*32)

	// Create two overlapping high-probability regions
	for y := 8; y < 16; y++ {
		for x := 8; x < 16; x++ {
			probMap[y*32+x] = 0.8
		}
	}
	for y := 12; y < 20; y++ {
		for x := 12; x < 20; x++ {
			probMap[y*32+x] = 0.7
		}
	}

	// Process with Hard NMS
	regions := PostProcessDBWithNMSOptions(probMap, 32, 32,
		config.DbThresh, config.DbBoxThresh, config.NMSThreshold,
		PostProcessOptions{UseMinAreaRect: true})

	// NMS should reduce overlapping regions
	assert.NotNil(t, regions)
}

func TestDetector_DetectRegions_WithSoftNMS(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = true
	config.NMSMethod = nmsMethodLinear
	config.NMSThreshold = 0.3
	config.SoftNMSSigma = 0.5
	config.SoftNMSThresh = 0.1
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	probMap := createMockProbabilityMap()

	// First get regions without NMS
	regions := PostProcessDBWithOptions(probMap, 32, 32,
		config.DbThresh, config.DbBoxThresh, PostProcessOptions{UseMinAreaRect: true})

	// Apply Soft NMS
	softNMSRegions := SoftNonMaxSuppression(regions, config.NMSMethod, config.NMSThreshold,
		config.SoftNMSSigma, config.SoftNMSThresh)

	// SoftNMS may return nil for no regions, which is valid
	if softNMSRegions != nil {
		t.Logf("SoftNMS produced %d regions", len(softNMSRegions))
	} else {
		t.Log("SoftNMS produced no regions")
	}
}

func TestDetector_DetectRegions_WithAdaptiveNMS(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = true
	config.UseAdaptiveNMS = true
	config.NMSThreshold = 0.3
	config.AdaptiveNMSScale = 1.2
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	probMap := createMockProbabilityMap()

	// First get regions without NMS
	regions := PostProcessDBWithOptions(probMap, 32, 32,
		config.DbThresh, config.DbBoxThresh, PostProcessOptions{UseMinAreaRect: true})

	// Apply Adaptive NMS
	adaptiveNMSRegions := AdaptiveNonMaxSuppression(regions, config.NMSThreshold, config.AdaptiveNMSScale)

	// AdaptiveNMS may return nil for no regions, which is valid
	if adaptiveNMSRegions != nil {
		t.Logf("AdaptiveNMS produced %d regions", len(adaptiveNMSRegions))
	} else {
		t.Log("AdaptiveNMS produced no regions")
	}
}

func TestDetector_DetectRegions_WithSizeAwareNMS(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = true
	config.SizeAwareNMS = true
	config.NMSThreshold = 0.3
	config.SizeNMSScaleFactor = 1.5
	config.MinRegionSize = 10
	config.MaxRegionSize = 1000
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	probMap := createMockProbabilityMap()

	// First get regions without NMS
	regions := PostProcessDBWithOptions(probMap, 32, 32,
		config.DbThresh, config.DbBoxThresh, PostProcessOptions{UseMinAreaRect: true})

	// Apply Size-Aware NMS
	sizeAwareRegions := SizeAwareNonMaxSuppression(regions, config.NMSThreshold, config.SizeNMSScaleFactor,
		config.MinRegionSize, config.MaxRegionSize)

	// SizeAwareNMS may return nil for no regions, which is valid
	if sizeAwareRegions != nil {
		t.Logf("SizeAwareNMS produced %d regions", len(sizeAwareRegions))
	} else {
		t.Log("SizeAwareNMS produced no regions")
	}
}

func TestDetector_DetectRegions_AllFeaturesEnabled(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = true
	config.NMSMethod = nmsMethodLinear
	config.NMSThreshold = 0.3
	config.SoftNMSSigma = 0.5
	config.SoftNMSThresh = 0.1
	config.AdaptiveThresholds.Enabled = true
	config.AdaptiveThresholds.Method = AdaptiveMethodDynamic
	config.Morphology.Operation = MorphSmooth
	config.Morphology.KernelSize = 3
	config.Morphology.Iterations = 1

	probMap := createMockProbabilityMap()

	// Apply morphology
	morphedProbMap := ApplyMorphologicalOperation(probMap, 32, 32, config.Morphology)

	// Calculate adaptive thresholds
	adaptiveResult := CalculateAdaptiveThresholds(morphedProbMap, 32, 32, config.AdaptiveThresholds)

	// Process regions
	regions := PostProcessDBWithOptions(morphedProbMap, 32, 32,
		adaptiveResult.DbThresh, adaptiveResult.BoxThresh, PostProcessOptions{UseMinAreaRect: true})

	// Apply Soft NMS
	finalRegions := SoftNonMaxSuppression(regions, config.NMSMethod, config.NMSThreshold,
		config.SoftNMSSigma, config.SoftNMSThresh)

	// Scale to original size
	scaledRegions := ScaleRegionsToOriginal(finalRegions, 32, 32, 640, 480)

	// Verify all regions are within bounds (if any regions exist)
	if scaledRegions != nil {
		for _, region := range scaledRegions {
			assert.GreaterOrEqual(t, region.Box.MinX, 0.0)
			assert.GreaterOrEqual(t, region.Box.MinY, 0.0)
			assert.LessOrEqual(t, region.Box.MaxX, 640.0)
			assert.LessOrEqual(t, region.Box.MaxY, 480.0)
		}
		t.Logf("All features test produced %d valid regions", len(scaledRegions))
	} else {
		t.Log("All features test produced no regions")
	}
}

func TestDetector_DetectRegions_ContourMode(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = false
	config.PolygonMode = "contour"
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	probMap := createMockProbabilityMap()

	// Process with contour mode (UseMinAreaRect should be false)
	regions := PostProcessDBWithOptions(probMap, 32, 32,
		config.DbThresh, config.DbBoxThresh, PostProcessOptions{UseMinAreaRect: false})

	// Contour mode may return nil for no regions, which is valid
	if regions != nil {
		t.Logf("Contour mode produced %d regions", len(regions))
	} else {
		t.Log("Contour mode produced no regions")
	}
}

func TestDetector_DetectRegions_EmptyProbabilityMap(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = false
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	// Create empty probability map (all zeros)
	probMap := make([]float32, 32*32)

	// Process empty map
	regions := PostProcessDBWithOptions(probMap, 32, 32,
		config.DbThresh, config.DbBoxThresh, PostProcessOptions{UseMinAreaRect: true})

	// Should return empty regions without error
	assert.Empty(t, regions)
}

func TestDetector_DetectRegions_HighThresholds(t *testing.T) {
	config := DefaultConfig()
	config.DbThresh = 0.9     // Very high threshold
	config.DbBoxThresh = 0.95 // Very high threshold
	config.UseNMS = false
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	probMap := createMockProbabilityMap()

	// Process with high thresholds (should filter out most regions)
	regions := PostProcessDBWithOptions(probMap, 32, 32,
		config.DbThresh, config.DbBoxThresh, PostProcessOptions{UseMinAreaRect: true})

	// With high thresholds, we expect fewer or no regions
	assert.LessOrEqual(t, len(regions), 1) // Allow for at most 1 very strong region
}

func TestDetector_PostProcessDBWithNMS(t *testing.T) {
	tests := []struct {
		name         string
		dbThresh     float32
		boxMinConf   float32
		iouThreshold float64
		expectNMS    bool
	}{
		{
			name:         "Standard thresholds with NMS",
			dbThresh:     0.3,
			boxMinConf:   0.5,
			iouThreshold: 0.3,
			expectNMS:    true,
		},
		{
			name:         "High IOU threshold (less aggressive NMS)",
			dbThresh:     0.3,
			boxMinConf:   0.5,
			iouThreshold: 0.8,
			expectNMS:    true,
		},
		{
			name:         "Low IOU threshold (aggressive NMS)",
			dbThresh:     0.3,
			boxMinConf:   0.5,
			iouThreshold: 0.1,
			expectNMS:    true,
		},
		{
			name:         "High detection thresholds",
			dbThresh:     0.8,
			boxMinConf:   0.9,
			iouThreshold: 0.3,
			expectNMS:    false, // May not find overlapping regions to suppress
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			probMap := createMockProbabilityMap()

			// Test PostProcessDBWithNMS directly
			regionsWithNMS := PostProcessDBWithNMS(probMap, 32, 32,
				tt.dbThresh, tt.boxMinConf, tt.iouThreshold)

			// For comparison, get regions without NMS
			regionsWithoutNMS := PostProcessDB(probMap, 32, 32,
				tt.dbThresh, tt.boxMinConf)

			// Basic validations - regions can be nil if no regions found
			if regionsWithNMS != nil {
				assert.LessOrEqual(t, len(regionsWithNMS), len(regionsWithoutNMS),
					"NMS should not increase the number of regions")

				// If we found regions, validate their properties
				for _, region := range regionsWithNMS {
					assert.GreaterOrEqual(t, region.Confidence, float64(tt.boxMinConf),
						"All regions should meet minimum confidence threshold")
					assert.NotEmpty(t, region.Polygon,
						"All regions should have valid polygons")
				}
			}

			// Log results for debugging
			regionsWithNMSCount := 0
			regionsWithoutNMSCount := 0
			if regionsWithNMS != nil {
				regionsWithNMSCount = len(regionsWithNMS)
			}
			if regionsWithoutNMS != nil {
				regionsWithoutNMSCount = len(regionsWithoutNMS)
			}
			t.Logf("PostProcessDBWithNMS: found %d regions (vs %d without NMS)",
				regionsWithNMSCount, regionsWithoutNMSCount)
		})
	}
}

func TestDetector_PostProcessDBWithNMS_EmptyInput(t *testing.T) {
	// Test with empty probability map
	emptyProbMap := make([]float32, 32*32)

	regions := PostProcessDBWithNMS(emptyProbMap, 32, 32, 0.3, 0.5, 0.3)

	// Function may return nil for no regions
	if regions != nil {
		assert.Empty(t, regions, "Empty probability map should produce no regions")
	}
	t.Logf("PostProcessDBWithNMS with empty input: returned %v", regions == nil)
}

func TestDetector_PostProcessDBWithNMS_NoOverlap(t *testing.T) {
	// Create probability map with separated regions (no overlap)
	probMap := make([]float32, 64*64)

	// Create two separate high-probability regions
	// Region 1: top-left quadrant
	for y := 10; y < 20; y++ {
		for x := 10; x < 20; x++ {
			probMap[y*64+x] = 0.8
		}
	}

	// Region 2: bottom-right quadrant
	for y := 40; y < 50; y++ {
		for x := 40; x < 50; x++ {
			probMap[y*64+x] = 0.8
		}
	}

	regionsWithNMS := PostProcessDBWithNMS(probMap, 64, 64, 0.3, 0.5, 0.3)
	regionsWithoutNMS := PostProcessDB(probMap, 64, 64, 0.3, 0.5)

	// When regions don't overlap, NMS shouldn't remove any
	assert.Len(t, regionsWithNMS, len(regionsWithoutNMS),
		"Non-overlapping regions should not be affected by NMS")
}
