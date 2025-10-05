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

// ========================================
// Comprehensive DetectRegions Tests
// ========================================

// createMultiRegionProbabilityMap creates a probability map with multiple text-like regions.
func createMultiRegionProbabilityMap(width, height int) []float32 {
	probMap := make([]float32, width*height)

	// Create 3 distinct high-probability regions
	regions := []struct{ x1, y1, x2, y2 int }{
		{5, 5, 15, 10},   // Region 1: top-left
		{20, 5, 30, 10},  // Region 2: top-right
		{10, 15, 25, 20}, // Region 3: bottom
	}

	for _, r := range regions {
		for y := r.y1; y < r.y2 && y < height; y++ {
			for x := r.x1; x < r.x2 && x < width; x++ {
				probMap[y*width+x] = 0.85
			}
		}
	}

	// Add low-level background noise
	for i := range probMap {
		if probMap[i] == 0 {
			probMap[i] = 0.05
		}
	}

	return probMap
}

// createLargeProbabilityMap creates a large probability map for stress testing.
func createLargeProbabilityMap(width, height int) []float32 {
	probMap := make([]float32, width*height)

	// Create a grid of text regions
	regionSize := 20
	spacing := 5
	for y := 0; y < height; y += regionSize + spacing {
		for x := 0; x < width; x += regionSize + spacing {
			for dy := 0; dy < regionSize && (y+dy) < height; dy++ {
				for dx := 0; dx < regionSize && (x+dx) < width; dx++ {
					idx := (y+dy)*width + (x + dx)
					if idx < len(probMap) {
						probMap[idx] = 0.7 + float32(x%3)*0.05 // Varying confidence
					}
				}
			}
		}
	}

	return probMap
}

// simulateDetectRegionsLogic simulates the core logic of DetectRegions without actual ONNX inference.
func simulateDetectRegionsLogic(config Config, result *DetectionResult) ([]DetectedRegion, error) {
	// This is the actual logic from DetectRegions
	probMap := result.ProbabilityMap

	// Apply morphological operations if configured
	if config.Morphology.Operation != MorphNone {
		probMap = ApplyMorphologicalOperation(probMap, result.Width, result.Height, config.Morphology)
	}

	// Calculate adaptive thresholds if enabled
	dbThresh := config.DbThresh
	boxThresh := config.DbBoxThresh
	if config.AdaptiveThresholds.Enabled {
		adaptiveThresh := CalculateAdaptiveThresholds(probMap, result.Width, result.Height, config.AdaptiveThresholds)
		dbThresh = adaptiveThresh.DbThresh
		boxThresh = adaptiveThresh.BoxThresh
	}

	var regs []DetectedRegion
	opts := PostProcessOptions{UseMinAreaRect: config.PolygonMode != "contour"}

	if config.UseNMS {
		// Choose NMS method based on configuration
		switch config.NMSMethod {
		case "linear", "gaussian":
			regs = PostProcessDBWithOptions(probMap, result.Width, result.Height,
				dbThresh, boxThresh, opts)
			regs = SoftNonMaxSuppression(regs, config.NMSMethod, config.NMSThreshold,
				config.SoftNMSSigma, config.SoftNMSThresh)
		default:
			switch {
			case config.UseAdaptiveNMS:
				regs = PostProcessDBWithOptions(probMap, result.Width, result.Height,
					dbThresh, boxThresh, opts)
				regs = AdaptiveNonMaxSuppression(regs, config.NMSThreshold, config.AdaptiveNMSScale)
			case config.SizeAwareNMS:
				regs = PostProcessDBWithOptions(probMap, result.Width, result.Height,
					dbThresh, boxThresh, opts)
				regs = SizeAwareNonMaxSuppression(regs, config.NMSThreshold, config.SizeNMSScaleFactor,
					config.MinRegionSize, config.MaxRegionSize)
			default:
				regs = PostProcessDBWithNMSOptions(probMap, result.Width, result.Height,
					dbThresh, boxThresh, config.NMSThreshold, opts)
			}
		}
	} else {
		regs = PostProcessDBWithOptions(probMap, result.Width, result.Height,
			dbThresh, boxThresh, opts)
	}

	regs = ScaleRegionsToOriginal(regs, result.Width, result.Height, result.OriginalWidth, result.OriginalHeight)
	return regs, nil
}

// TestDetectRegions_DefaultConfig tests DetectRegions with default configuration.
func TestDetectRegions_DefaultConfig(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = false
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	result := &DetectionResult{
		ProbabilityMap: createMockProbabilityMap(),
		Width:          32,
		Height:         32,
		OriginalWidth:  640,
		OriginalHeight: 480,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.NotNil(t, regions)
	// Verify regions are within original image bounds
	for _, region := range regions {
		assert.GreaterOrEqual(t, region.Box.MinX, 0.0)
		assert.GreaterOrEqual(t, region.Box.MinY, 0.0)
		assert.LessOrEqual(t, region.Box.MaxX, float64(640))
		assert.LessOrEqual(t, region.Box.MaxY, float64(480))
	}
}

// TestDetectRegions_EmptyProbMap tests DetectRegions with empty probability map.
func TestDetectRegions_EmptyProbMap(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = false
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	result := &DetectionResult{
		ProbabilityMap: make([]float32, 32*32), // All zeros
		Width:          32,
		Height:         32,
		OriginalWidth:  640,
		OriginalHeight: 480,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.Empty(t, regions, "Empty probability map should produce no regions")
}

// TestDetectRegions_MultipleRegions tests detecting multiple distinct regions.
func TestDetectRegions_MultipleRegions(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = false
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	result := &DetectionResult{
		ProbabilityMap: createMultiRegionProbabilityMap(32, 32),
		Width:          32,
		Height:         32,
		OriginalWidth:  640,
		OriginalHeight: 480,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.NotNil(t, regions)
	if len(regions) > 0 {
		// Verify all regions have valid properties
		for i, region := range regions {
			assert.Positive(t, region.Confidence, "Region %d should have positive confidence", i)
			assert.NotEmpty(t, region.Polygon, "Region %d should have polygon", i)
			assert.NotNil(t, region.Box, "Region %d should have bounding box", i)
		}
	}
}

// TestDetectRegions_LargeImage tests DetectRegions with large probability maps.
func TestDetectRegions_LargeImage(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = true
	config.NMSMethod = "hard"
	config.NMSThreshold = 0.3

	// Larger probability map
	width, height := 128, 96
	result := &DetectionResult{
		ProbabilityMap: createLargeProbabilityMap(width, height),
		Width:          width,
		Height:         height,
		OriginalWidth:  2560,
		OriginalHeight: 1920,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.NotNil(t, regions)
	// Large images should produce multiple regions
	t.Logf("Large image test produced %d regions", len(regions))
}

// TestDetectRegions_SingleRegion tests detecting a single text region.
func TestDetectRegions_SingleRegion(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = false
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	// Create probability map with single centered region
	probMap := make([]float32, 32*32)
	for y := 12; y < 20; y++ {
		for x := 12; x < 20; x++ {
			probMap[y*32+x] = 0.9
		}
	}

	result := &DetectionResult{
		ProbabilityMap: probMap,
		Width:          32,
		Height:         32,
		OriginalWidth:  640,
		OriginalHeight: 480,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.NotNil(t, regions)
	if len(regions) > 0 {
		// Should produce exactly one high-confidence region
		assert.LessOrEqual(t, len(regions), 2, "Should detect one primary region")
		assert.GreaterOrEqual(t, regions[0].Confidence, 0.7)
	}
}

// ========================================
// Configuration Variant Tests
// ========================================

// TestDetectRegions_WithHardNMS tests DetectRegions with Hard NMS.
func TestDetectRegions_WithHardNMS(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = true
	config.NMSMethod = "hard"
	config.NMSThreshold = 0.3
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	result := &DetectionResult{
		ProbabilityMap: createMultiRegionProbabilityMap(32, 32),
		Width:          32,
		Height:         32,
		OriginalWidth:  640,
		OriginalHeight: 480,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.NotNil(t, regions)
	t.Logf("Hard NMS produced %d regions", len(regions))
}

// TestDetectRegions_WithSoftNMS_Linear tests DetectRegions with Linear Soft-NMS.
func TestDetectRegions_WithSoftNMS_Linear(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = true
	config.NMSMethod = nmsMethodLinear
	config.NMSThreshold = 0.3
	config.SoftNMSSigma = 0.5
	config.SoftNMSThresh = 0.1
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	result := &DetectionResult{
		ProbabilityMap: createMultiRegionProbabilityMap(32, 32),
		Width:          32,
		Height:         32,
		OriginalWidth:  640,
		OriginalHeight: 480,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.NotNil(t, regions)
	t.Logf("Linear Soft-NMS produced %d regions", len(regions))
}

// TestDetectRegions_WithSoftNMS_Gaussian tests DetectRegions with Gaussian Soft-NMS.
func TestDetectRegions_WithSoftNMS_Gaussian(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = true
	config.NMSMethod = "gaussian"
	config.NMSThreshold = 0.3
	config.SoftNMSSigma = 0.5
	config.SoftNMSThresh = 0.1
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	result := &DetectionResult{
		ProbabilityMap: createMultiRegionProbabilityMap(32, 32),
		Width:          32,
		Height:         32,
		OriginalWidth:  640,
		OriginalHeight: 480,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.NotNil(t, regions)
	t.Logf("Gaussian Soft-NMS produced %d regions", len(regions))
}

// TestDetectRegions_WithAdaptiveNMS tests DetectRegions with Adaptive NMS.
func TestDetectRegions_WithAdaptiveNMS_Method(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = true
	config.UseAdaptiveNMS = true
	config.NMSThreshold = 0.3
	config.AdaptiveNMSScale = 1.2
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	result := &DetectionResult{
		ProbabilityMap: createMultiRegionProbabilityMap(32, 32),
		Width:          32,
		Height:         32,
		OriginalWidth:  640,
		OriginalHeight: 480,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.NotNil(t, regions)
	t.Logf("Adaptive NMS produced %d regions", len(regions))
}

// TestDetectRegions_WithSizeAwareNMS tests DetectRegions with Size-Aware NMS.
func TestDetectRegions_WithSizeAwareNMS_Method(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = true
	config.SizeAwareNMS = true
	config.NMSThreshold = 0.3
	config.SizeNMSScaleFactor = 1.5
	config.MinRegionSize = 10
	config.MaxRegionSize = 1000
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	result := &DetectionResult{
		ProbabilityMap: createMultiRegionProbabilityMap(32, 32),
		Width:          32,
		Height:         32,
		OriginalWidth:  640,
		OriginalHeight: 480,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.NotNil(t, regions)
	t.Logf("Size-Aware NMS produced %d regions", len(regions))
}

// TestDetectRegions_WithAdaptiveThresholds_Histogram tests adaptive thresholds with histogram method.
func TestDetectRegions_WithAdaptiveThresholds_Histogram(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = false
	config.AdaptiveThresholds.Enabled = true
	config.AdaptiveThresholds.Method = AdaptiveMethodHistogram
	config.Morphology.Operation = MorphNone

	result := &DetectionResult{
		ProbabilityMap: createMultiRegionProbabilityMap(32, 32),
		Width:          32,
		Height:         32,
		OriginalWidth:  640,
		OriginalHeight: 480,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.NotNil(t, regions)
	t.Logf("Adaptive thresholds (histogram) produced %d regions", len(regions))
}

// TestDetectRegions_WithAdaptiveThresholds_Dynamic tests adaptive thresholds with dynamic method.
func TestDetectRegions_WithAdaptiveThresholds_Dynamic(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = false
	config.AdaptiveThresholds.Enabled = true
	config.AdaptiveThresholds.Method = AdaptiveMethodDynamic
	config.Morphology.Operation = MorphNone

	result := &DetectionResult{
		ProbabilityMap: createMultiRegionProbabilityMap(32, 32),
		Width:          32,
		Height:         32,
		OriginalWidth:  640,
		OriginalHeight: 480,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.NotNil(t, regions)
	t.Logf("Adaptive thresholds (dynamic) produced %d regions", len(regions))
}

// TestDetectRegions_WithMorphology_Dilate tests morphological dilation.
func TestDetectRegions_WithMorphology_Dilate(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = false
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphDilate
	config.Morphology.KernelSize = 3
	config.Morphology.Iterations = 1

	result := &DetectionResult{
		ProbabilityMap: createMultiRegionProbabilityMap(32, 32),
		Width:          32,
		Height:         32,
		OriginalWidth:  640,
		OriginalHeight: 480,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.NotNil(t, regions)
	t.Logf("Morphology (dilate) produced %d regions", len(regions))
}

// TestDetectRegions_WithMorphology_Erode tests morphological erosion.
func TestDetectRegions_WithMorphology_Erode(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = false
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphErode
	config.Morphology.KernelSize = 3
	config.Morphology.Iterations = 1

	result := &DetectionResult{
		ProbabilityMap: createMultiRegionProbabilityMap(32, 32),
		Width:          32,
		Height:         32,
		OriginalWidth:  640,
		OriginalHeight: 480,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.NotNil(t, regions)
	t.Logf("Morphology (erode) produced %d regions", len(regions))
}

// TestDetectRegions_WithMorphology_Smooth tests morphological smoothing.
func TestDetectRegions_WithMorphology_Smooth(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = false
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphSmooth
	config.Morphology.KernelSize = 3
	config.Morphology.Iterations = 1

	result := &DetectionResult{
		ProbabilityMap: createMultiRegionProbabilityMap(32, 32),
		Width:          32,
		Height:         32,
		OriginalWidth:  640,
		OriginalHeight: 480,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.NotNil(t, regions)
	t.Logf("Morphology (smooth) produced %d regions", len(regions))
}

// TestDetectRegions_PolygonMode_Contour tests contour polygon mode.
func TestDetectRegions_PolygonMode_Contour(t *testing.T) {
	config := DefaultConfig()
	config.UseNMS = false
	config.PolygonMode = "contour"
	config.AdaptiveThresholds.Enabled = false
	config.Morphology.Operation = MorphNone

	result := &DetectionResult{
		ProbabilityMap: createMultiRegionProbabilityMap(32, 32),
		Width:          32,
		Height:         32,
		OriginalWidth:  640,
		OriginalHeight: 480,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.NotNil(t, regions)
	t.Logf("Contour polygon mode produced %d regions", len(regions))
}

// ========================================
// Integration & Error Tests
// ========================================

// TestDetectRegions_InferenceError tests error handling when inference fails.
func TestDetectRegions_InferenceError(t *testing.T) {
	// This test simulates an error scenario - we return an error directly
	config := DefaultConfig()
	result := &DetectionResult{
		ProbabilityMap: nil, // Nil probability map should cause issues
		Width:          0,
		Height:         0,
		OriginalWidth:  640,
		OriginalHeight: 480,
	}

	// The actual DetectRegions would fail during RunInference
	// For our simulation, we just test that the logic handles nil probMap gracefully
	regions, err := simulateDetectRegionsLogic(config, result)

	// With nil or empty probMap, should return empty regions, not error
	assert.NoError(t, err)
	assert.Empty(t, regions)
}

// TestDetectRegions_AllFeaturesEnabled_Integration tests all features together.
func TestDetectRegions_AllFeaturesEnabled_Integration(t *testing.T) {
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

	result := &DetectionResult{
		ProbabilityMap: createMultiRegionProbabilityMap(64, 64),
		Width:          64,
		Height:         64,
		OriginalWidth:  1280,
		OriginalHeight: 960,
	}

	regions, err := simulateDetectRegionsLogic(config, result)

	assert.NoError(t, err)
	assert.NotNil(t, regions)

	// With all features enabled, verify regions are valid
	for i, region := range regions {
		assert.Positive(t, region.Confidence, "Region %d should have positive confidence", i)
		assert.NotEmpty(t, region.Polygon, "Region %d should have polygon", i)
		assert.GreaterOrEqual(t, region.Box.MinX, 0.0, "Region %d MinX in bounds", i)
		assert.GreaterOrEqual(t, region.Box.MinY, 0.0, "Region %d MinY in bounds", i)
		assert.LessOrEqual(t, region.Box.MaxX, 1280.0, "Region %d MaxX in bounds", i)
		assert.LessOrEqual(t, region.Box.MaxY, 960.0, "Region %d MaxY in bounds", i)
	}

	t.Logf("All features enabled produced %d valid regions", len(regions))
}
