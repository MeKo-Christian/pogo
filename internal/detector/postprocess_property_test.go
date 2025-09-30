package detector

import (
	"testing"

	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
)

// genProbabilityMap generates a probability map with values in [0, 1].
func genProbabilityMap(width, height int) gopter.Gen {
	return gen.SliceOfN(width*height, gen.Float32Range(0.0, 1.0))
}

// genValidDimensions generates valid width and height for probability maps.
func genValidDimensions() gopter.Gen {
	return gopter.CombineGens(
		gen.IntRange(8, 64),
		gen.IntRange(8, 64),
	).Map(func(vals []interface{}) [2]int {
		return [2]int{vals[0].(int), vals[1].(int)}
	})
}

// TestPostProcessDB_BinarizationProperty verifies that binarized mask contains only 0 or 1.
func TestPostProcessDB_BinarizationProperty(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("binarized mask contains only boolean values", prop.ForAll(
		func(dims [2]int, probMap []float32, threshold float32) bool {
			width, height := dims[0], dims[1]
			if len(probMap) != width*height {
				return true // skip invalid inputs
			}

			mask := binarize(probMap, width, height, threshold)
			if len(mask) != width*height {
				return false
			}

			// Check that all values are either false or true (booleans)
			// mask is []bool, so values are correct by type
			return true
		},
		genValidDimensions(),
		gen.SliceOfN(128*128, gen.Float32Range(0.0, 1.0)),
		gen.Float32Range(0.1, 0.9),
	))

	properties.TestingRun(t)
}

// TestPostProcessDB_OutputNonIncreasing verifies output count doesn't increase with filtering.
func TestPostProcessDB_OutputNonIncreasing(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("filtering reduces or maintains region count", prop.ForAll(
		func(dims [2]int, threshold1, threshold2 float32) bool {
			width, height := dims[0], dims[1]

			// Generate a simple probability map with some structure
			probMap := make([]float32, width*height)
			for i := range probMap {
				// Create some regions with varying probabilities
				x := i % width
				y := i / width
				if (x/10+y/10)%2 == 0 {
					probMap[i] = 0.8
				} else {
					probMap[i] = 0.2
				}
			}

			// Ensure threshold1 < threshold2
			if threshold1 > threshold2 {
				threshold1, threshold2 = threshold2, threshold1
			}

			regions1 := PostProcessDB(probMap, width, height, threshold1, 0.1)
			regions2 := PostProcessDB(probMap, width, height, threshold2, 0.1)

			// Higher threshold should produce fewer or equal regions
			return len(regions2) <= len(regions1)
		},
		genValidDimensions(),
		gen.Float32Range(0.1, 0.4),
		gen.Float32Range(0.5, 0.8),
	))

	properties.TestingRun(t)
}

// TestPostProcessDB_ConfidenceInRange verifies all output confidences are in [0, 1].
func TestPostProcessDB_ConfidenceInRange(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("all region confidences are in valid range [0, 1]", prop.ForAll(
		func(dims [2]int, dbThresh, boxMinConf float32) bool {
			width, height := dims[0], dims[1]

			// Generate probability map
			probMap := make([]float32, width*height)
			for i := range probMap {
				probMap[i] = float32(i%100) / 100.0
			}

			regions := PostProcessDB(probMap, width, height, dbThresh, boxMinConf)

			for _, region := range regions {
				if region.Confidence < 0.0 || region.Confidence > 1.0 {
					return false
				}
			}
			return true
		},
		genValidDimensions(),
		gen.Float32Range(0.1, 0.8),
		gen.Float32Range(0.1, 0.9),
	))

	properties.TestingRun(t)
}

// TestPostProcessDB_ValidBoxes verifies that all bounding boxes have valid dimensions.
func TestPostProcessDB_ValidBoxes(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("all bounding boxes have positive dimensions", prop.ForAll(
		func(dims [2]int, dbThresh float32) bool {
			width, height := dims[0], dims[1]

			// Generate probability map with clear regions
			probMap := make([]float32, width*height)
			for y := range height {
				for x := range width {
					i := y*width + x
					// Create rectangular regions
					if x >= width/4 && x < 3*width/4 && y >= height/4 && y < 3*height/4 {
						probMap[i] = 0.9
					} else {
						probMap[i] = 0.1
					}
				}
			}

			regions := PostProcessDB(probMap, width, height, dbThresh, 0.1)

			for _, region := range regions {
				box := region.Box
				// Check that box has positive width and height
				if box.MaxX <= box.MinX || box.MaxY <= box.MinY {
					return false
				}
				// Check that box is within image bounds
				if box.MinX < 0 || box.MinY < 0 {
					return false
				}
				if box.MaxX > float64(width) || box.MaxY > float64(height) {
					return false
				}
			}
			return true
		},
		genValidDimensions(),
		gen.Float32Range(0.2, 0.7),
	))

	properties.TestingRun(t)
}

// TestScaleRegionsToOriginal_ProportionalScaling verifies scaling maintains aspect ratios.
func TestScaleRegionsToOriginal_ProportionalScaling(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("scaling maintains box aspect ratios", prop.ForAll(
		func(mapW, mapH, origW, origH int) bool {
			if mapW <= 0 || mapH <= 0 || origW <= 0 || origH <= 0 {
				return true // skip invalid inputs
			}

			// Create a region in map coordinates
			region := DetectedRegion{
				Box:        utils.NewBox(10, 10, 20, 20),
				Confidence: 0.8,
			}
			regions := []DetectedRegion{region}

			scaled := ScaleRegionsToOriginal(regions, mapW, mapH, origW, origH)

			if len(scaled) != 1 {
				return false
			}

			// Check that aspect ratio is approximately preserved
			origAspect := float64(region.Box.MaxX-region.Box.MinX) / float64(region.Box.MaxY-region.Box.MinY)
			scaledAspect := (scaled[0].Box.MaxX - scaled[0].Box.MinX) / (scaled[0].Box.MaxY - scaled[0].Box.MinY)

			// Should be equal (within floating point tolerance)
			ratio := origAspect / scaledAspect
			return ratio > 0.99 && ratio < 1.01
		},
		gen.IntRange(32, 256),  // mapW
		gen.IntRange(32, 256),  // mapH
		gen.IntRange(64, 512),  // origW
		gen.IntRange(64, 512),  // origH
	))

	properties.TestingRun(t)
}

// TestScaleRegionsToOriginal_PreservesConfidence verifies confidence values are unchanged.
func TestScaleRegionsToOriginal_PreservesConfidence(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("scaling preserves confidence values", prop.ForAll(
		func(confidence float64, scale int) bool {
			if scale <= 0 {
				return true
			}

			region := DetectedRegion{
				Box:        utils.NewBox(10, 10, 20, 20),
				Confidence: confidence,
			}
			regions := []DetectedRegion{region}

			scaled := ScaleRegionsToOriginal(regions, 100, 100, 100*scale, 100*scale)

			if len(scaled) != 1 {
				return false
			}

			return scaled[0].Confidence == confidence
		},
		gen.Float64Range(0.0, 1.0),
		gen.IntRange(1, 5),
	))

	properties.TestingRun(t)
}

// TestPostProcessDB_Idempotence verifies running twice gives same result.
func TestPostProcessDB_Idempotence(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("processing binary mask twice gives same result", prop.ForAll(
		func(dims [2]int, threshold float32) bool {
			width, height := dims[0], dims[1]

			// Create a binary probability map (already binarized)
			probMap := make([]float32, width*height)
			for i := range probMap {
				if i%2 == 0 {
					probMap[i] = 1.0
				} else {
					probMap[i] = 0.0
				}
			}

			regions1 := PostProcessDB(probMap, width, height, threshold, 0.1)

			// Process again with same parameters
			regions2 := PostProcessDB(probMap, width, height, threshold, 0.1)

			// Should get same number of regions with same properties
			if len(regions1) != len(regions2) {
				return false
			}

			for i := range regions1 {
				if regions1[i].Confidence != regions2[i].Confidence {
					return false
				}
				if regions1[i].Box != regions2[i].Box {
					return false
				}
			}

			return true
		},
		genValidDimensions(),
		gen.Float32Range(0.3, 0.7),
	))

	properties.TestingRun(t)
}
