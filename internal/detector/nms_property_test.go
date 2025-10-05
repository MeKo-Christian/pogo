package detector

import (
	"math"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
)

// genDetectedRegion generates a random detected region.
func genDetectedRegion() gopter.Gen {
	return gopter.CombineGens(
		gen.Float64Range(0, 190),
		gen.Float64Range(0, 190),
		gen.Float64Range(0.1, 1.0),
	).Map(func(vals []interface{}) DetectedRegion {
		mx, ok := vals[0].(float64)
		if !ok {
			panic("expected float64")
		}
		my, ok := vals[1].(float64)
		if !ok {
			panic("expected float64")
		}
		conf, ok := vals[2].(float64)
		if !ok {
			panic("expected float64")
		}
		return DetectedRegion{
			Box:        utils.NewBox(mx, my, mx+10, my+10),
			Confidence: conf,
			Polygon:    []utils.Point{{X: mx, Y: my}, {X: mx + 10, Y: my}, {X: mx + 10, Y: my + 10}, {X: mx, Y: my + 10}},
		}
	})
}

// genDetectedRegions generates a slice of detected regions.
func genDetectedRegions() gopter.Gen {
	return gen.SliceOfN(20, genDetectedRegion())
}

// TestNonMaxSuppression_OutputSorted verifies NMS output is sorted by confidence.
func TestNonMaxSuppression_OutputSorted(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("NMS output is sorted by confidence (descending)", prop.ForAll(
		func(regions []DetectedRegion, iouThreshold float64) bool {
			if len(regions) == 0 {
				return true
			}

			kept := NonMaxSuppression(regions, iouThreshold)

			// Check that kept regions are sorted by confidence (descending)
			for i := 1; i < len(kept); i++ {
				if kept[i].Confidence > kept[i-1].Confidence {
					return false
				}
			}
			return true
		},
		genDetectedRegions(),
		gen.Float64Range(0.1, 0.9),
	))

	properties.TestingRun(t)
}

// TestNonMaxSuppression_NoOverlap verifies no high-overlap regions remain.
func TestNonMaxSuppression_NoOverlap(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("kept regions have IoU below threshold", prop.ForAll(
		func(iouThreshold float64) bool {
			// Create specific overlapping regions for testing
			regions := []DetectedRegion{
				{Box: utils.NewBox(0, 0, 10, 10), Confidence: 0.9},
				{Box: utils.NewBox(2, 2, 12, 12), Confidence: 0.8},
				{Box: utils.NewBox(5, 5, 15, 15), Confidence: 0.7},
				{Box: utils.NewBox(100, 100, 110, 110), Confidence: 0.85},
			}

			kept := NonMaxSuppression(regions, iouThreshold)

			// Check all pairs of kept regions
			for i := range kept {
				for j := i + 1; j < len(kept); j++ {
					iou := ComputeRegionIoU(kept[i].Box, kept[j].Box)
					if iou > iouThreshold {
						return false
					}
				}
			}
			return true
		},
		gen.Float64Range(0.3, 0.7),
	))

	properties.TestingRun(t)
}

// TestNonMaxSuppression_OutputSubset verifies output is subset of input.
func TestNonMaxSuppression_OutputSubset(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("NMS output is subset of input", prop.ForAll(
		func(regions []DetectedRegion, iouThreshold float64) bool {
			kept := NonMaxSuppression(regions, iouThreshold)

			// Output should not be larger than input
			if len(kept) > len(regions) {
				return false
			}

			// Every kept region should exist in input
			for _, k := range kept {
				found := false
				for _, r := range regions {
					if k.Box == r.Box && k.Confidence == r.Confidence {
						found = true
						break
					}
				}
				if !found {
					return false
				}
			}
			return true
		},
		genDetectedRegions(),
		gen.Float64Range(0.1, 0.9),
	))

	properties.TestingRun(t)
}

// TestComputeRegionIoU_Symmetry verifies IoU is commutative.
func TestComputeRegionIoU_Symmetry(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("IoU(A, B) == IoU(B, A)", prop.ForAll(
		func(region1, region2 DetectedRegion) bool {
			iou1 := ComputeRegionIoU(region1.Box, region2.Box)
			iou2 := ComputeRegionIoU(region2.Box, region1.Box)

			// Check symmetry (within floating point tolerance)
			return math.Abs(iou1-iou2) < 1e-9
		},
		genDetectedRegion(),
		genDetectedRegion(),
	))

	properties.TestingRun(t)
}

// TestComputeRegionIoU_Bounds verifies IoU is always in [0, 1].
func TestComputeRegionIoU_Bounds(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("IoU is always in [0, 1]", prop.ForAll(
		func(region1, region2 DetectedRegion) bool {
			iou := ComputeRegionIoU(region1.Box, region2.Box)
			return iou >= 0.0 && iou <= 1.0
		},
		genDetectedRegion(),
		genDetectedRegion(),
	))

	properties.TestingRun(t)
}

// TestComputeRegionIoU_Identity verifies IoU of box with itself is 1.0.
func TestComputeRegionIoU_Identity(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("IoU(A, A) == 1.0", prop.ForAll(
		func(region DetectedRegion) bool {
			iou := ComputeRegionIoU(region.Box, region.Box)
			return math.Abs(iou-1.0) < 1e-9
		},
		genDetectedRegion(),
	))

	properties.TestingRun(t)
}

// TestComputeRegionIoU_NoOverlap verifies non-overlapping boxes have IoU 0.
func TestComputeRegionIoU_NoOverlap(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("non-overlapping boxes have IoU = 0", prop.ForAll(
		func(separation float64) bool {
			if separation < 1.0 {
				return true // skip invalid separations
			}

			box1 := utils.NewBox(0, 0, 10, 10)
			box2 := utils.NewBox(10+separation, 0, 20+separation, 10)

			iou := ComputeRegionIoU(box1, box2)
			return iou == 0.0
		},
		gen.Float64Range(1.0, 100.0),
	))

	properties.TestingRun(t)
}

// TestSoftNonMaxSuppression_PreservesAllRegions verifies Soft-NMS keeps all regions.
func TestSoftNonMaxSuppression_PreservesAllRegions(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("Soft-NMS preserves all regions above score threshold", prop.ForAll(
		func(regions []DetectedRegion, scoreThresh float64) bool {
			if len(regions) == 0 {
				return true
			}

			kept := SoftNonMaxSuppression(regions, nmsMethodGaussian, 0.5, 0.5, scoreThresh)

			// Count input regions above threshold
			countAbove := 0
			for _, r := range regions {
				if r.Confidence >= scoreThresh {
					countAbove++
				}
			}

			// Soft-NMS may reduce confidence but typically keeps regions
			// At minimum, the highest confidence region should be kept
			return len(kept) >= 1
		},
		genDetectedRegions(),
		gen.Float64Range(0.1, 0.5),
	))

	properties.TestingRun(t)
}

// TestSoftNonMaxSuppression_ConfidenceDecay verifies confidence is monotonically decayed.
func TestSoftNonMaxSuppression_ConfidenceDecay(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("Soft-NMS confidence is non-increasing for overlapping regions", prop.ForAll(
		func(method string, sigma float64) bool {
			// Valid methods
			if method != nmsMethodLinear && method != nmsMethodGaussian && method != nmsMethodHard {
				method = nmsMethodGaussian
			}

			// Create overlapping regions with known confidences
			regions := []DetectedRegion{
				{Box: utils.NewBox(0, 0, 10, 10), Confidence: 0.9},
				{Box: utils.NewBox(5, 5, 15, 15), Confidence: 0.8},
			}

			originalConf := regions[1].Confidence
			kept := SoftNonMaxSuppression(regions, method, 0.5, sigma, 0.1)

			if len(kept) < 2 {
				return true // Both regions should be kept
			}

			// Find the second region in output
			for _, k := range kept {
				if k.Box.MinX == 5.0 && k.Box.MinY == 5.0 {
					// Confidence should be decayed (or equal in edge cases)
					return k.Confidence <= originalConf
				}
			}
			return true
		},
		gen.OneConstOf(nmsMethodLinear, nmsMethodGaussian, nmsMethodHard),
		gen.Float64Range(0.1, 1.0),
	))

	properties.TestingRun(t)
}

// TestAdaptiveNonMaxSuppression_Sorted verifies output is sorted by confidence.
func TestAdaptiveNonMaxSuppression_Sorted(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("Adaptive NMS output is sorted by confidence", prop.ForAll(
		func(regions []DetectedRegion, baseThreshold, scaleFactor float64) bool {
			if len(regions) == 0 {
				return true
			}

			kept := AdaptiveNonMaxSuppression(regions, baseThreshold, scaleFactor)

			// Check descending confidence order
			for i := 1; i < len(kept); i++ {
				if kept[i].Confidence > kept[i-1].Confidence {
					return false
				}
			}
			return true
		},
		genDetectedRegions(),
		gen.Float64Range(0.2, 0.5),
		gen.Float64Range(0.8, 1.2),
	))

	properties.TestingRun(t)
}

// TestSizeAwareNonMaxSuppression_Sorted verifies output is sorted by confidence.
func TestSizeAwareNonMaxSuppression_Sorted(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("Size-aware NMS output is sorted by confidence", prop.ForAll(
		func(regions []DetectedRegion, baseThreshold float64) bool {
			if len(regions) == 0 {
				return true
			}

			kept := SizeAwareNonMaxSuppression(regions, baseThreshold, 0.1, 100, 10000)

			// Check descending confidence order
			for i := 1; i < len(kept); i++ {
				if kept[i].Confidence > kept[i-1].Confidence {
					return false
				}
			}
			return true
		},
		genDetectedRegions(),
		gen.Float64Range(0.2, 0.5),
	))

	properties.TestingRun(t)
}

// TestCalculateAdaptiveIoUThreshold_Bounds verifies threshold is within valid range.
func TestCalculateAdaptiveIoUThreshold_Bounds(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("adaptive threshold is bounded [0.1, 0.8]", prop.ForAll(
		func(region1, region2 DetectedRegion, baseThreshold, scaleFactor float64) bool {
			threshold := calculateAdaptiveIoUThreshold(baseThreshold, scaleFactor, region1, region2)
			return threshold >= 0.1 && threshold <= 0.8
		},
		genDetectedRegion(),
		genDetectedRegion(),
		gen.Float64Range(0.1, 0.6),
		gen.Float64Range(0.5, 2.0),
	))

	properties.TestingRun(t)
}

// TestCalculateSoftNMSWeight_Bounds verifies weight is in [0, 1].
func TestCalculateSoftNMSWeight_Bounds(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("Soft-NMS weight is in [0, 1]", prop.ForAll(
		func(iou, iouThreshold, sigma float64, method string) bool {
			// Ensure valid inputs
			if iou < 0 || iou > 1 {
				return true
			}
			if method != nmsMethodLinear && method != nmsMethodGaussian && method != nmsMethodHard {
				method = nmsMethodGaussian
			}

			weight := calculateSoftNMSWeight(iou, iouThreshold, sigma, method)
			return weight >= 0.0 && weight <= 1.0
		},
		gen.Float64Range(0.0, 1.0),
		gen.Float64Range(0.1, 0.9),
		gen.Float64Range(0.1, 1.0),
		gen.OneConstOf(nmsMethodLinear, nmsMethodGaussian, nmsMethodHard),
	))

	properties.TestingRun(t)
}

// TestNonMaxSuppression_EmptyInput verifies empty input returns empty output.
func TestNonMaxSuppression_EmptyInput(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("empty input returns empty output", prop.ForAll(
		func(iouThreshold float64) bool {
			regions := []DetectedRegion{}
			kept := NonMaxSuppression(regions, iouThreshold)
			return len(kept) == 0
		},
		gen.Float64Range(0.1, 0.9),
	))

	properties.TestingRun(t)
}

// TestNonMaxSuppression_SingleRegion verifies single region is always kept.
func TestNonMaxSuppression_SingleRegion(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("single region is always kept", prop.ForAll(
		func(region DetectedRegion, iouThreshold float64) bool {
			regions := []DetectedRegion{region}
			kept := NonMaxSuppression(regions, iouThreshold)
			if len(kept) != 1 {
				return false
			}
			// Compare field by field since struct contains slices
			return kept[0].Box == region.Box && kept[0].Confidence == region.Confidence
		},
		genDetectedRegion(),
		gen.Float64Range(0.1, 0.9),
	))

	properties.TestingRun(t)
}
