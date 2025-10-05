package detector

import (
	"math"
	"sort"

	"github.com/MeKo-Tech/pogo/internal/utils"
)

const (
	nmsMethodHard     = "hard"
	nmsMethodGaussian = "gaussian"
	nmsMethodLinear   = "linear"
)

// AdaptiveNMSThresholds contains tunable parameters for adaptive NMS.
type AdaptiveNMSThresholds struct {
	BaseThreshold float64 // Base IoU threshold
	ScaleFactor   float64 // Overall scaling factor
	MinThreshold  float64 // Minimum allowed threshold
	MaxThreshold  float64 // Maximum allowed threshold
}

// DefaultAdaptiveNMSThresholds returns default adaptive NMS parameters.
func DefaultAdaptiveNMSThresholds() AdaptiveNMSThresholds {
	return AdaptiveNMSThresholds{
		BaseThreshold: 0.3,
		ScaleFactor:   1.0,
		MinThreshold:  0.1,
		MaxThreshold:  0.8,
	}
}

// NonMaxSuppression performs standard Non-Maximum Suppression.
func NonMaxSuppression(regions []DetectedRegion, iouThreshold float64) []DetectedRegion {
	if len(regions) <= 1 {
		return regions
	}

	// Sort regions by confidence (descending)
	indices := sortRegionsByConfidence(regions)
	suppressed := make([]bool, len(regions))
	kept := make([]DetectedRegion, 0, len(regions))

	for _, a := range indices {
		if suppressed[a] {
			continue
		}
		kept = append(kept, regions[a])

		// Suppress overlapping regions with lower confidence
		for _, b := range indices {
			if suppressed[b] || a == b {
				continue
			}
			if ComputeRegionIoU(regions[a].Box, regions[b].Box) > iouThreshold {
				suppressed[b] = true
			}
		}
	}

	return kept
}

// AdaptiveNonMaxSuppression performs greedy NMS with adaptive IoU thresholds.
func AdaptiveNonMaxSuppression(regions []DetectedRegion, baseThreshold, scaleFactor float64) []DetectedRegion {
	if len(regions) <= 1 {
		return regions
	}

	// Sort regions by confidence (descending)
	sortRegionsByConfidenceDesc(regions)
	suppressed := make([]bool, len(regions))
	kept := make([]DetectedRegion, 0, len(regions))

	for a := range regions {
		if suppressed[a] {
			continue
		}
		kept = append(kept, regions[a])

		// Suppress overlapping regions with adaptive thresholds
		for b := a + 1; b < len(regions); b++ {
			if suppressed[b] {
				continue
			}
			// Use adaptive threshold based on region characteristics
			adaptiveThreshold := calculateAdaptiveIoUThreshold(baseThreshold, scaleFactor, regions[a], regions[b])
			if ComputeRegionIoU(regions[a].Box, regions[b].Box) > adaptiveThreshold {
				suppressed[b] = true
			}
		}
	}

	return kept
}

// SizeAwareNonMaxSuppression performs greedy NMS with size-based adaptive thresholds.
func SizeAwareNonMaxSuppression(regions []DetectedRegion, baseThreshold, sizeScaleFactor float64,
	minSize, maxSize int,
) []DetectedRegion {
	if len(regions) <= 1 {
		return regions
	}

	// Sort regions by confidence (descending)
	sortRegionsByConfidenceDesc(regions)
	suppressed := make([]bool, len(regions))
	kept := make([]DetectedRegion, 0, len(regions))

	for a := range regions {
		if suppressed[a] {
			continue
		}
		kept = append(kept, regions[a])

		// Suppress overlapping regions with size-based thresholds
		for b := a + 1; b < len(regions); b++ {
			if suppressed[b] {
				continue
			}
			// Use size-based adaptive threshold
			adaptiveThreshold := calculateSizeBasedIoUThreshold(baseThreshold, sizeScaleFactor,
				minSize, maxSize, regions[a], regions[b])
			if ComputeRegionIoU(regions[a].Box, regions[b].Box) > adaptiveThreshold {
				suppressed[b] = true
			}
		}
	}

	return kept
}

// SoftNonMaxSuppression performs Soft-NMS with configurable decay functions.
func SoftNonMaxSuppression(regions []DetectedRegion, method string,
	iouThreshold, sigma, scoreThresh float64,
) []DetectedRegion {
	n := len(regions)
	if n <= 1 {
		return regions
	}

	// Handle edge cases
	if handled := handleEdgeCases(regions, scoreThresh); handled != nil {
		return handled
	}

	// Create working copy and apply Soft-NMS
	regs := make([]DetectedRegion, n)
	copy(regs, regions)
	applySoftNMS(&regs, iouThreshold, sigma, method)

	// Filter and sort results
	return filterAndSortResults(regs, scoreThresh)
}

// calculateAdaptiveIoUThreshold computes an adaptive IoU threshold based on region characteristics.
func calculateAdaptiveIoUThreshold(baseThreshold, scaleFactor float64, regionA, regionB DetectedRegion) float64 {
	// Calculate size-based adjustment
	sizeA := regionA.Box.Width() * regionA.Box.Height()
	sizeB := regionB.Box.Width() * regionB.Box.Height()
	avgSize := (sizeA + sizeB) / 2.0

	// Normalize size (simple heuristic based on typical text region sizes)
	normalizedSize := math.Min(1.0, avgSize/10000.0) // Assume 100x100 as reference

	// Calculate confidence-based adjustment
	avgConf := (regionA.Confidence + regionB.Confidence) / 2.0

	// Adaptive threshold: base + size adjustment + confidence adjustment
	adaptiveThreshold := baseThreshold * scaleFactor
	adaptiveThreshold += 0.1 * normalizedSize   // Larger regions can have higher IoU tolerance
	adaptiveThreshold -= 0.05 * (avgConf - 0.5) // Higher confidence regions can be more strict

	// Clamp to reasonable bounds
	if adaptiveThreshold < 0.1 {
		adaptiveThreshold = 0.1
	}
	if adaptiveThreshold > 0.8 {
		adaptiveThreshold = 0.8
	}

	return adaptiveThreshold
}

// calculateSizeBasedIoUThreshold computes size-aware IoU threshold.
func calculateSizeBasedIoUThreshold(baseThreshold, sizeScaleFactor float64,
	minSize, maxSize int, regionA, regionB DetectedRegion,
) float64 {
	sizeA := regionA.Box.Width() * regionA.Box.Height()
	sizeB := regionB.Box.Width() * regionB.Box.Height()
	avgSize := (sizeA + sizeB) / 2.0

	// Normalize size between min and max
	sizeRange := float64(maxSize - minSize)
	if sizeRange <= 0 {
		return baseThreshold
	}

	normalizedSize := (avgSize - float64(minSize)) / sizeRange
	normalizedSize = math.Max(0, math.Min(1, normalizedSize)) // Clamp to [0,1]

	// Smaller regions get stricter thresholds (less tolerance for overlap)
	// Larger regions get more lenient thresholds (more tolerance for overlap)
	sizeAdjustment := sizeScaleFactor * (normalizedSize - 0.5)

	return baseThreshold + sizeAdjustment
}

// calculateSoftNMSWeight calculates the weight decay for Soft-NMS.
func calculateSoftNMSWeight(iou, iouThreshold, sigma float64, method string) float64 {
	switch method {
	case "linear":
		if iou >= iouThreshold {
			return 1.0 - iou
		}
		return 1.0
	case "gaussian":
		return math.Exp(-(iou * iou) / sigma)
	default: // hard
		if iou >= iouThreshold {
			return 0.0
		}
		return 1.0
	}
}

// handleEdgeCases handles edge cases for Soft-NMS.
func handleEdgeCases(regions []DetectedRegion, scoreThresh float64) []DetectedRegion {
	// Filter regions that already meet the score threshold
	var validRegions []DetectedRegion
	for _, r := range regions {
		if r.Confidence >= scoreThresh {
			validRegions = append(validRegions, r)
		}
	}
	if len(validRegions) <= 1 {
		return validRegions
	}
	return nil // No edge case, continue with normal processing
}

// applySoftNMS applies the Soft-NMS algorithm to regions.
func applySoftNMS(regs *[]DetectedRegion, iouThreshold, sigma float64, method string) {
	n := len(*regs)
	for i := range n {
		// Find the region with highest confidence in remaining regions
		maxIdx := i
		for j := i + 1; j < n; j++ {
			if (*regs)[j].Confidence > (*regs)[maxIdx].Confidence {
				maxIdx = j
			}
		}
		// Swap to bring highest confidence region to position i
		(*regs)[i], (*regs)[maxIdx] = (*regs)[maxIdx], (*regs)[i]

		// Apply decay to overlapping regions
		decayOverlappingRegions(*regs, i, iouThreshold, sigma, method)
	}
}

// decayOverlappingRegions applies confidence decay to regions overlapping with the selected region.
func decayOverlappingRegions(regs []DetectedRegion, i int,
	iouThreshold, sigma float64, method string,
) {
	for j := i + 1; j < len(regs); j++ {
		iou := ComputeRegionIoU(regs[i].Box, regs[j].Box)
		weight := calculateSoftNMSWeight(iou, iouThreshold, sigma, method)
		regs[j].Confidence *= weight
	}
}

// filterAndSortResults filters regions by score threshold and sorts by confidence.
func filterAndSortResults(regs []DetectedRegion, scoreThresh float64) []DetectedRegion {
	var filtered []DetectedRegion
	for _, r := range regs {
		if r.Confidence >= scoreThresh {
			filtered = append(filtered, r)
		}
	}

	// Sort by confidence descending
	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].Confidence > filtered[j].Confidence
	})

	return filtered
}

// ComputeRegionIoU computes Intersection over Union (IoU) for two regions.
func ComputeRegionIoU(a, b utils.Box) float64 {
	// Calculate intersection
	intersectionLeft := math.Max(a.MinX, b.MinX)
	intersectionTop := math.Max(a.MinY, b.MinY)
	intersectionRight := math.Min(a.MaxX, b.MaxX)
	intersectionBottom := math.Min(a.MaxY, b.MaxY)

	if intersectionLeft >= intersectionRight || intersectionTop >= intersectionBottom {
		return 0.0
	}

	intersectionArea := (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop)
	areaA := a.Width() * a.Height()
	areaB := b.Width() * b.Height()
	unionArea := areaA + areaB - intersectionArea

	if unionArea <= 0 {
		return 0.0
	}

	return intersectionArea / unionArea
}
