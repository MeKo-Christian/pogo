package detector

import (
	"sort"
)

// filterRegions filters regions by minimum confidence threshold.
func filterRegions(regions []DetectedRegion, minConf float64) []DetectedRegion {
	var filtered []DetectedRegion
	for _, r := range regions {
		if r.Confidence >= minConf {
			filtered = append(filtered, r)
		}
	}
	return filtered
}

// sortRegionsByConfidence returns indices of regions sorted by confidence (descending).
func sortRegionsByConfidence(regions []DetectedRegion) []int {
	indices := make([]int, len(regions))
	for i := range indices {
		indices[i] = i
	}

	sort.Slice(indices, func(i, j int) bool {
		return regions[indices[i]].Confidence > regions[indices[j]].Confidence
	})

	return indices
}

// sortRegionsByConfidenceDesc sorts regions in-place by confidence (descending).
func sortRegionsByConfidenceDesc(regions []DetectedRegion) {
	sort.Slice(regions, func(i, j int) bool {
		return regions[i].Confidence > regions[j].Confidence
	})
}

// sortRegionsByConfidenceDescFrom sorts regions in-place by confidence (descending) starting from index.
func sortRegionsByConfidenceDescFrom(regions []DetectedRegion, start int) {
	if start >= len(regions) {
		return
	}

	sort.Slice(regions[start:], func(i, j int) bool {
		return regions[start+i].Confidence > regions[start+j].Confidence
	})
}