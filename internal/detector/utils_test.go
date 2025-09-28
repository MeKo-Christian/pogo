package detector

import (
	"testing"

	"github.com/MeKo-Tech/pogo/internal/utils"
)

func TestFilterRegions(t *testing.T) {
	regions := []DetectedRegion{
		{Box: utils.NewBox(0, 0, 10, 10), Confidence: 0.9},
		{Box: utils.NewBox(10, 10, 20, 20), Confidence: 0.4},
		{Box: utils.NewBox(20, 20, 30, 30), Confidence: 0.7},
		{Box: utils.NewBox(30, 30, 40, 40), Confidence: 0.2},
	}

	filtered := filterRegions(regions, 0.5)

	// Should keep regions with confidence >= 0.5
	if len(filtered) != 2 {
		t.Fatalf("expected 2 filtered regions, got %d", len(filtered))
	}

	// Check that kept regions have correct confidences
	expectedConfidences := []float64{0.9, 0.7}
	for i, region := range filtered {
		if region.Confidence != expectedConfidences[i] {
			t.Errorf("filtered region %d: expected confidence %f, got %f",
				i, expectedConfidences[i], region.Confidence)
		}
	}
}

func TestFilterRegions_EmptyInput(t *testing.T) {
	filtered := filterRegions([]DetectedRegion{}, 0.5)
	if len(filtered) != 0 {
		t.Fatalf("expected empty result for empty input, got %d regions", len(filtered))
	}
}

func TestFilterRegions_AllFiltered(t *testing.T) {
	regions := []DetectedRegion{
		{Box: utils.NewBox(0, 0, 10, 10), Confidence: 0.2},
		{Box: utils.NewBox(10, 10, 20, 20), Confidence: 0.3},
	}

	filtered := filterRegions(regions, 0.5)

	if len(filtered) != 0 {
		t.Fatalf("expected no regions to pass filter, got %d", len(filtered))
	}
}

func TestSortRegionsByConfidence(t *testing.T) {
	regions := []DetectedRegion{
		{Box: utils.NewBox(0, 0, 10, 10), Confidence: 0.5},
		{Box: utils.NewBox(10, 10, 20, 20), Confidence: 0.9},
		{Box: utils.NewBox(20, 20, 30, 30), Confidence: 0.3},
		{Box: utils.NewBox(30, 30, 40, 40), Confidence: 0.7},
	}

	indices := sortRegionsByConfidence(regions)

	// Should return indices sorted by confidence descending
	expectedOrder := []int{1, 3, 0, 2} // indices of regions with conf 0.9, 0.7, 0.5, 0.3
	if len(indices) != len(expectedOrder) {
		t.Fatalf("expected %d indices, got %d", len(expectedOrder), len(indices))
	}

	for i, idx := range indices {
		if idx != expectedOrder[i] {
			t.Errorf("index %d: expected %d, got %d", i, expectedOrder[i], idx)
		}
	}

	// Verify the sorting by checking confidence order
	for i := 1; i < len(indices); i++ {
		prevConf := regions[indices[i-1]].Confidence
		currConf := regions[indices[i]].Confidence
		if prevConf < currConf {
			t.Errorf("indices not sorted by confidence: %f < %f", prevConf, currConf)
		}
	}
}

func TestSortRegionsByConfidenceDesc(t *testing.T) {
	regions := []DetectedRegion{
		{Box: utils.NewBox(0, 0, 10, 10), Confidence: 0.5},
		{Box: utils.NewBox(10, 10, 20, 20), Confidence: 0.9},
		{Box: utils.NewBox(20, 20, 30, 30), Confidence: 0.3},
		{Box: utils.NewBox(30, 30, 40, 40), Confidence: 0.7},
	}

	// Make a copy to sort in-place
	regionsCopy := make([]DetectedRegion, len(regions))
	copy(regionsCopy, regions)

	sortRegionsByConfidenceDesc(regionsCopy)

	// Should be sorted by confidence descending
	expectedConfidences := []float64{0.9, 0.7, 0.5, 0.3}
	for i, region := range regionsCopy {
		if region.Confidence != expectedConfidences[i] {
			t.Errorf("region %d: expected confidence %f, got %f",
				i, expectedConfidences[i], region.Confidence)
		}
	}

	// Verify original array is unchanged
	if regions[0].Confidence != 0.5 {
		t.Error("original array was modified")
	}
}

func TestSortRegionsByConfidenceDescFrom(t *testing.T) {
	regions := []DetectedRegion{
		{Box: utils.NewBox(0, 0, 10, 10), Confidence: 0.5},   // index 0 - not sorted
		{Box: utils.NewBox(10, 10, 20, 20), Confidence: 0.3}, // index 1 - start sorting from here
		{Box: utils.NewBox(20, 20, 30, 30), Confidence: 0.9}, // index 2
		{Box: utils.NewBox(30, 30, 40, 40), Confidence: 0.7}, // index 3
	}

	sortRegionsByConfidenceDescFrom(regions, 1)

	// First element should be unchanged
	if regions[0].Confidence != 0.5 {
		t.Errorf("first element should be unchanged, got confidence %f", regions[0].Confidence)
	}

	// Elements from index 1 onwards should be sorted
	expectedConfidencesFromIndex1 := []float64{0.9, 0.7, 0.3}
	for i := 1; i < len(regions); i++ {
		expected := expectedConfidencesFromIndex1[i-1]
		if regions[i].Confidence != expected {
			t.Errorf("region %d: expected confidence %f, got %f",
				i, expected, regions[i].Confidence)
		}
	}
}

func TestSortRegionsByConfidenceDescFrom_OutOfBounds(t *testing.T) {
	regions := []DetectedRegion{
		{Box: utils.NewBox(0, 0, 10, 10), Confidence: 0.5},
		{Box: utils.NewBox(10, 10, 20, 20), Confidence: 0.3},
	}

	originalRegions := make([]DetectedRegion, len(regions))
	copy(originalRegions, regions)

	// Test with start index >= length
	sortRegionsByConfidenceDescFrom(regions, 2)

	// Array should be unchanged
	for i, region := range regions {
		if region.Confidence != originalRegions[i].Confidence {
			t.Errorf("region %d should be unchanged when start index is out of bounds", i)
		}
	}
}

func TestSortRegionsByConfidence_EmptyInput(t *testing.T) {
	indices := sortRegionsByConfidence([]DetectedRegion{})
	if len(indices) != 0 {
		t.Fatalf("expected empty indices for empty input, got %d", len(indices))
	}
}

func TestSortRegionsByConfidence_SingleElement(t *testing.T) {
	regions := []DetectedRegion{
		{Box: utils.NewBox(0, 0, 10, 10), Confidence: 0.5},
	}

	indices := sortRegionsByConfidence(regions)
	if len(indices) != 1 || indices[0] != 0 {
		t.Fatalf("expected single index [0], got %v", indices)
	}

	sortRegionsByConfidenceDesc(regions)
	if regions[0].Confidence != 0.5 {
		t.Error("single element should remain unchanged")
	}
}
