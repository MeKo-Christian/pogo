package pdf

import (
	"errors"
	"math"
	"sort"
	"strings"
)

// HybridResult represents the combined result of vector text extraction and OCR.
type HybridResult struct {
	PageNumber     int                  `json:"page_number"`
	VectorText     *TextExtraction      `json:"vector_text,omitempty"`
	OCRResults     []ImageResult        `json:"ocr_results,omitempty"`
	CombinedText   string               `json:"combined_text"`
	MergedRegions  []OCRRegion          `json:"merged_regions"`
	ProcessingInfo HybridProcessingInfo `json:"processing_info"`
	QualityMetrics HybridQualityMetrics `json:"quality_metrics"`
}

// HybridProcessingInfo contains information about the hybrid processing.
type HybridProcessingInfo struct {
	Strategy             ProcessingStrategy `json:"strategy"`
	VectorTextUsed       bool               `json:"vector_text_used"`
	OCRUsed              bool               `json:"ocr_used"`
	MergeMethod          string             `json:"merge_method"`
	ConfidenceThreshold  float64            `json:"confidence_threshold"`
	SpatialMergingUsed   bool               `json:"spatial_merging_used"`
	DeduplicationApplied bool               `json:"deduplication_applied"`
}

// HybridQualityMetrics contains quality metrics for the hybrid result.
type HybridQualityMetrics struct {
	OverallScore      float64 `json:"overall_score"`
	VectorTextContrib float64 `json:"vector_text_contribution"`
	OCRContrib        float64 `json:"ocr_contribution"`
	DuplicationLevel  float64 `json:"duplication_level"`
	TextCoverage      float64 `json:"text_coverage"`
	ConfidenceAverage float64 `json:"confidence_average"`
}

// MergeStrategy represents different strategies for combining vector text and OCR results.
type MergeStrategy int

const (
	// MergeStrategyAppend simply appends OCR text to vector text.
	MergeStrategyAppend MergeStrategy = iota
	// MergeStrategySpatial uses spatial information to merge texts by position.
	MergeStrategySpatial
	// MergeStrategyConfidence uses confidence scores to choose the best text.
	MergeStrategyConfidence
	// MergeStrategyHybridSmart uses multiple criteria for intelligent merging.
	MergeStrategyHybridSmart
)

// String returns the string representation of the merge strategy.
func (s MergeStrategy) String() string {
	switch s {
	case MergeStrategyAppend:
		return "append"
	case MergeStrategySpatial:
		return "spatial"
	case MergeStrategyConfidence:
		return "confidence"
	case MergeStrategyHybridSmart:
		return "hybrid_smart"
	default:
		return "unknown"
	}
}

// HybridConfig contains configuration for hybrid processing.
type HybridConfig struct {
	MergeStrategy           MergeStrategy `json:"merge_strategy"`
	ConfidenceThreshold     float64       `json:"confidence_threshold"`
	SpatialOverlapThreshold float64       `json:"spatial_overlap_threshold"`
	DeduplicationEnabled    bool          `json:"deduplication_enabled"`
	DeduplicationSimilarity float64       `json:"deduplication_similarity"`
	PreferVectorText        bool          `json:"prefer_vector_text"`
	MinOCRConfidence        float64       `json:"min_ocr_confidence"`
}

// DefaultHybridConfig returns the default hybrid processing configuration.
func DefaultHybridConfig() *HybridConfig {
	return &HybridConfig{
		MergeStrategy:           MergeStrategyHybridSmart,
		ConfidenceThreshold:     0.5,
		SpatialOverlapThreshold: 0.1,
		DeduplicationEnabled:    true,
		DeduplicationSimilarity: 0.8,
		PreferVectorText:        true,
		MinOCRConfidence:        0.3,
	}
}

// HybridProcessor combines vector text extraction with OCR results.
type HybridProcessor struct {
	config *HybridConfig
}

// NewHybridProcessor creates a new hybrid processor with the given configuration.
func NewHybridProcessor(config *HybridConfig) *HybridProcessor {
	if config == nil {
		config = DefaultHybridConfig()
	}

	return &HybridProcessor{
		config: config,
	}
}

// MergeResults combines vector text extraction with OCR results for a single page.
func (h *HybridProcessor) MergeResults(
	vectorText *TextExtraction,
	ocrResults []ImageResult,
	pageWidth, pageHeight float64,
) (*HybridResult, error) {
	// Validate inputs
	if vectorText == nil && len(ocrResults) == 0 {
		return nil, errors.New("no input data provided for hybrid processing")
	}

	result := &HybridResult{
		VectorText: vectorText,
		OCRResults: ocrResults,
		ProcessingInfo: HybridProcessingInfo{
			Strategy:             h.determineStrategy(vectorText, ocrResults),
			VectorTextUsed:       vectorText != nil,
			OCRUsed:              len(ocrResults) > 0,
			MergeMethod:          h.config.MergeStrategy.String(),
			ConfidenceThreshold:  h.config.ConfidenceThreshold,
			SpatialMergingUsed:   h.config.MergeStrategy == MergeStrategySpatial || h.config.MergeStrategy == MergeStrategyHybridSmart,
			DeduplicationApplied: h.config.DeduplicationEnabled,
		},
	}

	if vectorText != nil {
		result.PageNumber = vectorText.PageNumber
	} else if len(ocrResults) > 0 {
		// Assume all OCR results are from the same page
		result.PageNumber = 1 // Default page number
	}

	// Perform the merge based on strategy
	switch h.config.MergeStrategy {
	case MergeStrategyAppend:
		h.mergeByAppending(result)
	case MergeStrategySpatial:
		h.mergeBySpatialLayout(result, pageWidth, pageHeight)
	case MergeStrategyConfidence:
		h.mergeByConfidence(result)
	case MergeStrategyHybridSmart:
		h.mergeWithSmartStrategy(result, pageWidth, pageHeight)
	default:
		h.mergeByAppending(result)
	}

	// Apply deduplication if enabled
	if h.config.DeduplicationEnabled {
		h.deduplicateText(result)
	}

	// Calculate quality metrics
	h.calculateQualityMetrics(result)

	return result, nil
}

// determineStrategy determines the processing strategy based on available data.
func (h *HybridProcessor) determineStrategy(vectorText *TextExtraction, ocrResults []ImageResult) ProcessingStrategy {
	hasVectorText := vectorText != nil && vectorText.Quality.HasText
	hasOCRResults := len(ocrResults) > 0

	switch {
	case hasVectorText && hasOCRResults:
		return StrategyHybrid
	case hasVectorText:
		return StrategyVectorText
	case hasOCRResults:
		return StrategyOCR
	default:
		return StrategySkip
	}
}

// mergeByAppending simply appends all text sources.
func (h *HybridProcessor) mergeByAppending(result *HybridResult) {
	var texts []string

	// Add vector text if available
	if result.VectorText != nil && result.VectorText.Text != "" {
		texts = append(texts, result.VectorText.Text)
	}

	// Add OCR text
	for _, ocrResult := range result.OCRResults {
		// Use OCRRegions if available (from full pipeline), otherwise convert from Regions
		if len(ocrResult.OCRRegions) > 0 {
			for _, region := range ocrResult.OCRRegions {
				if region.RecConfidence >= h.config.MinOCRConfidence {
					texts = append(texts, region.Text)
					result.MergedRegions = append(result.MergedRegions, region)
				}
			}
		} else {
			// Fallback: create OCRRegions from DetectedRegions (no text available)
			for _, detRegion := range ocrResult.Regions {
				if detRegion.Confidence >= h.config.MinOCRConfidence {
					// Create OCRRegion from DetectedRegion
					ocrRegion := OCRRegion{
						Box: struct{ X, Y, W, H int }{
							X: int(detRegion.Box.MinX),
							Y: int(detRegion.Box.MinY),
							W: int(detRegion.Box.MaxX - detRegion.Box.MinX),
							H: int(detRegion.Box.MaxY - detRegion.Box.MinY),
						},
						DetConfidence: detRegion.Confidence,
						Text:          "", // No text available from detector only
						RecConfidence: 0,  // No recognition performed
					}
					result.MergedRegions = append(result.MergedRegions, ocrRegion)
				}
			}
		}
	}

	result.CombinedText = strings.Join(texts, "\n")
}

// TextElement represents a text element with position information for spatial merging.
type TextElement struct {
	Text       string
	X, Y       float64
	Width      float64
	Height     float64
	Confidence float64
	Source     string // "vector" or "ocr"
	Region     *OCRRegion
}

// mergeBySpatialLayout merges text based on spatial positioning.
func (h *HybridProcessor) mergeBySpatialLayout(result *HybridResult, pageWidth, pageHeight float64) {
	// Create a list of all text elements with positions

	var elements []TextElement

	// Add vector text elements
	if result.VectorText != nil {
		for _, pos := range result.VectorText.Positions {
			elements = append(elements, TextElement{
				Text:       pos.Text,
				X:          pos.X,
				Y:          pos.Y,
				Width:      pos.Width,
				Height:     pos.Height,
				Confidence: 1.0, // Vector text has maximum confidence
				Source:     "vector",
			})
		}
	}

	// Add OCR elements
	for _, ocrResult := range result.OCRResults {
		// Use OCRRegions if available (from full pipeline)
		if len(ocrResult.OCRRegions) > 0 {
			for _, region := range ocrResult.OCRRegions {
				if region.RecConfidence >= h.config.MinOCRConfidence {
					// Convert region coordinates to text element
					centerX := float64(region.Box.X + region.Box.W/2)
					centerY := float64(region.Box.Y + region.Box.H/2)
					width := float64(region.Box.W)
					height := float64(region.Box.H)

					elements = append(elements, TextElement{
						Text:       region.Text,
						X:          centerX,
						Y:          centerY,
						Width:      width,
						Height:     height,
						Confidence: region.RecConfidence,
						Source:     "ocr",
						Region:     &region,
					})
				}
			}
		} else {
			// Fallback: use DetectedRegions (no text available)
			for _, detRegion := range ocrResult.Regions {
				if detRegion.Confidence >= h.config.MinOCRConfidence {
					centerX := (detRegion.Box.MinX + detRegion.Box.MaxX) / 2
					centerY := (detRegion.Box.MinY + detRegion.Box.MaxY) / 2
					width := detRegion.Box.MaxX - detRegion.Box.MinX
					height := detRegion.Box.MaxY - detRegion.Box.MinY

					// Create OCRRegion from DetectedRegion
					ocrRegion := OCRRegion{
						Box: struct{ X, Y, W, H int }{
							X: int(detRegion.Box.MinX),
							Y: int(detRegion.Box.MinY),
							W: int(width),
							H: int(height),
						},
						DetConfidence: detRegion.Confidence,
						Text:          "", // No text available
						RecConfidence: 0,
					}

					elements = append(elements, TextElement{
						Text:       "", // No text available from detector only
						X:          centerX,
						Y:          centerY,
						Width:      width,
						Height:     height,
						Confidence: detRegion.Confidence,
						Source:     "ocr",
						Region:     &ocrRegion,
					})
				}
			}
		}
	}

	// Sort elements by reading order (top to bottom, left to right)
	sort.Slice(elements, func(i, j int) bool {
		// Primary sort by Y coordinate (top to bottom)
		if math.Abs(elements[i].Y-elements[j].Y) > elements[i].Height/2 {
			return elements[i].Y < elements[j].Y
		}
		// Secondary sort by X coordinate (left to right)
		return elements[i].X < elements[j].X
	})

	// Merge elements, avoiding duplicates in overlapping areas
	var mergedElements []TextElement
	for _, element := range elements {
		shouldAdd := true

		// Check for spatial overlap with existing elements
		for _, existing := range mergedElements {
			if h.elementsOverlap(element, existing) {
				// If elements overlap, prefer vector text or higher confidence
				if (h.config.PreferVectorText && existing.Source == "vector") ||
					(!h.config.PreferVectorText && existing.Confidence > element.Confidence) {
					shouldAdd = false
					break
				} else {
					// Replace existing element with current one
					for i, e := range mergedElements {
						if e.X == existing.X && e.Y == existing.Y {
							mergedElements[i] = element
							shouldAdd = false
							break
						}
					}
				}
			}
		}

		if shouldAdd {
			mergedElements = append(mergedElements, element)
		}
	}

	// Build combined text and regions
	texts := make([]string, 0, len(mergedElements))
	for _, element := range mergedElements {
		texts = append(texts, element.Text)
		if element.Region != nil {
			result.MergedRegions = append(result.MergedRegions, *element.Region)
		}
	}

	result.CombinedText = strings.Join(texts, " ")
}

// mergeByConfidence merges text based on confidence scores.
func (h *HybridProcessor) mergeByConfidence(result *HybridResult) {
	var highConfidenceTexts []string

	// Add vector text (always high confidence)
	if result.VectorText != nil && result.VectorText.Text != "" {
		highConfidenceTexts = append(highConfidenceTexts, result.VectorText.Text)
	}

	// Add high-confidence OCR text
	for _, ocrResult := range result.OCRResults {
		// Use OCRRegions if available (from full pipeline)
		if len(ocrResult.OCRRegions) > 0 {
			for _, region := range ocrResult.OCRRegions {
				if region.RecConfidence >= h.config.ConfidenceThreshold {
					highConfidenceTexts = append(highConfidenceTexts, region.Text)
					result.MergedRegions = append(result.MergedRegions, region)
				}
			}
		}
	}

	result.CombinedText = strings.Join(highConfidenceTexts, " ")
}

// mergeWithSmartStrategy uses multiple criteria for intelligent merging.
func (h *HybridProcessor) mergeWithSmartStrategy(result *HybridResult, pageWidth, pageHeight float64) {
	// First, try spatial merging
	h.mergeBySpatialLayout(result, pageWidth, pageHeight)

	// Then apply confidence filtering
	if result.VectorText == nil || result.VectorText.Quality.Score < 0.8 {
		// If vector text quality is poor, enhance with high-confidence OCR
		additionalTexts := []string{result.CombinedText}

		for _, ocrResult := range result.OCRResults {
			// Use OCRRegions if available (from full pipeline)
			if len(ocrResult.OCRRegions) > 0 {
				for _, region := range ocrResult.OCRRegions {
					if region.RecConfidence >= h.config.ConfidenceThreshold {
						// Check if this text is already included
						if !h.textAlreadyIncluded(region.Text, result.CombinedText) {
							additionalTexts = append(additionalTexts, region.Text)
							if !h.regionAlreadyIncluded(region, result.MergedRegions) {
								result.MergedRegions = append(result.MergedRegions, region)
							}
						}
					}
				}
			}
		}

		result.CombinedText = strings.Join(additionalTexts, " ")
	}
}

// elementsOverlap checks if two text elements spatially overlap.
func (h *HybridProcessor) elementsOverlap(a, b TextElement) bool {
	// Calculate overlap area
	overlapX := math.Max(0, math.Min(a.X+a.Width/2, b.X+b.Width/2)-math.Max(a.X-a.Width/2, b.X-b.Width/2))
	overlapY := math.Max(0, math.Min(a.Y+a.Height/2, b.Y+b.Height/2)-math.Max(a.Y-a.Height/2, b.Y-b.Height/2))
	overlapArea := overlapX * overlapY

	// Calculate areas
	areaA := a.Width * a.Height
	areaB := b.Width * b.Height
	minArea := math.Min(areaA, areaB)

	// Check if overlap exceeds threshold
	if minArea > 0 {
		overlapRatio := overlapArea / minArea
		return overlapRatio > h.config.SpatialOverlapThreshold
	}

	return false
}

// deduplicateText removes duplicate or highly similar text segments.
func (h *HybridProcessor) deduplicateText(result *HybridResult) {
	if result.CombinedText == "" {
		return
	}

	// Split text into lines or sentences for deduplication
	lines := strings.Split(result.CombinedText, "\n")
	var uniqueLines []string

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		isDuplicate := false
		for _, existing := range uniqueLines {
			if h.textSimilarity(line, existing) > h.config.DeduplicationSimilarity {
				isDuplicate = true
				break
			}
		}

		if !isDuplicate {
			uniqueLines = append(uniqueLines, line)
		}
	}

	result.CombinedText = strings.Join(uniqueLines, "\n")
}

// textSimilarity calculates similarity between two text strings (0-1).
func (h *HybridProcessor) textSimilarity(a, b string) float64 {
	// Simple similarity based on common words
	wordsA := strings.Fields(strings.ToLower(a))
	wordsB := strings.Fields(strings.ToLower(b))

	if len(wordsA) == 0 && len(wordsB) == 0 {
		return 1.0
	}

	if len(wordsA) == 0 || len(wordsB) == 0 {
		return 0.0
	}

	// Count common words
	commonWords := 0
	wordMapA := make(map[string]bool)
	for _, word := range wordsA {
		wordMapA[word] = true
	}

	for _, word := range wordsB {
		if wordMapA[word] {
			commonWords++
		}
	}

	// Calculate Jaccard similarity
	totalWords := len(wordsA) + len(wordsB) - commonWords
	if totalWords == 0 {
		return 1.0
	}

	return float64(commonWords) / float64(totalWords)
}

// textAlreadyIncluded checks if text is already included in the combined text.
func (h *HybridProcessor) textAlreadyIncluded(text, combinedText string) bool {
	return strings.Contains(strings.ToLower(combinedText), strings.ToLower(strings.TrimSpace(text)))
}

// regionAlreadyIncluded checks if a region is already in the merged regions list.
func (h *HybridProcessor) regionAlreadyIncluded(region OCRRegion, regions []OCRRegion) bool {
	for _, existing := range regions {
		if existing.Text == region.Text &&
			existing.Box.X == region.Box.X &&
			existing.Box.Y == region.Box.Y {
			return true
		}
	}
	return false
}

// calculateQualityMetrics calculates quality metrics for the hybrid result.
func (h *HybridProcessor) calculateQualityMetrics(result *HybridResult) {
	metrics := &HybridQualityMetrics{}

	// Calculate contributions
	vectorTextLen := 0.0
	ocrTextLen := 0.0

	if result.VectorText != nil {
		vectorTextLen = float64(len(result.VectorText.Text))
	}

	for _, ocrResult := range result.OCRResults {
		// Use OCRRegions if available (from full pipeline)
		if len(ocrResult.OCRRegions) > 0 {
			for _, region := range ocrResult.OCRRegions {
				ocrTextLen += float64(len(region.Text))
			}
		}
		// Note: DetectedRegions don't have text, so we skip them for text length calculation
	}

	totalTextLen := vectorTextLen + ocrTextLen
	if totalTextLen > 0 {
		metrics.VectorTextContrib = vectorTextLen / totalTextLen
		metrics.OCRContrib = ocrTextLen / totalTextLen
	}

	// Calculate overall score
	metrics.OverallScore = h.calculateOverallScore(result)

	// Calculate text coverage
	metrics.TextCoverage = h.calculateTextCoverage(result)

	// Calculate average confidence
	metrics.ConfidenceAverage = h.calculateAverageConfidence(result)

	// Calculate duplication level
	metrics.DuplicationLevel = h.calculateDuplicationLevel(result)

	result.QualityMetrics = *metrics
}

// calculateOverallScore calculates the overall quality score.
func (h *HybridProcessor) calculateOverallScore(result *HybridResult) float64 {
	score := 0.0

	// Base score from vector text
	if result.VectorText != nil {
		score += result.VectorText.Quality.Score * 0.6
	}

	// Add score from OCR confidence
	if len(result.MergedRegions) > 0 {
		totalConfidence := 0.0
		for _, region := range result.MergedRegions {
			totalConfidence += region.RecConfidence
		}
		avgConfidence := totalConfidence / float64(len(result.MergedRegions))
		score += avgConfidence * 0.4
	}

	return math.Min(score, 1.0)
}

// calculateTextCoverage calculates the text coverage metric.
func (h *HybridProcessor) calculateTextCoverage(result *HybridResult) float64 {
	// This is a simplified implementation
	// In a full implementation, we would calculate actual spatial coverage
	if len(result.CombinedText) > 0 {
		return math.Min(float64(len(result.CombinedText))/1000.0, 1.0)
	}
	return 0.0
}

// calculateAverageConfidence calculates the average confidence across all sources.
func (h *HybridProcessor) calculateAverageConfidence(result *HybridResult) float64 {
	totalConfidence := 0.0
	count := 0

	// Vector text confidence
	if result.VectorText != nil {
		totalConfidence += result.VectorText.Quality.Score
		count++
	}

	// OCR confidence
	for _, region := range result.MergedRegions {
		totalConfidence += region.RecConfidence
		count++
	}

	if count > 0 {
		return totalConfidence / float64(count)
	}

	return 0.0
}

// calculateDuplicationLevel estimates the level of text duplication.
func (h *HybridProcessor) calculateDuplicationLevel(result *HybridResult) float64 {
	// This is a simplified implementation
	// In a full implementation, we would analyze text overlap more thoroughly
	originalLen := 0
	if result.VectorText != nil {
		originalLen += len(result.VectorText.Text)
	}
	for _, ocrResult := range result.OCRResults {
		// Use OCRRegions if available (DetectedRegions don't have text)
		if len(ocrResult.OCRRegions) > 0 {
			for _, region := range ocrResult.OCRRegions {
				originalLen += len(region.Text)
			}
		}
		// Skip DetectedRegions as they don't contain text
	}

	combinedLen := len(result.CombinedText)
	if originalLen > 0 && combinedLen > 0 {
		compressionRatio := float64(combinedLen) / float64(originalLen)
		return math.Max(0, 1.0-compressionRatio)
	}

	return 0.0
}

// GetConfig returns the current hybrid configuration.
func (h *HybridProcessor) GetConfig() *HybridConfig {
	return h.config
}

// UpdateConfig updates the hybrid configuration.
func (h *HybridProcessor) UpdateConfig(config *HybridConfig) {
	if config != nil {
		h.config = config
	}
}
