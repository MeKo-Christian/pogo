package pdf

import (
	"fmt"
	"math"
	"strconv"
)

// ProcessingStrategy represents the recommended processing approach for a PDF page.
type ProcessingStrategy int

const (
	// StrategyVectorText indicates the page should be processed using vector text extraction.
	StrategyVectorText ProcessingStrategy = iota
	// StrategyOCR indicates the page should be processed using OCR.
	StrategyOCR
	// StrategyHybrid indicates the page should be processed using both vector and OCR.
	StrategyHybrid
	// StrategySkip indicates the page should be skipped (no useful content).
	StrategySkip
)

const (
	strategyUnknown = "unknown"
)

// String returns the string representation of the processing strategy.
func (s ProcessingStrategy) String() string {
	switch s {
	case StrategyVectorText:
		return "vector_text"
	case StrategyOCR:
		return "ocr"
	case StrategyHybrid:
		return "hybrid"
	case StrategySkip:
		return "skip"
	default:
		return strategyUnknown
	}
}

// PageAnalysis contains the analysis results for a PDF page.
type PageAnalysis struct {
	PageNumber           int                `json:"page_number"`
	RecommendedStrategy  ProcessingStrategy `json:"recommended_strategy"`
	VectorTextExtraction *TextExtraction    `json:"vector_text_extraction,omitempty"`
	HasImages            bool               `json:"has_images"`
	ImageCount           int                `json:"image_count"`
	VectorTextQuality    float64            `json:"vector_text_quality"`
	VectorTextCoverage   float64            `json:"vector_text_coverage"`
	EstimatedOCRBenefit  float64            `json:"estimated_ocr_benefit"`
	AnalysisScore        float64            `json:"analysis_score"`
	Reasoning            string             `json:"reasoning"`
}

// AnalyzerConfig contains configuration for the PDF analyzer.
type AnalyzerConfig struct {
	// VectorTextQualityThreshold is the minimum quality score for vector text to be considered usable.
	VectorTextQualityThreshold float64 `json:"vector_text_quality_threshold"`

	// VectorTextCoverageThreshold is the minimum coverage for vector text to be preferred over OCR.
	VectorTextCoverageThreshold float64 `json:"vector_text_coverage_threshold"`

	// HybridModeEnabled determines if hybrid processing (vector + OCR) is allowed.
	HybridModeEnabled bool `json:"hybrid_mode_enabled"`

	// OCRFallbackEnabled determines if OCR fallback is allowed when vector text quality is poor.
	OCRFallbackEnabled bool `json:"ocr_fallback_enabled"`

	// MinTextDensityForOCR is the minimum text density to consider OCR worthwhile.
	MinTextDensityForOCR float64 `json:"min_text_density_for_ocr"`
}

// DefaultAnalyzerConfig returns the default analyzer configuration.
func DefaultAnalyzerConfig() *AnalyzerConfig {
	return &AnalyzerConfig{
		VectorTextQualityThreshold:  0.7,
		VectorTextCoverageThreshold: 0.8,
		HybridModeEnabled:           true,
		OCRFallbackEnabled:          true,
		MinTextDensityForOCR:        0.1,
	}
}

// PageAnalyzer analyzes PDF pages to determine the best processing strategy.
type PageAnalyzer struct {
	config        *AnalyzerConfig
	textExtractor *VectorTextExtractor
}

// NewPageAnalyzer creates a new page analyzer with the given configuration.
func NewPageAnalyzer(config *AnalyzerConfig) *PageAnalyzer {
	if config == nil {
		config = DefaultAnalyzerConfig()
	}

	textExtractor := NewVectorTextExtractor(config.VectorTextQualityThreshold)

	return &PageAnalyzer{
		config:        config,
		textExtractor: textExtractor,
	}
}

// AnalyzePage analyzes a single PDF page and returns the recommended processing strategy.
func (a *PageAnalyzer) AnalyzePage(filename string, pageNum int) (*PageAnalysis, error) {
	// Extract vector text from this specific page
	pageRange := strconv.Itoa(pageNum)
	extractions, err := a.textExtractor.ExtractText(filename, pageRange)
	if err != nil {
		return nil, fmt.Errorf("failed to extract text from page %d: %w", pageNum, err)
	}

	var extraction *TextExtraction
	if ext, exists := extractions[pageNum]; exists {
		extraction = ext
	}

	// Analyze images on the page (simplified for now)
	hasImages, imageCount := a.analyzeImages(filename, pageNum)

	// Determine the best processing strategy
	strategy, score, reasoning := a.determineStrategy(extraction, hasImages, imageCount)

	analysis := &PageAnalysis{
		PageNumber:           pageNum,
		RecommendedStrategy:  strategy,
		VectorTextExtraction: extraction,
		HasImages:            hasImages,
		ImageCount:           imageCount,
		AnalysisScore:        score,
		Reasoning:            reasoning,
	}

	// Set quality and coverage if we have vector text extraction
	if extraction != nil {
		analysis.VectorTextQuality = extraction.Quality.Score
		analysis.VectorTextCoverage = extraction.Coverage
		analysis.EstimatedOCRBenefit = a.estimateOCRBenefit(extraction, hasImages)
	}

	return analysis, nil
}

// AnalyzePages analyzes multiple PDF pages and returns analysis for each.
func (a *PageAnalyzer) AnalyzePages(filename string, pageRange string) (map[int]*PageAnalysis, error) {
	// Parse page range
	pageNumbers, err := parsePageRange(pageRange)
	if err != nil {
		return nil, fmt.Errorf("invalid page range %q: %w", pageRange, err)
	}

	// If no specific pages, we need to determine total pages
	// For now, we'll extract all vector text and analyze each page found
	extractions, err := a.textExtractor.ExtractText(filename, pageRange)
	if err != nil {
		return nil, fmt.Errorf("failed to extract text: %w", err)
	}

	results := make(map[int]*PageAnalysis)

	// If specific pages were requested, analyze only those
	if len(pageNumbers) > 0 {
		for _, pageNum := range pageNumbers {
			analysis, err := a.AnalyzePage(filename, pageNum)
			if err != nil {
				// Log error but continue with other pages
				continue
			}
			results[pageNum] = analysis
		}
	} else {
		// Analyze all pages found in extractions
		for pageNum := range extractions {
			analysis, err := a.AnalyzePage(filename, pageNum)
			if err != nil {
				continue
			}
			results[pageNum] = analysis
		}
	}

	return results, nil
}

// determineStrategy determines the best processing strategy based on analysis.
func (a *PageAnalyzer) determineStrategy(extraction *TextExtraction, hasImages bool,
	_ int,
) (ProcessingStrategy, float64, string) {
	if extraction == nil {
		return a.handleNoVectorText(hasImages)
	}

	quality := extraction.Quality.Score
	coverage := extraction.Coverage

	if a.isHighQualityVectorText(quality, coverage) {
		return a.handleHighQualityVectorText(hasImages)
	}

	if a.isModerateQualityVectorText(quality) {
		return a.handleModerateQualityVectorText(hasImages)
	}

	return a.handlePoorQualityVectorText(extraction, hasImages)
}

// handleNoVectorText handles the case when no vector text extraction was possible.
func (a *PageAnalyzer) handleNoVectorText(hasImages bool) (ProcessingStrategy, float64, string) {
	if hasImages {
		return StrategyOCR, 0.8, "No vector text found, but images present - recommend OCR"
	}
	return StrategySkip, 0.1, "No vector text or images found - recommend skipping"
}

// isHighQualityVectorText checks if vector text meets high quality thresholds.
func (a *PageAnalyzer) isHighQualityVectorText(quality, coverage float64) bool {
	return quality >= a.config.VectorTextQualityThreshold && coverage >= a.config.VectorTextCoverageThreshold
}

// handleHighQualityVectorText handles high-quality vector text cases.
func (a *PageAnalyzer) handleHighQualityVectorText(hasImages bool) (ProcessingStrategy, float64, string) {
	if hasImages && a.config.HybridModeEnabled {
		return StrategyHybrid, 0.9, "High-quality vector text with images - recommend hybrid processing"
	}
	return StrategyVectorText, 0.95, "High-quality vector text with good coverage - recommend vector text extraction"
}

// isModerateQualityVectorText checks if vector text has moderate quality.
func (a *PageAnalyzer) isModerateQualityVectorText(quality float64) bool {
	return quality >= 0.5 && quality < a.config.VectorTextQualityThreshold
}

// handleModerateQualityVectorText handles moderate quality vector text cases.
func (a *PageAnalyzer) handleModerateQualityVectorText(hasImages bool) (ProcessingStrategy, float64, string) {
	if hasImages && a.config.HybridModeEnabled {
		return StrategyHybrid, 0.7, "Moderate vector text quality with images - recommend hybrid processing"
	}
	if a.config.OCRFallbackEnabled && hasImages {
		return StrategyOCR, 0.6, "Moderate vector text quality, OCR may provide better results"
	}
	return StrategyVectorText, 0.5, "Moderate vector text quality - use vector text extraction"
}

// handlePoorQualityVectorText handles poor quality vector text cases.
func (a *PageAnalyzer) handlePoorQualityVectorText(extraction *TextExtraction,
	hasImages bool,
) (ProcessingStrategy, float64, string) {
	if hasImages && a.config.OCRFallbackEnabled {
		return StrategyOCR, 0.7, "Poor vector text quality but images present - recommend OCR"
	}

	if extraction.Quality.HasText {
		return StrategyVectorText, 0.3, "Poor quality vector text but no images - use vector text extraction"
	}

	return StrategySkip, 0.1, "No useful content found - recommend skipping"
}

// analyzeImages analyzes the presence of images on a PDF page.
// This is a simplified implementation - a full version would parse the PDF content stream.
func (a *PageAnalyzer) analyzeImages(filename string, pageNum int) (bool, int) {
	// For now, we'll use the existing ExtractImages function to check for images
	pageRange := strconv.Itoa(pageNum)
	images, err := ExtractImages(filename, pageRange)
	if err != nil {
		return false, 0
	}

	if pageImages, exists := images[pageNum]; exists {
		return len(pageImages) > 0, len(pageImages)
	}

	return false, 0
}

// estimateOCRBenefit estimates how much OCR might benefit text extraction.
func (a *PageAnalyzer) estimateOCRBenefit(extraction *TextExtraction, hasImages bool) float64 {
	if !hasImages {
		return 0.0 // No images means no OCR benefit
	}

	if extraction == nil {
		return 1.0 // No vector text means maximum OCR benefit
	}

	// Calculate potential benefit based on text quality and coverage
	qualityGap := 1.0 - extraction.Quality.Score
	coverageGap := 1.0 - extraction.Coverage

	// Estimate benefit as the average of quality and coverage gaps
	benefit := (qualityGap + coverageGap) / 2.0

	// Apply a minimum benefit if images are present
	minBenefit := 0.1
	return math.Max(benefit, minBenefit)
}

// GetConfig returns the current analyzer configuration.
func (a *PageAnalyzer) GetConfig() *AnalyzerConfig {
	return a.config
}

// UpdateConfig updates the analyzer configuration.
func (a *PageAnalyzer) UpdateConfig(config *AnalyzerConfig) {
	if config != nil {
		a.config = config
		// Update text extractor threshold
		a.textExtractor.SetQualityThreshold(config.VectorTextQualityThreshold)
	}
}
