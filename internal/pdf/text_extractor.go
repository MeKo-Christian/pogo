package pdf

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/dslipak/pdf"
)

// TextExtraction represents the result of vector text extraction from a PDF page.
type TextExtraction struct {
	PageNumber int                `json:"page_number"`
	Text       string             `json:"text"`
	WordCount  int                `json:"word_count"`
	Coverage   float64            `json:"coverage"`  // Percentage of page covered by text (0-1)
	Quality    TextQuality        `json:"quality"`   // Overall quality assessment
	Positions  []TextPosition     `json:"positions"` // Text positions for layout preservation
	Metadata   ExtractionMetadata `json:"metadata"`  // Additional extraction info
}

// TextPosition represents the position and content of text on a page.
type TextPosition struct {
	Text     string  `json:"text"`
	X        float64 `json:"x"`
	Y        float64 `json:"y"`
	Width    float64 `json:"width"`
	Height   float64 `json:"height"`
	FontSize float64 `json:"font_size"`
}

// TextQuality represents the quality assessment of extracted text.
type TextQuality struct {
	Score        float64 `json:"score"`         // Overall quality score (0-1)
	HasText      bool    `json:"has_text"`      // Whether any text was found
	IsSearchable bool    `json:"is_searchable"` // Whether PDF has searchable text
	TextDensity  float64 `json:"text_density"`  // Characters per page area
}

// ExtractionMetadata contains additional information about the extraction process.
type ExtractionMetadata struct {
	PageWidth        float64  `json:"page_width"`
	PageHeight       float64  `json:"page_height"`
	TextElements     int      `json:"text_elements"`
	FontsUsed        []string `json:"fonts_used"`
	ExtractionMethod string   `json:"extraction_method"`
}

// VectorTextExtractor handles extraction of vector text from PDFs.
type VectorTextExtractor struct {
	qualityThreshold float64 // Minimum quality score to consider text usable
}

// NewVectorTextExtractor creates a new vector text extractor.
func NewVectorTextExtractor(qualityThreshold float64) *VectorTextExtractor {
	if qualityThreshold <= 0 {
		qualityThreshold = 0.7 // Default threshold
	}
	return &VectorTextExtractor{
		qualityThreshold: qualityThreshold,
	}
}

// ExtractText extracts vector text from a PDF file for specified pages.
func (e *VectorTextExtractor) ExtractText(filename string, pageRange string) (map[int]*TextExtraction, error) {
	// Parse page range
	pageNumbers, err := parsePageRange(pageRange)
	if err != nil {
		return nil, fmt.Errorf("invalid page range %q: %w", pageRange, err)
	}

	// Open PDF file
	pdfReader, err := pdf.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open PDF %q: %w", filename, err)
	}
	// Note: dslipak/pdf Reader doesn't need explicit closing

	totalPages := pdfReader.NumPage()

	// Filter valid pages or use all pages if none specified
	var pagesToProcess []int
	if len(pageNumbers) == 0 {
		// Process all pages
		for i := 1; i <= totalPages; i++ {
			pagesToProcess = append(pagesToProcess, i)
		}
	} else {
		// Filter to valid page numbers
		for _, pageNum := range pageNumbers {
			if pageNum >= 1 && pageNum <= totalPages {
				pagesToProcess = append(pagesToProcess, pageNum)
			}
		}
	}

	results := make(map[int]*TextExtraction)

	// Extract text from each page
	for _, pageNum := range pagesToProcess {
		extraction, err := e.extractPageText(pdfReader, pageNum)
		if err != nil {
			// Log error but continue with other pages
			continue
		}
		results[pageNum] = extraction
	}

	return results, nil
}

// extractPageText extracts text from a single PDF page.
func (e *VectorTextExtractor) extractPageText(reader *pdf.Reader, pageNum int) (*TextExtraction, error) {
	page := reader.Page(pageNum)
	if page.V.IsNull() {
		return nil, fmt.Errorf("page %d is null", pageNum)
	}

	// Get page dimensions
	pageWidth, pageHeight := e.getPageDimensions(page)

	// Extract text content and positions
	textContent, positions, fonts := e.extractTextWithPositions(page)

	// Calculate text quality metrics
	quality := e.assessTextQuality(textContent, pageWidth, pageHeight)

	// Calculate coverage
	coverage := e.calculateTextCoverage(positions, pageWidth, pageHeight)

	extraction := &TextExtraction{
		PageNumber: pageNum,
		Text:       textContent,
		WordCount:  len(strings.Fields(textContent)),
		Coverage:   coverage,
		Quality:    quality,
		Positions:  positions,
		Metadata: ExtractionMetadata{
			PageWidth:        pageWidth,
			PageHeight:       pageHeight,
			TextElements:     len(positions),
			FontsUsed:        fonts,
			ExtractionMethod: "vector",
		},
	}

	return extraction, nil
}

// extractTextWithPositions extracts text content along with position information.
func (e *VectorTextExtractor) extractTextWithPositions(page pdf.Page) (string, []TextPosition, []string) {
	var allText strings.Builder
	var positions []TextPosition
	fontMap := make(map[string]bool)

	// Try to get text with row information first
	rows, err := page.GetTextByRow()
	if err == nil && len(rows) > 0 {
		for _, row := range rows {
			for _, text := range row.Content {
				allText.WriteString(text.S)
				allText.WriteString(" ")

				// Create position information from the text element
				// Note: dslipak/pdf doesn't provide detailed positioning info
				positions = append(positions, TextPosition{
					Text:     text.S,
					X:        0,   // Position info not available in this version
					Y:        0,   // Position info not available in this version
					Width:    100, // Placeholder
					Height:   12,  // Placeholder
					FontSize: 12,  // Placeholder
				})

				// Track font if available (simplified for this implementation)
				fontMap["default"] = true
			}
			allText.WriteString("\n") // Add newline at end of row
		}
	} else {
		// Fallback: try to get plain text
		fonts := make(map[string]*pdf.Font)
		plainText, _ := page.GetPlainText(fonts)
		allText.WriteString(plainText)

		// Create a basic position entry for the extracted text
		if len(plainText) > 0 {
			positions = append(positions, TextPosition{
				Text:     plainText,
				X:        0,
				Y:        0,
				Width:    100, // Placeholder values
				Height:   12,
				FontSize: 12,
			})
			fontMap["default"] = true
		}
	}

	// Convert font map to slice
	fonts := make([]string, 0, len(fontMap))
	for font := range fontMap {
		fonts = append(fonts, font)
	}
	sort.Strings(fonts)

	return allText.String(), positions, fonts
}

// getPageDimensions extracts the page dimensions from a PDF page.
func (e *VectorTextExtractor) getPageDimensions(page pdf.Page) (float64, float64) {
	// Default dimensions if we can't extract them
	width, height := 612.0, 792.0 // Standard letter size in points

	// Try to get actual dimensions from the page
	// The pdf package doesn't expose MediaBox directly in this version
	// This is a simplified implementation
	// In a full implementation, we would parse the page dictionary
	// to get the actual MediaBox or CropBox values

	return width, height
}

// assessTextQuality evaluates the quality of extracted text.
func (e *VectorTextExtractor) assessTextQuality(text string, pageWidth, pageHeight float64) TextQuality {
	hasText := len(strings.TrimSpace(text)) > 0
	wordCount := len(strings.Fields(text))
	charCount := len(text)

	pageArea := pageWidth * pageHeight
	textDensity := 0.0
	if pageArea > 0 {
		textDensity = float64(charCount) / pageArea * 1000 // Characters per 1000 square points
	}

	// Calculate quality score based on multiple factors
	score := 0.0

	if hasText {
		score += 0.4 // Base score for having text

		// Add score based on text density
		if textDensity > 0.1 {
			score += 0.3
		}

		// Add score based on word count
		if wordCount > 5 {
			score += 0.2
		}

		// Add score based on reasonable character distribution
		if e.hasReasonableCharacterDistribution(text) {
			score += 0.1
		}
	}

	// Cap score at 1.0
	if score > 1.0 {
		score = 1.0
	}

	return TextQuality{
		Score:        score,
		HasText:      hasText,
		IsSearchable: hasText && wordCount > 0,
		TextDensity:  textDensity,
	}
}

// hasReasonableCharacterDistribution checks if the text has a reasonable character distribution.
func (e *VectorTextExtractor) hasReasonableCharacterDistribution(text string) bool {
	if len(text) == 0 {
		return false
	}

	// Count alphanumeric characters
	alphanumericCount := 0
	for _, r := range text {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			alphanumericCount++
		}
	}

	// At least 50% should be alphanumeric for reasonable text
	ratio := float64(alphanumericCount) / float64(len(text))
	return ratio >= 0.5
}

// calculateTextCoverage estimates how much of the page is covered by text.
func (e *VectorTextExtractor) calculateTextCoverage(positions []TextPosition, pageWidth, pageHeight float64) float64 {
	if len(positions) == 0 || pageWidth <= 0 || pageHeight <= 0 {
		return 0.0
	}

	totalTextArea := 0.0
	for _, pos := range positions {
		textArea := pos.Width * pos.Height
		totalTextArea += textArea
	}

	pageArea := pageWidth * pageHeight
	coverage := totalTextArea / pageArea

	// Cap coverage at 1.0 to handle overlapping text
	return math.Min(coverage, 1.0)
}

// IsQualityAcceptable checks if the extracted text quality meets the threshold.
func (e *VectorTextExtractor) IsQualityAcceptable(extraction *TextExtraction) bool {
	return extraction != nil && extraction.Quality.Score >= e.qualityThreshold
}

// GetQualityThreshold returns the current quality threshold.
func (e *VectorTextExtractor) GetQualityThreshold() float64 {
	return e.qualityThreshold
}

// SetQualityThreshold updates the quality threshold.
func (e *VectorTextExtractor) SetQualityThreshold(threshold float64) {
	if threshold > 0 && threshold <= 1.0 {
		e.qualityThreshold = threshold
	}
}
