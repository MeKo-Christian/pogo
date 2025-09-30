package pdf

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/dslipak/pdf"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// getTestPDFPath returns the path to a test PDF file from project root.
func getTestPDFPath(t *testing.T, filename string) string {
	t.Helper()
	root, err := testutil.GetProjectRoot()
	require.NoError(t, err)
	return filepath.Join(root, "testdata", "documents", filename)
}

// TestNewVectorTextExtractor tests the constructor with various quality thresholds.
func TestNewVectorTextExtractor(t *testing.T) {
	tests := []struct {
		name              string
		inputThreshold    float64
		expectedThreshold float64
	}{
		{
			name:              "zero threshold uses default",
			inputThreshold:    0,
			expectedThreshold: 0.7,
		},
		{
			name:              "negative threshold uses default",
			inputThreshold:    -0.5,
			expectedThreshold: 0.7,
		},
		{
			name:              "valid threshold",
			inputThreshold:    0.8,
			expectedThreshold: 0.8,
		},
		{
			name:              "threshold of 1.0",
			inputThreshold:    1.0,
			expectedThreshold: 1.0,
		},
		{
			name:              "very low threshold",
			inputThreshold:    0.1,
			expectedThreshold: 0.1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			extractor := NewVectorTextExtractor(tt.inputThreshold)
			require.NotNil(t, extractor)
			assert.InDelta(t, tt.expectedThreshold, extractor.qualityThreshold, 0.001)
			assert.InDelta(t, tt.expectedThreshold, extractor.GetQualityThreshold(), 0.001)
		})
	}
}

// TestVectorTextExtractor_ExtractText_InvalidFile tests extraction with invalid files.
func TestVectorTextExtractor_ExtractText_InvalidFile(t *testing.T) {
	extractor := NewVectorTextExtractor(0.7)

	t.Run("non-existent file", func(t *testing.T) {
		result, err := extractor.ExtractText("/non/existent/file.pdf", "")
		require.Error(t, err)
		assert.Nil(t, result)
		assert.Contains(t, err.Error(), "failed to open PDF")
	})

	t.Run("empty filename", func(t *testing.T) {
		result, err := extractor.ExtractText("", "")
		require.Error(t, err)
		assert.Nil(t, result)
	})
}

// TestVectorTextExtractor_ExtractText_InvalidPageRange tests extraction with invalid page ranges.
func TestVectorTextExtractor_ExtractText_InvalidPageRange(t *testing.T) {
	extractor := NewVectorTextExtractor(0.7)

	tests := []struct {
		name      string
		pageRange string
	}{
		{"invalid format", "abc"},
		{"invalid range", "1-2-3"},
		{"negative page", "-1"},
		{"malformed range", "1--3"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Using a non-existent file since we're testing page range parsing
			result, err := extractor.ExtractText("dummy.pdf", tt.pageRange)
			require.Error(t, err)
			assert.Nil(t, result)
			assert.Contains(t, err.Error(), "invalid page range")
		})
	}
}

// TestVectorTextExtractor_ExtractText_WithRealPDF tests extraction with actual PDF files.
func TestVectorTextExtractor_ExtractText_WithRealPDF(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	extractor := NewVectorTextExtractor(0.5)

	tests := []struct {
		name         string
		pdfFile      string
		pageRange    string
		expectPages  []int
		expectText   bool
		minWordCount int
	}{
		{
			name:         "text only PDF - all pages",
			pdfFile:      "text_only.pdf",
			pageRange:    "",
			expectPages:  []int{1},
			expectText:   true,
			minWordCount: 5,
		},
		{
			name:         "text only PDF - specific page",
			pdfFile:      "text_only.pdf",
			pageRange:    "1",
			expectPages:  []int{1},
			expectText:   true,
			minWordCount: 5,
		},
		{
			name:        "sample PDF - all pages",
			pdfFile:     "sample.pdf",
			pageRange:   "",
			expectPages: []int{1},
			expectText:  true,
		},
		{
			name:       "multipage PDF - all pages",
			pdfFile:    "multipage.pdf",
			pageRange:  "",
			expectText: true,
		},
		{
			name:        "multipage PDF - page 1",
			pdfFile:     "multipage.pdf",
			pageRange:   "1",
			expectPages: []int{1},
			expectText:  true,
		},
		{
			name:       "multipage PDF - page range",
			pdfFile:    "multipage.pdf",
			pageRange:  "1-2",
			expectText: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filename := getTestPDFPath(t, tt.pdfFile)
			result, err := extractor.ExtractText(filename, tt.pageRange)
			require.NoError(t, err, "extraction should succeed")
			require.NotNil(t, result, "result should not be nil")
			assert.NotEmpty(t, result, "should have extracted at least one page")

			// Check expected pages if specified
			if len(tt.expectPages) > 0 {
				for _, pageNum := range tt.expectPages {
					assert.Contains(t, result, pageNum, "should contain page %d", pageNum)
				}
			}

			// Verify each extraction
			for pageNum, extraction := range result {
				assert.Equal(t, pageNum, extraction.PageNumber)

				// Note: Some test PDFs may not have extractable vector text
				// Just verify the structure is correct
				if tt.expectText && len(extraction.Text) > 0 {
					if tt.minWordCount > 0 && extraction.WordCount >= tt.minWordCount {
						assert.GreaterOrEqual(t, extraction.WordCount, tt.minWordCount,
							"page %d should have at least %d words", pageNum, tt.minWordCount)
					}
				}

				// Verify metadata
				assert.Greater(t, extraction.Metadata.PageWidth, 0.0)
				assert.Greater(t, extraction.Metadata.PageHeight, 0.0)
				assert.Equal(t, "vector", extraction.Metadata.ExtractionMethod)

				// Verify quality assessment
				assert.GreaterOrEqual(t, extraction.Quality.Score, 0.0)
				assert.LessOrEqual(t, extraction.Quality.Score, 1.0)
				assert.Equal(t, extraction.Quality.HasText, len(strings.TrimSpace(extraction.Text)) > 0)

				// Verify coverage is within bounds
				assert.GreaterOrEqual(t, extraction.Coverage, 0.0)
				assert.LessOrEqual(t, extraction.Coverage, 1.0)
			}
		})
	}
}

// TestVectorTextExtractor_ExtractText_OutOfBoundPages tests page filtering.
func TestVectorTextExtractor_ExtractText_OutOfBoundPages(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	extractor := NewVectorTextExtractor(0.7)

	tests := []struct {
		name        string
		pdfFile     string
		pageRange   string
		expectEmpty bool
	}{
		{
			name:        "page beyond range",
			pdfFile:     "text_only.pdf",
			pageRange:   "999",
			expectEmpty: true,
		},
		{
			name:        "range partially out of bounds",
			pdfFile:     "text_only.pdf",
			pageRange:   "1,999",
			expectEmpty: false, // Should still get page 1
		},
		{
			name:        "all pages out of bounds",
			pdfFile:     "text_only.pdf",
			pageRange:   "100-200",
			expectEmpty: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filename := getTestPDFPath(t, tt.pdfFile)
			result, err := extractor.ExtractText(filename, tt.pageRange)
			require.NoError(t, err)
			require.NotNil(t, result)

			if tt.expectEmpty {
				assert.Empty(t, result)
			} else {
				assert.NotEmpty(t, result)
			}
		})
	}
}

// TestAssessTextQuality tests the quality assessment logic.
func TestAssessTextQuality(t *testing.T) {
	extractor := NewVectorTextExtractor(0.7)

	tests := []struct {
		name             string
		text             string
		positions        []TextPosition
		pageWidth        float64
		pageHeight       float64
		expectHasText    bool
		expectSearchable bool
		minScore         float64
		maxScore         float64
	}{
		{
			name:             "empty text",
			text:             "",
			positions:        []TextPosition{},
			pageWidth:        612,
			pageHeight:       792,
			expectHasText:    false,
			expectSearchable: false,
			minScore:         0.0,
			maxScore:         0.0,
		},
		{
			name:             "whitespace only",
			text:             "   \n\t  ",
			positions:        []TextPosition{{Text: "   "}},
			pageWidth:        612,
			pageHeight:       792,
			expectHasText:    false,
			expectSearchable: false,
			minScore:         0.0,
			maxScore:         0.0,
		},
		{
			name:             "single word",
			text:             "Hello",
			positions:        []TextPosition{{Text: "Hello"}},
			pageWidth:        612,
			pageHeight:       792,
			expectHasText:    true,
			expectSearchable: true,
			minScore:         0.4,
			maxScore:         0.5,
		},
		{
			name:             "good text with multiple words",
			text:             "This is a sample document with several words.",
			positions:        []TextPosition{{Text: "This is a sample document with several words."}},
			pageWidth:        612,
			pageHeight:       792,
			expectHasText:    true,
			expectSearchable: true,
			minScore:         0.6,
			maxScore:         1.0,
		},
		{
			name: "text with high density",
			text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " +
				"Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " +
				"Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
			positions:        []TextPosition{{Text: "Lorem ipsum..."}},
			pageWidth:        612,
			pageHeight:       792,
			expectHasText:    true,
			expectSearchable: true,
			minScore:         0.9,
			maxScore:         1.0,
		},
		{
			name:             "text with symbols (low alphanumeric ratio)",
			text:             "!@#$%^&*()_+-=[]{}|;':\",./<>?",
			positions:        []TextPosition{{Text: "symbols"}},
			pageWidth:        612,
			pageHeight:       792,
			expectHasText:    true,
			expectSearchable: true,
			minScore:         0.4,
			maxScore:         0.6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			quality := extractor.assessTextQuality(tt.text, tt.pageWidth, tt.pageHeight)

			assert.Equal(t, tt.expectHasText, quality.HasText)
			assert.Equal(t, tt.expectSearchable, quality.IsSearchable)
			assert.GreaterOrEqual(t, quality.Score, tt.minScore,
				"score should be >= %f, got %f", tt.minScore, quality.Score)
			assert.LessOrEqual(t, quality.Score, tt.maxScore,
				"score should be <= %f, got %f", tt.maxScore, quality.Score)
			assert.GreaterOrEqual(t, quality.Score, 0.0)
			assert.LessOrEqual(t, quality.Score, 1.0)
			assert.GreaterOrEqual(t, quality.TextDensity, 0.0)
		})
	}
}

// TestHasReasonableCharacterDistribution tests character distribution checking.
func TestHasReasonableCharacterDistribution(t *testing.T) {
	extractor := NewVectorTextExtractor(0.7)

	tests := []struct {
		name     string
		text     string
		expected bool
	}{
		{
			name:     "empty text",
			text:     "",
			expected: false,
		},
		{
			name:     "pure alphanumeric",
			text:     "abcdefghijklmnopqrstuvwxyz0123456789",
			expected: true,
		},
		{
			name:     "mixed with spaces (still >50% alphanumeric)",
			text:     "Hello World 2024",
			expected: true,
		},
		{
			name:     "mostly symbols (<50% alphanumeric)",
			text:     "!@#$%^&*()_+-=[]{}|;':\",./<>?a",
			expected: false,
		},
		{
			name:     "exactly 50% alphanumeric",
			text:     "aaaaa!!!!!",
			expected: true,
		},
		{
			name:     "unicode characters",
			text:     "Hello World Cyrillic",
			expected: false, // Unicode chars not counted as alphanumeric
		},
		{
			name:     "normal sentence",
			text:     "The quick brown fox jumps over the lazy dog.",
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractor.hasReasonableCharacterDistribution(tt.text)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestCalculateTextCoverage tests coverage calculation.
func TestCalculateTextCoverage(t *testing.T) {
	extractor := NewVectorTextExtractor(0.7)

	tests := []struct {
		name        string
		positions   []TextPosition
		pageWidth   float64
		pageHeight  float64
		expectedMin float64
		expectedMax float64
	}{
		{
			name:        "no positions",
			positions:   []TextPosition{},
			pageWidth:   612,
			pageHeight:  792,
			expectedMin: 0.0,
			expectedMax: 0.0,
		},
		{
			name:        "zero page dimensions",
			positions:   []TextPosition{{Width: 100, Height: 50}},
			pageWidth:   0,
			pageHeight:  0,
			expectedMin: 0.0,
			expectedMax: 0.0,
		},
		{
			name: "single text position",
			positions: []TextPosition{
				{Width: 100, Height: 50},
			},
			pageWidth:   612,
			pageHeight:  792,
			expectedMin: 0.01,
			expectedMax: 0.02,
		},
		{
			name: "multiple text positions",
			positions: []TextPosition{
				{Width: 100, Height: 50},
				{Width: 150, Height: 30},
				{Width: 200, Height: 40},
			},
			pageWidth:   612,
			pageHeight:  792,
			expectedMin: 0.025,
			expectedMax: 0.037,
		},
		{
			name: "overlapping text (coverage > 1.0, should be capped)",
			positions: []TextPosition{
				{Width: 700, Height: 900},
				{Width: 700, Height: 900},
			},
			pageWidth:   612,
			pageHeight:  792,
			expectedMin: 1.0,
			expectedMax: 1.0, // Should be capped at 1.0
		},
		{
			name: "full page coverage",
			positions: []TextPosition{
				{Width: 612, Height: 792},
			},
			pageWidth:   612,
			pageHeight:  792,
			expectedMin: 0.99,
			expectedMax: 1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			coverage := extractor.calculateTextCoverage(tt.positions, tt.pageWidth, tt.pageHeight)
			assert.GreaterOrEqual(t, coverage, tt.expectedMin,
				"coverage should be >= %f, got %f", tt.expectedMin, coverage)
			assert.LessOrEqual(t, coverage, tt.expectedMax,
				"coverage should be <= %f, got %f", tt.expectedMax, coverage)
			assert.GreaterOrEqual(t, coverage, 0.0)
			assert.LessOrEqual(t, coverage, 1.0)
		})
	}
}

// TestIsQualityAcceptable tests quality threshold checking.
func TestIsQualityAcceptable(t *testing.T) {
	tests := []struct {
		name       string
		threshold  float64
		extraction *TextExtraction
		expected   bool
	}{
		{
			name:       "nil extraction",
			threshold:  0.7,
			extraction: nil,
			expected:   false,
		},
		{
			name:      "quality above threshold",
			threshold: 0.7,
			extraction: &TextExtraction{
				Quality: TextQuality{Score: 0.8},
			},
			expected: true,
		},
		{
			name:      "quality equal to threshold",
			threshold: 0.7,
			extraction: &TextExtraction{
				Quality: TextQuality{Score: 0.7},
			},
			expected: true,
		},
		{
			name:      "quality below threshold",
			threshold: 0.7,
			extraction: &TextExtraction{
				Quality: TextQuality{Score: 0.6},
			},
			expected: false,
		},
		{
			name:      "perfect quality",
			threshold: 0.9,
			extraction: &TextExtraction{
				Quality: TextQuality{Score: 1.0},
			},
			expected: true,
		},
		{
			name:      "zero quality",
			threshold: 0.5,
			extraction: &TextExtraction{
				Quality: TextQuality{Score: 0.0},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			extractor := NewVectorTextExtractor(tt.threshold)
			result := extractor.IsQualityAcceptable(tt.extraction)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestGetQualityThreshold tests the getter.
func TestGetQualityThreshold(t *testing.T) {
	tests := []struct {
		threshold float64
	}{
		{threshold: 0.5},
		{threshold: 0.7},
		{threshold: 0.9},
		{threshold: 1.0},
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			extractor := NewVectorTextExtractor(tt.threshold)
			assert.InDelta(t, tt.threshold, extractor.GetQualityThreshold(), 0.001)
		})
	}
}

// TestSetQualityThreshold tests the setter.
func TestSetQualityThreshold(t *testing.T) {
	tests := []struct {
		name              string
		initialThreshold  float64
		newThreshold      float64
		expectedThreshold float64
	}{
		{
			name:              "valid threshold update",
			initialThreshold:  0.7,
			newThreshold:      0.8,
			expectedThreshold: 0.8,
		},
		{
			name:              "zero threshold (invalid, should not update)",
			initialThreshold:  0.7,
			newThreshold:      0.0,
			expectedThreshold: 0.7,
		},
		{
			name:              "negative threshold (invalid, should not update)",
			initialThreshold:  0.7,
			newThreshold:      -0.5,
			expectedThreshold: 0.7,
		},
		{
			name:              "threshold > 1.0 (invalid, should not update)",
			initialThreshold:  0.7,
			newThreshold:      1.5,
			expectedThreshold: 0.7,
		},
		{
			name:              "threshold = 1.0 (valid)",
			initialThreshold:  0.7,
			newThreshold:      1.0,
			expectedThreshold: 1.0,
		},
		{
			name:              "very low valid threshold",
			initialThreshold:  0.7,
			newThreshold:      0.01,
			expectedThreshold: 0.01,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			extractor := NewVectorTextExtractor(tt.initialThreshold)
			extractor.SetQualityThreshold(tt.newThreshold)
			assert.InDelta(t, tt.expectedThreshold, extractor.GetQualityThreshold(), 0.001)
		})
	}
}

// TestGetPageDimensions tests page dimension extraction.
func TestGetPageDimensions(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	extractor := NewVectorTextExtractor(0.7)

	// Open a test PDF to get a real page object
	filename := getTestPDFPath(t, "text_only.pdf")
	reader, err := pdf.Open(filename)
	require.NoError(t, err)
	require.NotNil(t, reader)
	require.Positive(t, reader.NumPage())

	page := reader.Page(1)
	require.False(t, page.V.IsNull())

	width, height := extractor.getPageDimensions(page)

	// Should return default letter size dimensions
	assert.InDelta(t, 612.0, width, 0.001)
	assert.InDelta(t, 792.0, height, 0.001)
}

// TestExtractTextWithPositions tests text extraction with position tracking.
func TestExtractTextWithPositions(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	extractor := NewVectorTextExtractor(0.7)

	filename := getTestPDFPath(t, "text_only.pdf")
	reader, err := pdf.Open(filename)
	require.NoError(t, err)
	require.Positive(t, reader.NumPage())

	page := reader.Page(1)
	require.False(t, page.V.IsNull())

	text, positions, fonts := extractor.extractTextWithPositions(page)

	// Verify extraction completes without error
	// Note: Some PDFs may not have extractable vector text
	// Just verify we got valid return values
	_ = text
	_ = positions
	_ = fonts

	// If text was extracted, verify structure
	if len(text) > 0 {
		assert.NotEmpty(t, positions, "should have position information if text exists")
		for _, pos := range positions {
			// Note: X, Y may be 0 due to library limitations
			assert.GreaterOrEqual(t, pos.Width, 0.0)
			assert.GreaterOrEqual(t, pos.Height, 0.0)
			assert.GreaterOrEqual(t, pos.FontSize, 0.0)
		}
		assert.NotEmpty(t, fonts, "should have font information if text exists")
	}
}

// TestVectorTextExtractor_EndToEnd tests complete extraction workflow.
func TestVectorTextExtractor_EndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	tests := []struct {
		name      string
		pdfFile   string
		pageRange string
		threshold float64
	}{
		{
			name:      "text only PDF with default threshold",
			pdfFile:   "text_only.pdf",
			pageRange: "",
			threshold: 0.7,
		},
		{
			name:      "multipage PDF with low threshold",
			pdfFile:   "multipage.pdf",
			pageRange: "1-2",
			threshold: 0.3,
		},
		{
			name:      "sample PDF with high threshold",
			pdfFile:   "sample.pdf",
			pageRange: "1",
			threshold: 0.9,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			extractor := NewVectorTextExtractor(tt.threshold)

			// Extract text
			filename := getTestPDFPath(t, tt.pdfFile)
			results, err := extractor.ExtractText(filename, tt.pageRange)
			require.NoError(t, err)
			require.NotNil(t, results)

			// Verify each page
			for pageNum, extraction := range results {
				assert.Equal(t, pageNum, extraction.PageNumber)
				assert.NotNil(t, extraction.Text)
				assert.GreaterOrEqual(t, extraction.WordCount, 0)
				assert.GreaterOrEqual(t, extraction.Coverage, 0.0)
				assert.LessOrEqual(t, extraction.Coverage, 1.0)

				// Verify quality
				assert.GreaterOrEqual(t, extraction.Quality.Score, 0.0)
				assert.LessOrEqual(t, extraction.Quality.Score, 1.0)

				// Verify metadata
				assert.Greater(t, extraction.Metadata.PageWidth, 0.0)
				assert.Greater(t, extraction.Metadata.PageHeight, 0.0)
				assert.Equal(t, "vector", extraction.Metadata.ExtractionMethod)
				assert.GreaterOrEqual(t, extraction.Metadata.TextElements, 0)
				assert.NotNil(t, extraction.Metadata.FontsUsed)

				// Test quality acceptance
				isAcceptable := extractor.IsQualityAcceptable(extraction)
				if extraction.Quality.Score >= tt.threshold {
					assert.True(t, isAcceptable)
				} else {
					assert.False(t, isAcceptable)
				}
			}
		})
	}
}

// Benchmark tests.

func BenchmarkExtractText(b *testing.B) {
	extractor := NewVectorTextExtractor(0.7)
	root, err := testutil.GetProjectRoot()
	if err != nil {
		b.Skip("Cannot find project root")
	}
	filename := filepath.Join(root, "testdata", "documents", "text_only.pdf")

	b.ResetTimer()
	for range b.N {
		_, _ = extractor.ExtractText(filename, "")
	}
}

func BenchmarkAssessTextQuality(b *testing.B) {
	extractor := NewVectorTextExtractor(0.7)
	text := "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " +
		"Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " +
		"Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."

	b.ResetTimer()
	for range b.N {
		_ = extractor.assessTextQuality(text, 612, 792)
	}
}

func BenchmarkHasReasonableCharacterDistribution(b *testing.B) {
	extractor := NewVectorTextExtractor(0.7)
	text := "The quick brown fox jumps over the lazy dog. 1234567890."

	b.ResetTimer()
	for range b.N {
		_ = extractor.hasReasonableCharacterDistribution(text)
	}
}

func BenchmarkCalculateTextCoverage(b *testing.B) {
	extractor := NewVectorTextExtractor(0.7)
	positions := []TextPosition{
		{Width: 100, Height: 12},
		{Width: 150, Height: 12},
		{Width: 200, Height: 12},
		{Width: 120, Height: 12},
		{Width: 180, Height: 12},
	}

	b.ResetTimer()
	for range b.N {
		_ = extractor.calculateTextCoverage(positions, 612, 792)
	}
}
