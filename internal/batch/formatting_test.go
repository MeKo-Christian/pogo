package batch

import (
	"strings"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFormatBatchResults_Text(t *testing.T) {
	results := []*pipeline.OCRImageResult{
		createMockOCRResult("Hello World", 0.95),
		createMockOCRResult("Test Image", 0.88),
	}
	imagePaths := []string{"/path/image1.png", "/path/image2.png"}

	output, err := formatBatchResults(results, imagePaths, "text")
	require.NoError(t, err)

	assert.Contains(t, output, "# /path/image1.png")
	assert.Contains(t, output, "# /path/image2.png")
	assert.Contains(t, output, "Hello World")
	assert.Contains(t, output, "Test Image")
}

func TestFormatBatchResults_JSON(t *testing.T) {
	results := []*pipeline.OCRImageResult{
		createMockOCRResult("JSON Test", 0.90),
	}
	imagePaths := []string{"/path/test.png"}

	output, err := formatBatchResults(results, imagePaths, "json")
	require.NoError(t, err)

	assert.Contains(t, output, `"file": "/path/test.png"`)
	assert.Contains(t, output, `"text": "JSON Test"`)
	assert.Contains(t, output, `"rec_confidence": 0.9`)

	// Should be valid JSON (basic check)
	assert.Contains(t, output, "{")
	assert.Contains(t, output, "}")
}

func TestFormatBatchResults_CSV(t *testing.T) {
	results := []*pipeline.OCRImageResult{
		createMockOCRResult("CSV Test", 0.85),
	}
	imagePaths := []string{"/path/test.png"}

	output, err := formatBatchResults(results, imagePaths, "csv")
	require.NoError(t, err)

	lines := strings.Split(strings.TrimSpace(output), "\n")
	assert.Len(t, lines, 2) // Header + 1 data row

	// Check header
	assert.Contains(t, lines[0], "file")
	assert.Contains(t, lines[0], "text")
	assert.Contains(t, lines[0], "confidence")

	// Check data row
	assert.Contains(t, lines[1], "/path/test.png")
	assert.Contains(t, lines[1], "CSV Test")
	assert.Contains(t, lines[1], "0.850")
}

func TestFormatBatchResults_InvalidFormat(t *testing.T) {
	results := []*pipeline.OCRImageResult{}
	imagePaths := []string{}

	output, err := formatBatchResults(results, imagePaths, "invalid")
	require.NoError(t, err)
	assert.Empty(t, output) // Invalid format defaults to text, empty results produce empty text
}

func TestFormatBatchResults_EmptyResults(t *testing.T) {
	results := []*pipeline.OCRImageResult{}
	imagePaths := []string{}

	output, err := formatBatchResults(results, imagePaths, "text")
	require.NoError(t, err)
	assert.Empty(t, output)
}

func TestFormatJSON_SingleResult(t *testing.T) {
	results := []*pipeline.OCRImageResult{
		createMockOCRResult("Single Test", 0.92),
	}
	imagePaths := []string{"/path/single.png"}

	output, err := formatJSON(results, imagePaths)
	require.NoError(t, err)

	assert.Contains(t, output, `"file": "/path/single.png"`)
	assert.Contains(t, output, `"text": "Single Test"`)
	assert.Contains(t, output, `"rec_confidence": 0.92`)
	assert.Contains(t, output, `"det_confidence": 0.92`)
}

func TestFormatJSON_MultipleResults(t *testing.T) {
	results := []*pipeline.OCRImageResult{
		createMockOCRResult("First", 0.90),
		createMockOCRResult("Second", 0.85),
	}
	imagePaths := []string{"/path/first.png", "/path/second.png"}

	output, err := formatJSON(results, imagePaths)
	require.NoError(t, err)

	assert.Contains(t, output, `"file": "/path/first.png"`)
	assert.Contains(t, output, `"file": "/path/second.png"`)
	assert.Contains(t, output, `"text": "First"`)
	assert.Contains(t, output, `"text": "Second"`)
}

func TestFormatJSON_NilResult(t *testing.T) {
	results := []*pipeline.OCRImageResult{nil}
	imagePaths := []string{"/path/nil.png"}

	output, err := formatJSON(results, imagePaths)
	require.NoError(t, err)

	assert.Contains(t, output, `"file": "/path/nil.png"`)
	assert.Contains(t, output, `"ocr": null`)
}

func TestFormatCSV_WithMultipleRegions(t *testing.T) {
	result := &pipeline.OCRImageResult{
		Width:  640,
		Height: 480,
		Regions: []pipeline.OCRRegionResult{
			{
				Text:          "First Region",
				RecConfidence: 0.95,
				DetConfidence: 0.90,
				Box:           struct{ X, Y, W, H int }{X: 10, Y: 10, W: 100, H: 20},
				Language:      "en",
			},
			{
				Text:          "Second Region",
				RecConfidence: 0.88,
				DetConfidence: 0.85,
				Box:           struct{ X, Y, W, H int }{X: 20, Y: 40, W: 80, H: 15},
				Language:      "en",
			},
		},
		AvgDetConf: 0.875,
	}
	results := []*pipeline.OCRImageResult{result}
	imagePaths := []string{"/path/multi.png"}

	output, err := formatCSV(results, imagePaths)
	require.NoError(t, err)

	lines := strings.Split(strings.TrimSpace(output), "\n")
	assert.Len(t, lines, 3) // Header + 2 data rows

	// Check both data rows contain the filename
	assert.Contains(t, lines[1], "/path/multi.png")
	assert.Contains(t, lines[2], "/path/multi.png")

	// Check first region
	assert.Contains(t, lines[1], "First Region")
	assert.Contains(t, lines[1], "0.950")
	assert.Contains(t, lines[1], "0.900")

	// Check second region
	assert.Contains(t, lines[2], "Second Region")
	assert.Contains(t, lines[2], "0.880")
	assert.Contains(t, lines[2], "0.850")
}

func TestFormatCSV_EmptyRegions(t *testing.T) {
	result := &pipeline.OCRImageResult{
		Width:      640,
		Height:     480,
		Regions:    []pipeline.OCRRegionResult{}, // No regions
		AvgDetConf: 0.0,
	}
	results := []*pipeline.OCRImageResult{result}
	imagePaths := []string{"/path/empty.png"}

	output, err := formatCSV(results, imagePaths)
	require.NoError(t, err)

	lines := strings.Split(strings.TrimSpace(output), "\n")
	assert.Len(t, lines, 2) // Header + 1 empty data row

	// Check empty row
	assert.Contains(t, lines[1], "/path/empty.png")
	assert.Contains(t, lines[1], "0")                // region_index
	assert.Equal(t, 9, strings.Count(lines[1], ",")) // 10 columns total
}

func TestFormatText_SingleResult(t *testing.T) {
	results := []*pipeline.OCRImageResult{
		createMockOCRResult("Plain text output", 0.87),
	}
	imagePaths := []string{"/path/text.png"}

	output, err := formatText(results, imagePaths)
	require.NoError(t, err)

	assert.Contains(t, output, "# /path/text.png")
	assert.Contains(t, output, "Plain text output")
}

func TestFormatText_MultipleResults(t *testing.T) {
	results := []*pipeline.OCRImageResult{
		createMockOCRResult("First text", 0.90),
		createMockOCRResult("Second text", 0.85),
	}
	imagePaths := []string{"/path/first.png", "/path/second.png"}

	output, err := formatText(results, imagePaths)
	require.NoError(t, err)

	// Should contain both files with newlines between them
	assert.Contains(t, output, "# /path/first.png")
	assert.Contains(t, output, "# /path/second.png")
	assert.Contains(t, output, "First text")
	assert.Contains(t, output, "Second text")

	// Check for proper separation (single newline between results)
	parts := strings.Split(strings.TrimSpace(output), "\n")
	// Should have header for first, text for first, header for second, text for second
	assert.GreaterOrEqual(t, len(parts), 4)
	assert.Contains(t, parts[0], "# /path/first.png")
	assert.Contains(t, parts[2], "# /path/second.png")
}

func TestFormatText_NilResult(t *testing.T) {
	results := []*pipeline.OCRImageResult{nil}
	imagePaths := []string{"/path/nil.png"}

	output, err := formatText(results, imagePaths)
	require.NoError(t, err)

	assert.Contains(t, output, "# /path/nil.png")
	// Should not contain any text after the header
	lines := strings.Split(output, "\n")
	assert.Len(t, lines, 1) // Just the header line
}

func TestFormatText_EmptyRegions(t *testing.T) {
	result := &pipeline.OCRImageResult{
		Width:      640,
		Height:     480,
		Regions:    []pipeline.OCRRegionResult{},
		AvgDetConf: 0.0,
	}
	results := []*pipeline.OCRImageResult{result}
	imagePaths := []string{"/path/empty.png"}

	output, err := formatText(results, imagePaths)
	require.NoError(t, err)

	assert.Contains(t, output, "# /path/empty.png")
	// Should be just the header with no text content
	assert.Equal(t, "# /path/empty.png\n", output)
}
