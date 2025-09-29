package batch

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResult_FormatResults_Text(t *testing.T) {
	result := &Result{
		Results: []*pipeline.OCRImageResult{
			createMockOCRResult("Hello World", 0.95),
			createMockOCRResult("Test Image", 0.88),
		},
		ImagePaths:  []string{"/path/image1.png", "/path/image2.png"},
		Duration:    time.Second * 5,
		WorkerCount: 2,
	}

	output, err := result.FormatResults("text")
	require.NoError(t, err)
	assert.Contains(t, output, "# /path/image1.png")
	assert.Contains(t, output, "# /path/image2.png")
	assert.Contains(t, output, "Hello World")
	assert.Contains(t, output, "Test Image")
}

func TestResult_FormatResults_JSON(t *testing.T) {
	result := &Result{
		Results: []*pipeline.OCRImageResult{
			createMockOCRResult("Hello World", 0.95),
		},
		ImagePaths:  []string{"/path/image1.png"},
		Duration:    time.Second * 5,
		WorkerCount: 1,
	}

	output, err := result.FormatResults("json")
	require.NoError(t, err)

	assert.Contains(t, output, `"file": "/path/image1.png"`)
	assert.Contains(t, output, `"text": "Hello World"`)
	assert.Contains(t, output, `"rec_confidence": 0.9`)

	// Should be valid JSON
	var jsonResult interface{}
	require.NoError(t, json.Unmarshal([]byte(output), &jsonResult))
}

func TestResult_FormatResults_CSV(t *testing.T) {
	result := &Result{
		Results: []*pipeline.OCRImageResult{
			createMockOCRResult("Hello World", 0.95),
		},
		ImagePaths:  []string{"/path/image1.png"},
		Duration:    time.Second * 5,
		WorkerCount: 1,
	}

	output, err := result.FormatResults("csv")
	require.NoError(t, err)

	lines := strings.Split(strings.TrimSpace(output), "\n")
	assert.Len(t, lines, 2) // Header + 1 data row

	// Check header
	assert.Contains(t, lines[0], "file")
	assert.Contains(t, lines[0], "text")
	assert.Contains(t, lines[0], "confidence")

	// Check data row
	assert.Contains(t, lines[1], "/path/image1.png")
	assert.Contains(t, lines[1], "Hello World")
	assert.Contains(t, lines[1], "0.950")
}

func TestResult_FormatResults_InvalidFormat(t *testing.T) {
	result := &Result{
		Results:     []*pipeline.OCRImageResult{},
		ImagePaths:  []string{},
		Duration:    time.Second,
		WorkerCount: 1,
	}

	// Invalid format defaults to text format
	output, err := result.FormatResults("invalid")
	require.NoError(t, err)
	assert.Empty(t, output) // Empty results produce empty text output
}

func TestResult_SaveResults_ToFile(t *testing.T) {
	tempDir := testutil.CreateTempDir(t)
	outputFile := filepath.Join(tempDir, "results.txt")

	result := &Result{
		Results: []*pipeline.OCRImageResult{
			createMockOCRResult("Test Content", 0.90),
		},
		ImagePaths:  []string{"/path/test.png"},
		Duration:    time.Second * 2,
		WorkerCount: 1,
	}

	err := result.SaveResults("text", outputFile, true)
	require.NoError(t, err)

	// Check file was created and contains expected content
	content, err := os.ReadFile(outputFile)
	require.NoError(t, err)
	assert.Contains(t, string(content), "Test Content")
}

func TestResult_SaveResults_Stdout(t *testing.T) {
	result := &Result{
		Results: []*pipeline.OCRImageResult{
			createMockOCRResult("Console Output", 0.85),
		},
		ImagePaths:  []string{"/path/console.png"},
		Duration:    time.Second * 3,
		WorkerCount: 1,
	}

	// Capture stdout
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	require.NoError(t, err)
	os.Stdout = w

	err = result.SaveResults("text", "", true) // Empty outputFile means stdout
	require.NoError(t, err)

	// Restore stdout and read captured output
	require.NoError(t, w.Close())
	os.Stdout = oldStdout

	var buf bytes.Buffer
	_, err = buf.ReadFrom(r)
	require.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "Console Output")
}

func TestResult_SaveResults_FormatError(t *testing.T) {
	tempDir := testutil.CreateTempDir(t)
	outputFile := filepath.Join(tempDir, "results.txt")

	result := &Result{
		Results:     []*pipeline.OCRImageResult{},
		ImagePaths:  []string{},
		Duration:    time.Second,
		WorkerCount: 1,
	}

	// Invalid format defaults to text format, should succeed
	err := result.SaveResults("invalid", outputFile, true)
	assert.NoError(t, err)

	// Verify file was written
	content, err := os.ReadFile(outputFile)
	require.NoError(t, err)
	assert.Empty(t, content) // Empty results produce empty output
}

func TestResult_SaveResults_WriteError(t *testing.T) {
	// Try to write to a directory that doesn't exist and can't be created
	invalidPath := "/nonexistent/deep/path/results.txt"

	result := &Result{
		Results: []*pipeline.OCRImageResult{
			createMockOCRResult("Test", 0.80),
		},
		ImagePaths:  []string{"/path/test.png"},
		Duration:    time.Second,
		WorkerCount: 1,
	}

	err := result.SaveResults("text", invalidPath, true)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to write output file")
}

func TestResult_PrintStats_Quiet(t *testing.T) {
	result := &Result{
		Results:     []*pipeline.OCRImageResult{},
		ImagePaths:  []string{},
		Duration:    time.Second * 10,
		WorkerCount: 4,
	}

	// Capture stdout
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	require.NoError(t, err)
	os.Stdout = w

	result.PrintStats(false) // Not quiet

	// Restore stdout and read captured output
	require.NoError(t, w.Close())
	os.Stdout = oldStdout

	var buf bytes.Buffer
	_, err = buf.ReadFrom(r)
	require.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "Processing Statistics:")
	assert.Contains(t, output, "Total images: 0")
	assert.Contains(t, output, "Workers: 4")
	assert.Contains(t, output, "Duration:")
}

func TestResult_PrintStats_WithResults(t *testing.T) {
	result := &Result{
		Results: []*pipeline.OCRImageResult{
			createMockOCRResult("Text 1", 0.90),
			createMockOCRResult("Text 2", 0.85),
		},
		ImagePaths:  []string{"img1.png", "img2.png"},
		Duration:    time.Millisecond * 1500, // 1.5 seconds
		WorkerCount: 2,
	}

	// Capture stdout
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	require.NoError(t, err)
	os.Stdout = w

	result.PrintStats(false)

	// Restore stdout and read captured output
	require.NoError(t, w.Close())
	os.Stdout = oldStdout

	var buf bytes.Buffer
	_, err = buf.ReadFrom(r)
	require.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "Total images: 2")
	assert.Contains(t, output, "Processed: 2")
	assert.Contains(t, output, "Failed: 0")
	assert.Contains(t, output, "Workers: 2")
	assert.Contains(t, output, "Duration:")
	assert.Contains(t, output, "Avg per image:")
	assert.Contains(t, output, "images/sec")
}

// Helper function to create mock OCR results for testing.
func createMockOCRResult(text string, confidence float64) *pipeline.OCRImageResult {
	return &pipeline.OCRImageResult{
		Width:  640,
		Height: 480,
		Regions: []pipeline.OCRRegionResult{
			{
				Text:          text,
				RecConfidence: confidence,
				DetConfidence: confidence,
				Box: struct{ X, Y, W, H int }{
					X: 10,
					Y: 10,
					W: 100,
					H: 20,
				},
				Language: "en",
			},
		},
		AvgDetConf: confidence,
	}
}
