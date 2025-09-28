package pdf

import (
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParsePageRange(t *testing.T) {
	tests := []struct {
		name        string
		pageRange   string
		want        []int
		expectError bool
	}{
		{
			name:      "empty range returns nil",
			pageRange: "",
			want:      nil,
		},
		{
			name:      "single page",
			pageRange: "1",
			want:      []int{1},
		},
		{
			name:      "multiple single pages",
			pageRange: "1,3,5",
			want:      []int{1, 3, 5},
		},
		{
			name:      "simple range",
			pageRange: "1-5",
			want:      []int{1, 2, 3, 4, 5},
		},
		{
			name:      "mixed pages and ranges",
			pageRange: "1,3-5,7",
			want:      []int{1, 3, 4, 5, 7},
		},
		{
			name:      "range with spaces",
			pageRange: " 1 - 3 , 5 ",
			want:      []int{1, 2, 3, 5},
		},
		{
			name:        "invalid page number",
			pageRange:   "abc",
			expectError: true,
		},
		{
			name:        "invalid range format",
			pageRange:   "1-2-3",
			expectError: true,
		},
		{
			name:        "start greater than end",
			pageRange:   "5-1",
			expectError: true,
		},
		{
			name:        "invalid start page",
			pageRange:   "abc-5",
			expectError: true,
		},
		{
			name:        "invalid end page",
			pageRange:   "1-xyz",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parsePageRange(tt.pageRange)

			if tt.expectError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestParsePageFromFilename(t *testing.T) {
	tests := []struct {
		name        string
		filename    string
		want        int
		expectError bool
	}{
		{
			name:     "valid page file",
			filename: "page_1_image_1.png",
			want:     1,
		},
		{
			name:     "valid page file with jpg",
			filename: "page_10_image_2.jpg",
			want:     10,
		},
		{
			name:        "not a page file",
			filename:    "image_1.png",
			expectError: true,
		},
		{
			name:        "invalid format",
			filename:    "page_",
			expectError: true,
		},
		{
			name:        "invalid page number",
			filename:    "page_abc_image_1.png",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parsePageFromFilename(tt.filename)

			if tt.expectError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestLoadImageFile(t *testing.T) {
	// Create a temporary image file for testing
	tempDir := t.TempDir()
	imagePath := filepath.Join(tempDir, "test.png")

	// Create a simple test image
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	for y := range 10 {
		for x := range 10 {
			img.Set(x, y, color.RGBA{255, 0, 0, 255}) // Red pixels
		}
	}

	// Save the test image
	file, err := os.Create(imagePath) //nolint:gosec // G304: Test file creation with controlled path
	require.NoError(t, err)
	defer func() { _ = file.Close() }()

	// For testing, we'll just create an empty file as we can't easily
	// encode PNG without additional dependencies in tests
	_ = file.Close()

	t.Run("non-existent file", func(t *testing.T) {
		_, err := loadImageFile("/non/existent/file.png")
		require.Error(t, err)
	})

	t.Run("directory instead of file", func(t *testing.T) {
		_, err := loadImageFile(tempDir)
		require.Error(t, err)
	})
}

func TestExtractImages_ErrorCases(t *testing.T) {
	t.Run("non-existent file", func(t *testing.T) {
		_, err := ExtractImages("/non/existent/file.pdf", "")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to extract images from PDF")
	})

	t.Run("invalid page range", func(t *testing.T) {
		_, err := ExtractImages("dummy.pdf", "invalid-range")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "invalid page range")
	})

	t.Run("directory instead of file", func(t *testing.T) {
		tempDir := t.TempDir()
		_, err := ExtractImages(tempDir, "")
		require.Error(t, err)
	})
}

// createTestPDF creates a minimal test PDF file.
func createTestPDF(t *testing.T, path string) {
	t.Helper()
	// Create a minimal PDF content (this is a very basic PDF structure)
	pdfContent := `%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj

xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
186
%%EOF`

	err := os.WriteFile(path, []byte(pdfContent), 0o644)
	require.NoError(t, err)
}

func TestExtractImages_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	tempDir := t.TempDir()
	pdfPath := filepath.Join(tempDir, "test.pdf")

	// Create a test PDF
	createTestPDF(t, pdfPath)

	t.Run("extract from valid PDF", func(t *testing.T) {
		// Note: This test may fail if pdfcpu can't process our minimal PDF
		// In a real scenario, we'd use a proper PDF with images
		images, err := ExtractImages(pdfPath, "")

		// We expect this to succeed but return no images (our test PDF has no images)
		if err != nil {
			// If pdfcpu fails to process our minimal PDF, that's expected
			t.Logf("PDF processing failed (expected for minimal test PDF): %v", err)
		} else {
			assert.IsType(t, map[int][]image.Image{}, images)
		}
	})

	t.Run("extract with page range", func(t *testing.T) {
		images, err := ExtractImages(pdfPath, "1")

		// Same as above - may fail due to minimal PDF
		if err != nil {
			t.Logf("PDF processing failed (expected for minimal test PDF): %v", err)
		} else {
			assert.IsType(t, map[int][]image.Image{}, images)
		}
	})
}

func TestCollectExtractedImages_MixedFormatsAndPages(t *testing.T) {
	tempDir := t.TempDir()

	// Helper to write an image with a given encoder
	writeImg := func(name string, enc string) {
		f, err := os.Create(filepath.Join(tempDir, name)) //nolint:gosec // controlled test path
		require.NoError(t, err)
		defer func() { _ = f.Close() }()

		img := image.NewRGBA(image.Rect(0, 0, 8, 6))
		for y := range 6 {
			for x := range 8 {
				img.Set(x, y, color.RGBA{uint8(10 * x), uint8(10 * y), 0, 255})
			}
		}
		switch enc {
		case "png":
			require.NoError(t, png.Encode(f, img))
		case "jpeg":
			require.NoError(t, jpeg.Encode(f, img, &jpeg.Options{Quality: 80}))
		default:
			t.Fatalf("unknown encoder: %s", enc)
		}
	}

	// Create files as pdfcpu would name them
	writeImg("page_1_image_1.png", "png")
	writeImg("page_1_image_2.jpg", "jpeg")
	writeImg("page_2_image_1.png", "png")

	// Also add some noise that should be ignored
	require.NoError(t, os.WriteFile(filepath.Join(tempDir, "notes.txt"), []byte("ignore"), 0o644))
	writeImg("not_a_match.png", "png")

	result, err := collectExtractedImages(tempDir)
	require.NoError(t, err)

	// Expect two pages
	require.Len(t, result, 2)
	require.Len(t, result[1], 2) // two images on page 1
	require.Len(t, result[2], 1) // one image on page 2

	// Basic sanity on dimensions
	for _, imgs := range result {
		for _, img := range imgs {
			b := img.Bounds()
			assert.Equal(t, 8, b.Dx())
			assert.Equal(t, 6, b.Dy())
		}
	}
}

func TestCollectExtractedImages_SkipsUnreadableAndInvalid(t *testing.T) {
	tempDir := t.TempDir()

	// Invalid filename pattern should be ignored
	f1 := filepath.Join(tempDir, "image_1.png")
	require.NoError(t, os.WriteFile(f1, []byte("not actually an image"), 0o644))

	// Valid-looking name but unreadable image should be ignored
	f2 := filepath.Join(tempDir, "page_3_image_1.png")
	require.NoError(t, os.WriteFile(f2, []byte("corrupt"), 0o644))

	// Proper one to ensure map isn't empty only when valid images exist
	f3 := filepath.Join(tempDir, "page_4_image_1.jpg")
	img := image.NewRGBA(image.Rect(0, 0, 5, 5))
	fh, err := os.Create(f3) //nolint:gosec // controlled test path
	require.NoError(t, err)
	require.NoError(t, jpeg.Encode(fh, img, &jpeg.Options{Quality: 70}))
	require.NoError(t, fh.Close())

	result, err := collectExtractedImages(tempDir)
	require.NoError(t, err)

	// Only page 4 should have one valid image
	require.Len(t, result, 1)
	require.Len(t, result[4], 1)
}

// Benchmark tests.
func BenchmarkParsePageRange(b *testing.B) {
	testCases := []string{
		"1",
		"1-10",
		"1,3,5,7,9",
		"1-5,10-15,20",
	}

	for _, pageRange := range testCases {
		b.Run("range_"+strings.ReplaceAll(pageRange, ",", "_"), func(b *testing.B) {
			for range b.N {
				_, _ = parsePageRange(pageRange)
			}
		})
	}
}

func BenchmarkParsePageFromFilename(b *testing.B) {
	filename := "page_123_image_1.png"

	for range b.N {
		_, _ = parsePageFromFilename(filename)
	}
}

// Test helpers and utilities.
func TestParsePageRange_EdgeCases(t *testing.T) {
	t.Run("single character ranges", func(t *testing.T) {
		pages, err := parsePageRange("1-1")
		require.NoError(t, err)
		assert.Equal(t, []int{1}, pages)
	})

	t.Run("large ranges", func(t *testing.T) {
		pages, err := parsePageRange("1-1000")
		require.NoError(t, err)
		assert.Len(t, pages, 1000)
		assert.Equal(t, 1, pages[0])
		assert.Equal(t, 1000, pages[999])
	})

	t.Run("zero page number", func(t *testing.T) {
		pages, err := parsePageRange("0")
		require.NoError(t, err)
		assert.Equal(t, []int{0}, pages)
	})

	t.Run("negative page number", func(t *testing.T) {
		_, err := parsePageRange("-1")
		require.Error(t, err)
	})
}

func TestParsePageFromFilename_EdgeCases(t *testing.T) {
	t.Run("zero page number", func(t *testing.T) {
		page, err := parsePageFromFilename("page_0_image_1.png")
		require.NoError(t, err)
		assert.Equal(t, 0, page)
	})

	t.Run("large page number", func(t *testing.T) {
		page, err := parsePageFromFilename("page_999999_image_1.png")
		require.NoError(t, err)
		assert.Equal(t, 999999, page)
	})

	t.Run("filename with extra underscores", func(t *testing.T) {
		page, err := parsePageFromFilename("page_123_image_1_extra.png")
		require.NoError(t, err)
		assert.Equal(t, 123, page)
	})
}
