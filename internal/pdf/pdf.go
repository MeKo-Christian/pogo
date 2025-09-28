package pdf

import (
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/pdfcpu/pdfcpu/pkg/api"
)

// validateAndFilterPages filters page numbers to valid range and returns valid pages.
func validateAndFilterPages(pageNumbers []int, pageCount int) []int {
	var validPages []int
	for _, page := range pageNumbers {
		if page >= 1 && page <= pageCount {
			validPages = append(validPages, page)
		}
	}
	return validPages
}

// shouldReturnEmpty checks if we should return empty result when no valid pages found.
func shouldReturnEmpty(pageNumbers []int, validPages []int) bool {
	return len(pageNumbers) > 0 && len(validPages) == 0
}

// preparePageStrings converts valid page numbers to string slice for pdfcpu.
func preparePageStrings(validPages []int) []string {
	if len(validPages) == 0 {
		return nil
	}
	pageStrings := make([]string, len(validPages))
	for i, pageNum := range validPages {
		pageStrings[i] = strconv.Itoa(pageNum)
	}
	return pageStrings
}

// extractImagesFromPDF performs the actual image extraction using pdfcpu.
func extractImagesFromPDF(filename string, tempDir string, pageStrings []string) error {
	return api.ExtractImagesFile(filename, tempDir, pageStrings, nil)
}

// ExtractImages extracts all images from a PDF file using pdfcpu's extract functionality.
func ExtractImages(filename string, pageRange string) (map[int][]image.Image, error) {
	// Parse page range if provided
	pageNumbers, err := parsePageRange(pageRange)
	if err != nil {
		return nil, fmt.Errorf("invalid page range %q: %w", pageRange, err)
	}

	// Get total page count
	pageCount, err := api.PageCountFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to get page count: %w", err)
	}

	// Filter page numbers to valid range
	validPages := validateAndFilterPages(pageNumbers, pageCount)

	// If no valid pages requested but some were specified, return empty result
	if shouldReturnEmpty(pageNumbers, validPages) {
		return make(map[int][]image.Image), nil
	}

	// Create temporary directory for extracted images
	tempDir, err := os.MkdirTemp("", "pdf-extract-*")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer func() { _ = os.RemoveAll(tempDir) }() // Clean up

	// Prepare page strings for extraction
	pageStrings := preparePageStrings(validPages)

	// Extract images using pdfcpu
	if err := extractImagesFromPDF(filename, tempDir, pageStrings); err != nil {
		return nil, fmt.Errorf("failed to extract images from PDF: %w", err)
	}

	// Load extracted images
	result, err := collectExtractedImages(tempDir)
	if err != nil {
		return nil, fmt.Errorf("failed to process extracted images: %w", err)
	}

	return result, nil
}

// loadImageFile loads an image from a file path.
func loadImageFile(path string) (image.Image, error) {
	file, err := os.Open(path) //nolint:gosec // G304: Reading user-provided PDF file path is expected
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()

	img, _, err := image.Decode(file)
	return img, err
}

// collectExtractedImages walks the given directory and groups images by page number.
// It expects filenames in the pdfcpu format: page_<num>_image_<idx>.<ext>.
func collectExtractedImages(dir string) (map[int][]image.Image, error) {
	result := make(map[int][]image.Image)

	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		pageNum, err := parsePageFromFilename(info.Name())
		if err != nil {
			// Skip files we can't parse as page images
			return nil //nolint:nilerr // Continue processing other files
		}

		img, err := loadImageFile(path)
		if err != nil {
			// Skip unreadable images
			return nil //nolint:nilerr // Continue processing other files
		}

		if img != nil {
			result[pageNum] = append(result[pageNum], img)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// parsePageFromFilename extracts page number from pdfcpu extracted filename.
func parsePageFromFilename(filename string) (int, error) {
	// pdfcpu creates files like: page_1_image_1.png, page_2_image_1.jpg, etc.
	if !strings.HasPrefix(filename, "page_") {
		return 0, errors.New("not a page file")
	}

	parts := strings.Split(filename, "_")
	if len(parts) < 2 {
		return 0, errors.New("invalid filename format")
	}

	pageNum, err := strconv.Atoi(parts[1])
	if err != nil {
		return 0, errors.New("invalid page number")
	}

	return pageNum, nil
}

// parsePageRange parses a page range string like "1-5" or "1,3,5".
func parsePageRange(pageRange string) ([]int, error) {
	if pageRange == "" {
		return nil, nil // Empty means all pages
	}

	var pages []int

	// Split by commas for individual pages/ranges
	parts := strings.Split(pageRange, ",")

	for _, part := range parts {
		part = strings.TrimSpace(part)
		// Delegate parsing of each token to reduce nesting
		tokenPages, err := parseRangeToken(part)
		if err != nil {
			return nil, err
		}
		pages = append(pages, tokenPages...)
	}

	return pages, nil
}

// parseRangeToken parses either a single page token (e.g., "3") or a range token (e.g., "1-5").
func parseRangeToken(part string) ([]int, error) {
	if strings.Contains(part, "-") {
		rangeParts := strings.Split(part, "-")
		if len(rangeParts) != 2 {
			return nil, fmt.Errorf("invalid range format: %s", part)
		}
		start, err := strconv.Atoi(strings.TrimSpace(rangeParts[0]))
		if err != nil {
			return nil, fmt.Errorf("invalid start page: %s", rangeParts[0])
		}
		end, err := strconv.Atoi(strings.TrimSpace(rangeParts[1]))
		if err != nil {
			return nil, fmt.Errorf("invalid end page: %s", rangeParts[1])
		}
		if start > end {
			return nil, fmt.Errorf("start page %d greater than end page %d", start, end)
		}
		out := make([]int, 0, end-start+1)
		for i := start; i <= end; i++ {
			out = append(out, i)
		}
		return out, nil
	}
	page, err := strconv.Atoi(part)
	if err != nil {
		return nil, fmt.Errorf("invalid page number: %s", part)
	}
	return []int{page}, nil
}
