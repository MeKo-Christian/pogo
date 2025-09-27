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

// ExtractImages extracts all images from a PDF file using pdfcpu's extract functionality.
func ExtractImages(filename string, pageRange string) (map[int][]image.Image, error) {
	// Parse page range if provided
	pageNumbers, err := parsePageRange(pageRange)
	if err != nil {
		return nil, fmt.Errorf("invalid page range %q: %w", pageRange, err)
	}

	// Create temporary directory for extracted images
	tempDir, err := os.MkdirTemp("", "pdf-extract-*")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer func() { _ = os.RemoveAll(tempDir) }() // Clean up

	// Convert page numbers to strings if provided
	var pageStrings []string
	if len(pageNumbers) > 0 {
		pageStrings = make([]string, len(pageNumbers))
		for i, pageNum := range pageNumbers {
			pageStrings[i] = strconv.Itoa(pageNum)
		}
	}

	// Extract images using pdfcpu
	err = api.ExtractImagesFile(filename, tempDir, pageStrings, nil)
	if err != nil {
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
			return nil
		}

		img, err := loadImageFile(path)
		if err != nil {
			// Skip unreadable images
			return nil
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
