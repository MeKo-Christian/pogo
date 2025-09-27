package pdf

import (
	"fmt"
	"image"
	"time"

	"github.com/MeKo-Tech/pogo/internal/detector"
)

// Processor handles PDF OCR processing using a text detector.
type Processor struct {
	detector *detector.Detector
}

// NewProcessor creates a new PDF processor with the given detector.
func NewProcessor(det *detector.Detector) *Processor {
	return &Processor{
		detector: det,
	}
}

// ProcessFile processes a single PDF file and returns OCR results.
func (p *Processor) ProcessFile(filename string, pageRange string) (*DocumentResult, error) {
	startTime := time.Now()

	// Extract images from PDF
	extractStart := time.Now()
	pageImages, err := ExtractImages(filename, pageRange)
	if err != nil {
		return nil, fmt.Errorf("failed to extract images from PDF: %w", err)
	}
	extractTime := time.Since(extractStart)

	// Process each page
	pages := make([]PageResult, 0, len(pageImages))
	var totalDetectionTime time.Duration

	for pageNum, images := range pageImages {
		pageResult, detectionTime, err := p.processPage(pageNum, images)
		if err != nil {
			return nil, fmt.Errorf("failed to process page %d: %w", pageNum, err)
		}

		pages = append(pages, *pageResult)
		totalDetectionTime += detectionTime
	}

	totalTime := time.Since(startTime)

	result := &DocumentResult{
		Filename:   filename,
		TotalPages: len(pages),
		Pages:      pages,
		Processing: ProcessingInfo{
			ExtractionTimeMs: extractTime.Milliseconds(),
			DetectionTimeMs:  totalDetectionTime.Milliseconds(),
			TotalTimeMs:      totalTime.Milliseconds(),
		},
	}

	return result, nil
}

// processPage processes all images from a single PDF page.
func (p *Processor) processPage(pageNum int, images []image.Image) (*PageResult, time.Duration, error) {
	imageResults := make([]ImageResult, 0, len(images))
	var totalDetectionTime time.Duration
	var pageWidth, pageHeight int

	for i, img := range images {
		bounds := img.Bounds()
		imgWidth := bounds.Dx()
		imgHeight := bounds.Dy()

		// Update page dimensions (use largest image dimensions)
		if imgWidth > pageWidth {
			pageWidth = imgWidth
		}
		if imgHeight > pageHeight {
			pageHeight = imgHeight
		}

		// Run OCR detection on the image
		detectionStart := time.Now()
		regions, err := p.detector.DetectRegions(img)
		if err != nil {
			return nil, 0, fmt.Errorf("OCR detection failed for image %d: %w", i, err)
		}
		detectionTime := time.Since(detectionStart)
		totalDetectionTime += detectionTime

		// Calculate average confidence
		var totalConfidence float64
		for _, region := range regions {
			totalConfidence += region.Confidence
		}

		avgConfidence := 0.0
		if len(regions) > 0 {
			avgConfidence = totalConfidence / float64(len(regions))
		}

		imageResult := ImageResult{
			ImageIndex: i,
			Width:      imgWidth,
			Height:     imgHeight,
			Regions:    regions,
			Confidence: avgConfidence,
		}

		imageResults = append(imageResults, imageResult)
	}

	pageResult := &PageResult{
		PageNumber: pageNum,
		Width:      pageWidth,
		Height:     pageHeight,
		Images:     imageResults,
		Processing: ProcessingInfo{
			DetectionTimeMs: totalDetectionTime.Milliseconds(),
			TotalTimeMs:     totalDetectionTime.Milliseconds(),
		},
	}

	return pageResult, totalDetectionTime, nil
}

// ProcessFiles processes multiple PDF files.
func (p *Processor) ProcessFiles(filenames []string, pageRange string) ([]*DocumentResult, error) {
	results := make([]*DocumentResult, 0, len(filenames))

	for _, filename := range filenames {
		result, err := p.ProcessFile(filename, pageRange)
		if err != nil {
			return nil, fmt.Errorf("failed to process %s: %w", filename, err)
		}
		results = append(results, result)
	}

	return results, nil
}
