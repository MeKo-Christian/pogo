package pipeline

import (
	"context"
	"errors"
	"fmt"
	"image"
	"time"

	"github.com/MeKo-Tech/pogo/internal/pdf"
)

// ProcessPDF processes a PDF file and returns OCR results for all pages.
func (p *Pipeline) ProcessPDF(filename string, pageRange string) (*OCRPDFResult, error) {
	return p.ProcessPDFContext(context.Background(), filename, pageRange)
}

// ProcessPDFContext processes a PDF file with context cancellation support.
func (p *Pipeline) ProcessPDFContext(ctx context.Context, filename string, pageRange string) (*OCRPDFResult, error) {
	if filename == "" {
		return nil, errors.New("filename cannot be empty")
	}
	if p == nil || p.Detector == nil || p.Recognizer == nil {
		return nil, errors.New("pipeline not initialized")
	}

	totalStart := time.Now()

	// Extract images from PDF
	extractStart := time.Now()
	pageImages, err := pdf.ExtractImages(filename, pageRange)
	if err != nil {
		return nil, fmt.Errorf("failed to extract images from PDF: %w", err)
	}
	extractNs := time.Since(extractStart).Nanoseconds()

	// Process each page
	pages := make([]OCRPDFPageResult, 0, len(pageImages))
	for pageNum, images := range pageImages {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		pageResult, err := p.processPDFPage(ctx, pageNum, images)
		if err != nil {
			return nil, fmt.Errorf("failed to process page %d: %w", pageNum, err)
		}

		pages = append(pages, *pageResult)
	}

	totalNs := time.Since(totalStart).Nanoseconds()

	result := &OCRPDFResult{
		Filename:   filename,
		TotalPages: len(pages),
		Pages:      pages,
		Processing: struct {
			ExtractionNs int64 `json:"extraction_ns"`
			TotalNs      int64 `json:"total_ns"`
		}{
			ExtractionNs: extractNs,
			TotalNs:      totalNs,
		},
	}

	return result, nil
}

// processPDFPage processes all images from a single PDF page.
func (p *Pipeline) processPDFPage(ctx context.Context, pageNum int, images []image.Image) (*OCRPDFPageResult, error) {
	pageStart := time.Now()

	imageResults := make([]OCRPDFImageResult, 0, len(images))
	var pageWidth, pageHeight int

	for i, img := range images {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		// Update page dimensions (use largest image dimensions)
		bounds := img.Bounds()
		imgWidth := bounds.Dx()
		imgHeight := bounds.Dy()
		if imgWidth > pageWidth {
			pageWidth = imgWidth
		}
		if imgHeight > pageHeight {
			pageHeight = imgHeight
		}

		// Process image with full OCR pipeline
		ocrResult, err := p.ProcessImageContext(ctx, img)
		if err != nil {
			return nil, fmt.Errorf("OCR processing failed for image %d: %w", i, err)
		}

        imageResult := OCRPDFImageResult{
            ImageIndex: i,
            Width:      imgWidth,
            Height:     imgHeight,
            Regions:    ocrResult.Regions,
            Barcodes:   ocrResult.Barcodes,
            Confidence: ocrResult.AvgDetConf,
        }

		imageResults = append(imageResults, imageResult)
	}

	pageNs := time.Since(pageStart).Nanoseconds()

	pageResult := &OCRPDFPageResult{
		PageNumber: pageNum,
		Width:      pageWidth,
		Height:     pageHeight,
		Images:     imageResults,
		Processing: struct {
			TotalNs int64 `json:"total_ns"`
		}{
			TotalNs: pageNs,
		},
	}

	return pageResult, nil
}
