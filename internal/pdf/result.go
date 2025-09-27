package pdf

import (
	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/utils"
)

// PageResult represents OCR results for a single PDF page.
type PageResult struct {
	PageNumber int            `json:"page_number"`
	Width      int            `json:"width"`
	Height     int            `json:"height"`
	Images     []ImageResult  `json:"images"`
	Processing ProcessingInfo `json:"processing"`
}

// ImageResult represents OCR results for a single image extracted from a PDF page.
type ImageResult struct {
	ImageIndex int                       `json:"image_index"`
	Width      int                       `json:"width"`
	Height     int                       `json:"height"`
	Regions    []detector.DetectedRegion `json:"regions"`
	Confidence float64                   `json:"confidence"`
	// Enriched OCR output (optional; present when processed via full pipeline)
	OCRRegions []OCRRegion `json:"ocr_regions,omitempty"`
	Text       string      `json:"text,omitempty"`
}

// DocumentResult represents complete OCR results for a PDF document.
type DocumentResult struct {
	Filename   string         `json:"filename"`
	TotalPages int            `json:"total_pages"`
	Pages      []PageResult   `json:"pages"`
	Processing ProcessingInfo `json:"processing"`
}

// ProcessingInfo contains timing and performance information.
type ProcessingInfo struct {
	ExtractionTimeMs int64 `json:"extraction_time_ms"`
	DetectionTimeMs  int64 `json:"detection_time_ms"`
	TotalTimeMs      int64 `json:"total_time_ms"`
}

// OCRRegion mirrors key fields from pipeline OCRRegionResult for JSON output.
type OCRRegion struct {
	Polygon       []utils.Point            `json:"polygon"`
	Box           struct{ X, Y, W, H int } `json:"box"`
	DetConfidence float64                  `json:"det_confidence"`
	Text          string                   `json:"text"`
	RecConfidence float64                  `json:"rec_confidence"`
	Language      string                   `json:"language,omitempty"`
}
