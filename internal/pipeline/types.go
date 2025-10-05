package pipeline

// OCRRegionResult combines detection geometry with recognition output.
type OCRRegionResult struct {
	// Geometry and detection
	Polygon       []struct{ X, Y float64 } `json:"polygon"`
	Box           struct{ X, Y, W, H int } `json:"box"`
	DetConfidence float64                  `json:"det_confidence"`

	// Recognition
	Text            string    `json:"text"`
	RecConfidence   float64   `json:"rec_confidence"`
	CharConfidences []float64 `json:"char_confidences,omitempty"`
	Rotated         bool      `json:"rotated"`
	Language        string    `json:"language,omitempty"`

	// Timing
	Timing struct {
		RecognizePreprocessNs int64 `json:"recognize_preprocess_ns"`
		RecognizeModelNs      int64 `json:"recognize_model_ns"`
		RecognizeDecodeNs     int64 `json:"recognize_decode_ns"`
		RecognizeTotalNs      int64 `json:"recognize_total_ns"`
	} `json:"timing"`
}

// OCRImageResult is the per-image aggregated OCR output.
type OCRImageResult struct {
    Width       int               `json:"width"`
    Height      int               `json:"height"`
    Regions     []OCRRegionResult `json:"regions"`
    Barcodes    []BarcodeResult   `json:"barcodes,omitempty"`
    AvgDetConf  float64           `json:"avg_det_confidence"`
	Orientation struct {
		Angle      int     `json:"angle"`
		Confidence float64 `json:"confidence"`
		Applied    bool    `json:"applied"`
	} `json:"orientation"`
	Processing struct {
		DetectionNs   int64 `json:"detection_ns"`
		RecognitionNs int64 `json:"recognition_ns"`
		TotalNs       int64 `json:"total_ns"`
	} `json:"processing"`
}

// OCRPDFResult represents the OCR result for a PDF document.
type OCRPDFResult struct {
	Filename   string             `json:"filename"`
	TotalPages int                `json:"total_pages"`
	Pages      []OCRPDFPageResult `json:"pages"`
	Processing struct {
		ExtractionNs int64 `json:"extraction_ns"`
		TotalNs      int64 `json:"total_ns"`
	} `json:"processing"`
}

// OCRPDFPageResult represents OCR results for a single PDF page.
type OCRPDFPageResult struct {
	PageNumber int                 `json:"page_number"`
	Width      int                 `json:"width"`
	Height     int                 `json:"height"`
	Images     []OCRPDFImageResult `json:"images"`
	Processing struct {
		TotalNs int64 `json:"total_ns"`
	} `json:"processing"`
}

// OCRPDFImageResult represents OCR results for a single image extracted from a PDF page.
type OCRPDFImageResult struct {
    ImageIndex int               `json:"image_index"`
    Width      int               `json:"width"`
    Height     int               `json:"height"`
    Regions    []OCRRegionResult `json:"regions"`
    Barcodes   []BarcodeResult   `json:"barcodes,omitempty"`
    Confidence float64           `json:"confidence"`
}

// BarcodeResult represents a decoded barcode in image coordinates.
type BarcodeResult struct {
    Type       string              `json:"type"`
    Value      string              `json:"value"`
    Confidence float64             `json:"confidence"`
    Rotation   float64             `json:"rotation"`
    Box        struct{ X, Y, W, H int } `json:"box"`
    Points     []struct{ X, Y int } `json:"points,omitempty"`
}
