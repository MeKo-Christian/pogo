package server

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"mime/multipart"
	"net/http"
	"net/http/httptest"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
)

// mockPipeline is a simple mock implementation of the pipeline for testing.
type mockPipeline struct{}

// ProcessImage returns a mock OCR result for testing.
func (m *mockPipeline) ProcessImage(img image.Image) (*pipeline.OCRImageResult, error) {
	bounds := img.Bounds()
	return &pipeline.OCRImageResult{
		Width:  bounds.Dx(),
		Height: bounds.Dy(),
		Regions: []pipeline.OCRRegionResult{
			{
				Polygon: []struct{ X, Y float64 }{
					{X: 10, Y: 10},
					{X: 100, Y: 10},
					{X: 100, Y: 30},
					{X: 10, Y: 30},
				},
				Box: struct{ X, Y, W, H int }{
					X: 10, Y: 10, W: 90, H: 20,
				},
				DetConfidence:   0.95,
				Text:            "Hello World",
				RecConfidence:   0.92,
				CharConfidences: []float64{0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9},
				Rotated:         false,
			},
		},
		AvgDetConf: 0.95,
		Orientation: struct {
			Angle      int     `json:"angle"`
			Confidence float64 `json:"confidence"`
			Applied    bool    `json:"applied"`
		}{
			Angle:      0,
			Confidence: 0.99,
			Applied:    false,
		},
		Processing: struct {
			DetectionNs   int64 `json:"detection_ns"`
			RecognitionNs int64 `json:"recognition_ns"`
			TotalNs       int64 `json:"total_ns"`
		}{
			DetectionNs:   1000000, // 1ms
			RecognitionNs: 2000000, // 2ms
			TotalNs:       3000000, // 3ms
		},
	}, nil
}

// ProcessPDF returns a mock PDF OCR result for testing.
func (m *mockPipeline) ProcessPDF(filename string, pageRange string) (*pipeline.OCRPDFResult, error) {
	return &pipeline.OCRPDFResult{
		Filename:   filename,
		TotalPages: 1,
		Pages: []pipeline.OCRPDFPageResult{
			{
				PageNumber: 1,
				Width:      612,
				Height:     792,
				Images: []pipeline.OCRPDFImageResult{
					{
						ImageIndex: 0,
						Width:      612,
						Height:     792,
						Regions: []pipeline.OCRRegionResult{
							{
								Text:          "Sample PDF text",
								RecConfidence: 0.95,
								Box:           struct{ X, Y, W, H int }{X: 50, Y: 50, W: 200, H: 30},
							},
						},
						Confidence: 0.95,
					},
				},
			},
		},
		Processing: struct {
			ExtractionNs int64 `json:"extraction_ns"`
			TotalNs      int64 `json:"total_ns"`
		}{
			ExtractionNs: 5000000, // 5ms
			TotalNs:      8000000, // 8ms
		},
	}, nil
}

func (m *mockPipeline) Close() error {
	return nil
}

// mockPipelineForTesting is a mock implementation of pipelineInterface for testing.
type mockPipelineForTesting struct {
	processImageResult *pipeline.OCRImageResult
	processImageError  error
	processPDFResult   *pipeline.OCRPDFResult
	processPDFError    error
}

func (m *mockPipelineForTesting) ProcessImage(img image.Image) (*pipeline.OCRImageResult, error) {
	return m.processImageResult, m.processImageError
}

func (m *mockPipelineForTesting) ProcessPDF(filename string, pageRange string) (*pipeline.OCRPDFResult, error) {
	return m.processPDFResult, m.processPDFError
}

func (m *mockPipelineForTesting) Close() error {
	return nil
}

// testServer wraps Server with mock pipeline for testing.
type testServer struct {
	*Server
	mockPipeline pipelineInterface
}

func (ts *testServer) ocrImageHandlerMock(w http.ResponseWriter, r *http.Request) {
	// Temporarily replace the server's pipeline with mock
	originalPipeline := ts.pipeline
	ts.pipeline = ts.mockPipeline
	defer func() {
		ts.pipeline = originalPipeline
	}()

	ts.ocrImageHandler(w, r)
}

// createTestImage creates a simple test image for testing.
func createTestImage(width, height int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			// Safe conversion: x and y are loop indices, guaranteed to be >= 0
			r := byte(x % 256)
			g := byte(y % 256)
			img.Set(x, y, color.RGBA{r, g, 0, 255})
		}
	}
	return img
}

// encodeImageToPNG encodes an image to PNG bytes.
func encodeImageToPNG(img image.Image) ([]byte, error) {
	var buf bytes.Buffer
	err := png.Encode(&buf, img)
	return buf.Bytes(), err
}

// createMultipartFormRequest creates a multipart form request with an image.
func createMultipartFormRequest(
	imageData []byte,
	filename string,
	extraFields map[string]string,
) (*http.Request, error) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add image file
	part, err := writer.CreateFormFile("image", filename)
	if err != nil {
		return nil, err
	}
	_, err = part.Write(imageData)
	if err != nil {
		return nil, err
	}

	// Add extra fields
	for key, value := range extraFields {
		err = writer.WriteField(key, value)
		if err != nil {
			return nil, err
		}
	}

	err = writer.Close()
	if err != nil {
		return nil, err
	}

	req := httptest.NewRequest(http.MethodPost, "/ocr/image", &buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	return req, nil
}

// createMultipartPDFFormRequest creates a multipart form request with a PDF file.
func createMultipartPDFFormRequest(
	pdfData []byte,
	filename string,
	extraFields map[string]string,
) (*http.Request, error) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add PDF file
	part, err := writer.CreateFormFile("pdf", filename)
	if err != nil {
		return nil, err
	}
	_, err = part.Write(pdfData)
	if err != nil {
		return nil, err
	}

	// Add extra fields
	for key, value := range extraFields {
		err = writer.WriteField(key, value)
		if err != nil {
			return nil, err
		}
	}

	err = writer.Close()
	if err != nil {
		return nil, err
	}

	req := httptest.NewRequest(http.MethodPost, "/ocr/pdf", &buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	return req, nil
}
