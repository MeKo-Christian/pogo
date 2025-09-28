package cmd

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/pdf"
	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/spf13/cobra"
)

// pdfCmd represents the pdf command.
var pdfCmd = &cobra.Command{
	Use:   "pdf [file...]",
	Short: "Process PDF files for OCR text extraction",
	Long: `Process PDF files to extract text using OCR on embedded images.

This command extracts images from PDF pages and performs OCR on them.
Works best with scanned PDFs or PDFs containing image-based text.

Examples:
  pogo pdf document.pdf
  pogo pdf *.pdf --format json
  pogo pdf scan.pdf --pages 1-5`,
	Args:         cobra.ArbitraryArgs,
	SilenceUsage: true,
	RunE:         processPDFs,
}

func init() {
	rootCmd.AddCommand(pdfCmd)

	// PDF-specific flags
	pdfCmd.Flags().StringP("format", "f", "text", "output format (text, json, csv)")
	pdfCmd.Flags().StringP("output", "o", "", "output file (default: stdout)")
	pdfCmd.Flags().String("pages", "", "page range to process (e.g., '1-5', '1,3,5')")
	pdfCmd.Flags().Float64("confidence", 0.5, "minimum detection confidence threshold")
	pdfCmd.Flags().StringP("language", "l", "en", "recognition language")

	// OCR/pipeline flags (align with image/batch)
	pdfCmd.Flags().String("det-model", "", "override detection model path")
	pdfCmd.Flags().String("rec-model", "", "override recognition model path")
	pdfCmd.Flags().String("dict", "", "comma-separated dictionary file paths")
	pdfCmd.Flags().String("dict-langs", "", "comma-separated language codes for dictionaries (e.g., en,de,fr)")
	pdfCmd.Flags().Int("rec-height", 0, "recognizer input height (0=auto, typical: 32 or 48)")
	pdfCmd.Flags().Bool("detect-orientation", false, "enable document orientation detection")
	pdfCmd.Flags().Float64("orientation-threshold", 0.7, "orientation confidence threshold (0..1)")
	pdfCmd.Flags().Bool("detect-textline", false, "enable per-text-line orientation detection")
	pdfCmd.Flags().Float64("textline-threshold", 0.6, "text line orientation confidence threshold (0..1)")
	pdfCmd.Flags().Bool("rectify", false, "enable document rectification (experimental)")
	pdfCmd.Flags().String("rectify-model",
		models.GetLayoutModelPath("", models.LayoutUVDoc), "override rectification model path")
	pdfCmd.Flags().Float64("rectify-mask-threshold", 0.5, "rectification mask threshold (0..1)")
	pdfCmd.Flags().Int("rectify-height", 1024, "rectified page output height (advisory)")
	pdfCmd.Flags().String("rectify-debug-dir", "",
		"directory to write rectification debug images (mask, overlay, compare)")
}

// processPDFs handles the main PDF processing logic.
func processPDFs(cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		return errors.New("no input files provided")
	}

	// Flags
	detConf, _ := cmd.Flags().GetFloat64("confidence")
	modelsDir, _ := cmd.InheritedFlags().GetString("models-dir")
	pages, _ := cmd.Flags().GetString("pages")
	format, _ := cmd.Flags().GetString("format")
	outputFile, _ := cmd.Flags().GetString("output")
	lang, _ := cmd.Flags().GetString("language")
	detModel, _ := cmd.Flags().GetString("det-model")
	recModel, _ := cmd.Flags().GetString("rec-model")
	dictCSV, _ := cmd.Flags().GetString("dict")
	dictLangs, _ := cmd.Flags().GetString("dict-langs")
	recH, _ := cmd.Flags().GetInt("rec-height")
	detectOrientation, _ := cmd.Flags().GetBool("detect-orientation")
	orientThresh, _ := cmd.Flags().GetFloat64("orientation-threshold")
	detectTextline, _ := cmd.Flags().GetBool("detect-textline")
	textlineThresh, _ := cmd.Flags().GetFloat64("textline-threshold")
	rectify, _ := cmd.Flags().GetBool("rectify")
	rectifyModel, _ := cmd.Flags().GetString("rectify-model")
	rectifyMask, _ := cmd.Flags().GetFloat64("rectify-mask-threshold")
	rectifyHeight, _ := cmd.Flags().GetInt("rectify-height")
	rectifyDebugDir, _ := cmd.Flags().GetString("rectify-debug-dir")

	// Validate confidence threshold
	if detConf < 0 || detConf > 1 {
		return fmt.Errorf("invalid confidence threshold: %.2f (must be between 0.0 and 1.0)", detConf)
	}

	// Validate output format
	validFormats := []string{"text", "json", "csv"}
	isValidFormat := false
	for _, f := range validFormats {
		if format == f {
			isValidFormat = true
			break
		}
	}
	if !isValidFormat {
		return fmt.Errorf("invalid output format: %s (must be one of: %s)", format, strings.Join(validFormats, ", "))
	}

	// Validate page range
	if pages != "" {
		if err := validatePageRange(pages); err != nil {
			return fmt.Errorf("invalid page range: %w", err)
		}
	}

	// Validate recognition height
	if recH < 0 {
		return fmt.Errorf("invalid recognition height: %d (must be positive)", recH)
	}

	// Validate orientation threshold
	if orientThresh < 0 || orientThresh > 1 {
		return fmt.Errorf("invalid orientation threshold: %.2f (must be between 0.0 and 1.0)", orientThresh)
	}

	// Validate textline threshold
	if textlineThresh < 0 || textlineThresh > 1 {
		return fmt.Errorf("invalid textline threshold: %.2f (must be between 0.0 and 1.0)", textlineThresh)
	}

	// Validate rectify mask threshold
	if rectifyMask < 0 || rectifyMask > 1 {
		return fmt.Errorf("invalid rectify mask threshold: %.2f (must be between 0.0 and 1.0)", rectifyMask)
	}

	// Validate rectify height
	if rectifyHeight <= 0 {
		return fmt.Errorf("invalid rectify height: %d (must be positive)", rectifyHeight)
	}

	fmt.Printf("Processing %d PDF(s): %v\n", len(args), args)

	// Build pipeline
	b := pipeline.NewBuilder().WithModelsDir(modelsDir).WithLanguage(lang)
	if detectOrientation {
		b = b.WithOrientation(true)
	}
	if detectTextline {
		b = b.WithTextLineOrientation(true)
	}
	if rectify {
		b = b.WithRectification(true)
	}
	if recH > 0 {
		b = b.WithImageHeight(recH)
	}
	b = b.WithDetectorThresholds(pipeline.DefaultConfig().Detector.DbThresh, float32(detConf))
	if detModel != "" {
		b = b.WithDetectorModelPath(detModel)
	}
	if recModel != "" {
		b = b.WithRecognizerModelPath(recModel)
	}
	if dictCSV != "" {
		b = b.WithDictionaryPaths(strings.Split(dictCSV, ","))
	}
	if dictLangs != "" {
		paths := models.GetDictionaryPathsForLanguages(modelsDir, strings.Split(dictLangs, ","))
		if len(paths) > 0 {
			b = b.WithDictionaryPaths(paths)
		}
	}
	if orientThresh > 0 {
		b = b.WithOrientationThreshold(orientThresh)
	}
	if textlineThresh > 0 {
		b = b.WithTextLineOrientationThreshold(textlineThresh)
	}
	if rectifyModel != "" {
		b = b.WithRectifyModelPath(rectifyModel)
	}
	if rectifyMask > 0 {
		b = b.WithRectifyMaskThreshold(rectifyMask)
	}
	if rectifyHeight > 0 {
		b = b.WithRectifyOutputHeight(rectifyHeight)
	}
	if rectifyDebugDir != "" {
		b = b.WithRectifyDebugDir(rectifyDebugDir)
	}

	pl, err := b.Build()
	if err != nil {
		return fmt.Errorf("failed to build OCR pipeline: %w", err)
	}
	defer func() { _ = pl.Close() }()

	// Extract images per page
	// For each file, produce a DocumentResult aggregating per-page results
	results := make([]*pdf.DocumentResult, 0, len(args))
	for _, file := range args {
		pageImages, err := pdf.ExtractImages(file, pages)
		if err != nil {
			return fmt.Errorf("failed to extract images from %s: %w", file, err)
		}
		doc := &pdf.DocumentResult{Filename: file}
		// Maintain page order by iterating sorted keys (optional small pass)
		// Simpler: iterate map; order not guaranteed but acceptable. For stability,
		// we can gather keys and sort.
		// Collect keys
		keys := make([]int, 0, len(pageImages))
		for k := range pageImages {
			keys = append(keys, k)
		}
		// simple insertion sort
		for i := 1; i < len(keys); i++ {
			v := keys[i]
			j := i - 1
			for j >= 0 && keys[j] > v {
				keys[j+1] = keys[j]
				j--
			}
			keys[j+1] = v
		}

		for _, pageNum := range keys {
			imgs := pageImages[pageNum]
			pageRes := pdf.PageResult{PageNumber: pageNum}
			for i, im := range imgs {
				if im == nil {
					continue
				}
				// Run full pipeline
				ocr, err := pl.ProcessImage(im)
				if err != nil {
					return fmt.Errorf("OCR failed for %s page %d image %d: %w", file, pageNum, i, err)
				}
				// Convert to detection-only ImageResult for compatibility
				ir := pdf.ImageResult{ImageIndex: i, Width: ocr.Width, Height: ocr.Height}
				// Map regions
				ir.Regions = make([]detector.DetectedRegion, 0, len(ocr.Regions))
				var sum float64
				// Sort for a more readable order before extracting text
				pipeline.SortRegionsTopLeft(ocr)
				for _, r := range ocr.Regions {
					// Box in OCRRegionResult is ints X,Y,W,H
					minX := float64(r.Box.X)
					minY := float64(r.Box.Y)
					maxX := float64(r.Box.X + r.Box.W)
					maxY := float64(r.Box.Y + r.Box.H)
					box := utils.NewBox(minX, minY, maxX, maxY)
					poly := make([]utils.Point, len(r.Polygon))
					for pi, pt := range r.Polygon {
						poly[pi] = utils.Point{X: pt.X, Y: pt.Y}
					}
					ir.Regions = append(ir.Regions, detector.DetectedRegion{Polygon: poly, Box: box, Confidence: r.DetConfidence})
					sum += r.DetConfidence
					// Enriched OCR region
					or := pdf.OCRRegion{
						Polygon:       poly,
						DetConfidence: r.DetConfidence,
						Text:          r.Text,
						RecConfidence: r.RecConfidence,
						Language:      r.Language,
					}
					or.Box = struct{ X, Y, W, H int }{X: r.Box.X, Y: r.Box.Y, W: r.Box.W, H: r.Box.H}
					ir.OCRRegions = append(ir.OCRRegions, or)
				}
				if len(ocr.Regions) > 0 {
					ir.Confidence = sum / float64(len(ocr.Regions))
				}
				// Aggregate plain text per image
				if txt, err := pipeline.ToPlainTextImage(ocr); err == nil {
					ir.Text = txt
				}
				pageRes.Images = append(pageRes.Images, ir)
				// Track page dims
				if ocr.Width > pageRes.Width {
					pageRes.Width = ocr.Width
				}
				if ocr.Height > pageRes.Height {
					pageRes.Height = ocr.Height
				}
			}
			doc.Pages = append(doc.Pages, pageRes)
		}
		doc.TotalPages = len(doc.Pages)
		results = append(results, doc)
	}

	return outputResults(cmd, results, format, outputFile)
}

// outputResults formats and outputs the PDF OCR results.
func outputResults(_ *cobra.Command, results []*pdf.DocumentResult, format string, outputFile string) error {
	var output string
	var err error

	switch format {
	case "json":
		output, err = formatJSON(results)
		if err != nil {
			return fmt.Errorf("failed to format output: %w", err)
		}
	case "csv":
		output = formatCSV(results)
	default: // text
		output = formatText(results)
	}

	// Write to file or stdout
	if outputFile != "" {
		err = os.WriteFile(outputFile, []byte(output), 0o600)
		if err != nil {
			return fmt.Errorf("failed to write output file: %w", err)
		}
		fmt.Printf("Results written to %s\n", outputFile)
	} else {
		fmt.Print(output)
	}

	return nil
}

// formatJSON formats results as JSON.
func formatJSON(results []*pdf.DocumentResult) (string, error) {
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// formatText formats results as plain text.
func formatText(results []*pdf.DocumentResult) string {
	var output string

	for _, doc := range results {
		output += fmt.Sprintf("File: %s\n", doc.Filename)
		output += fmt.Sprintf("Total Pages: %d\n", doc.TotalPages)
		output += fmt.Sprintf("Processing Time: %dms\n\n", doc.Processing.TotalTimeMs)

		for _, page := range doc.Pages {
			output += fmt.Sprintf("Page %d (%dx%d):\n", page.PageNumber, page.Width, page.Height)

			for _, img := range page.Images {
				output += fmt.Sprintf("  Image %d (%dx%d): %d region(s), confidence: %.3f\n",
					img.ImageIndex, img.Width, img.Height, len(img.Regions), img.Confidence)

				for i, region := range img.Regions {
					output += fmt.Sprintf("    #%d box=(%d,%d %dx%d) conf=%.3f\n",
						i+1,
						int(region.Box.MinX), int(region.Box.MinY),
						int(region.Box.Width()), int(region.Box.Height()),
						region.Confidence)
				}
			}
			output += "\n"
		}
		output += "---\n\n"
	}

	return output
}

// formatCSV formats results as CSV.
func formatCSV(results []*pdf.DocumentResult) string {
	var records [][]string

	// Header
	records = append(records, []string{
		"File", "Page", "Image", "Region", "X1", "Y1", "X2", "Y2", "Width", "Height", "Confidence",
	})

	// Data rows
	for _, doc := range results {
		for _, page := range doc.Pages {
			for _, img := range page.Images {
				for i, region := range img.Regions {
					records = append(records, []string{
						doc.Filename,
						strconv.Itoa(page.PageNumber),
						strconv.Itoa(img.ImageIndex),
						strconv.Itoa(i + 1),
						strconv.FormatFloat(float64(region.Box.MinX), 'f', 0, 32),
						strconv.FormatFloat(float64(region.Box.MinY), 'f', 0, 32),
						strconv.FormatFloat(float64(region.Box.MaxX), 'f', 0, 32),
						strconv.FormatFloat(float64(region.Box.MaxY), 'f', 0, 32),
						strconv.FormatFloat(float64(region.Box.Width()), 'f', 0, 32),
						strconv.FormatFloat(float64(region.Box.Height()), 'f', 0, 32),
						strconv.FormatFloat(region.Confidence, 'f', 3, 64),
					})
				}
			}
		}
	}

	// Convert to CSV string
	var csvOutput string
	for _, record := range records {
		for i, field := range record {
			if i > 0 {
				csvOutput += ","
			}
			// Quote fields that might contain commas or quotes
			if containsCSVSpecialChar(field) {
				csvOutput += `"` + field + `"`
			} else {
				csvOutput += field
			}
		}
		csvOutput += "\n"
	}

	return csvOutput
}

// containsCSVSpecialChar checks if a field needs CSV quoting.
func containsCSVSpecialChar(field string) bool {
	for _, char := range field {
		if char == ',' || char == '"' || char == '\n' || char == '\r' {
			return true
		}
	}
	return false
}

// validatePageRange validates a page range string.
func validatePageRange(pages string) error {
	// Simple validation for ranges like "1-5", "1,3,5"
	parts := strings.Split(pages, ",")
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if strings.Contains(part, "-") {
			rangeParts := strings.Split(part, "-")
			if len(rangeParts) != 2 {
				return fmt.Errorf("invalid range format: %s", part)
			}
			start, err1 := strconv.Atoi(strings.TrimSpace(rangeParts[0]))
			end, err2 := strconv.Atoi(strings.TrimSpace(rangeParts[1]))
			if err1 != nil || err2 != nil {
				return fmt.Errorf("invalid page numbers in range: %s", part)
			}
			if start > end {
				return fmt.Errorf("invalid page range: start (%d) > end (%d)", start, end)
			}
			if start < 1 || end < 1 {
				return fmt.Errorf("page numbers must be positive: %s", part)
			}
		} else {
			pageNum, err := strconv.Atoi(part)
			if err != nil {
				return fmt.Errorf("invalid page number: %s", part)
			}
			if pageNum < 1 {
				return fmt.Errorf("page number must be positive: %d", pageNum)
			}
		}
	}
	return nil
}
