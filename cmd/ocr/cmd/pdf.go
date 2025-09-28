package cmd

import (
	"encoding/json"
	"errors"
	"fmt"
	"image"
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

// pdfConfig holds all the configuration for PDF processing.
type pdfConfig struct {
	detConf           float64
	modelsDir         string
	pages             string
	format            string
	outputFile        string
	lang              string
	detModel          string
	recModel          string
	dictCSV           string
	dictLangs         string
	recH              int
	detectOrientation bool
	orientThresh      float64
	detectTextline    bool
	textlineThresh    float64
	rectify           bool
	rectifyModel      string
	rectifyMask       float64
	rectifyHeight     int
	rectifyDebugDir   string
}

// parsePDFFlags parses and validates all command line flags for PDF processing.
func parsePDFFlags(cmd *cobra.Command) (*pdfConfig, error) {
	cfg := &pdfConfig{}

	// Parse flags
	cfg.detConf, _ = cmd.Flags().GetFloat64("confidence")
	cfg.modelsDir, _ = cmd.InheritedFlags().GetString("models-dir")
	cfg.pages, _ = cmd.Flags().GetString("pages")
	cfg.format, _ = cmd.Flags().GetString("format")
	cfg.outputFile, _ = cmd.Flags().GetString("output")
	cfg.lang, _ = cmd.Flags().GetString("language")
	cfg.detModel, _ = cmd.Flags().GetString("det-model")
	cfg.recModel, _ = cmd.Flags().GetString("rec-model")
	cfg.dictCSV, _ = cmd.Flags().GetString("dict")
	cfg.dictLangs, _ = cmd.Flags().GetString("dict-langs")
	cfg.recH, _ = cmd.Flags().GetInt("rec-height")
	cfg.detectOrientation, _ = cmd.Flags().GetBool("detect-orientation")
	cfg.orientThresh, _ = cmd.Flags().GetFloat64("orientation-threshold")
	cfg.detectTextline, _ = cmd.Flags().GetBool("detect-textline")
	cfg.textlineThresh, _ = cmd.Flags().GetFloat64("textline-threshold")
	cfg.rectify, _ = cmd.Flags().GetBool("rectify")
	cfg.rectifyModel, _ = cmd.Flags().GetString("rectify-model")
	cfg.rectifyMask, _ = cmd.Flags().GetFloat64("rectify-mask-threshold")
	cfg.rectifyHeight, _ = cmd.Flags().GetInt("rectify-height")
	cfg.rectifyDebugDir, _ = cmd.Flags().GetString("rectify-debug-dir")

	// Validate parameters
	if err := validatePDFConfig(cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}

// validatePDFConfig validates the PDF configuration parameters.
func validatePDFConfig(cfg *pdfConfig) error {
	validators := []func(*pdfConfig) error{
		validateConfidenceThreshold,
		validateOutputFormat,
		validatePageRangeConfig,
		validateRecognitionHeight,
		validateOrientationThreshold,
		validateTextlineThreshold,
		validateRectifyThresholds,
	}

	for _, validator := range validators {
		if err := validator(cfg); err != nil {
			return err
		}
	}

	return nil
}

// validateConfidenceThreshold validates the confidence threshold.
func validateConfidenceThreshold(cfg *pdfConfig) error {
	if cfg.detConf < 0 || cfg.detConf > 1 {
		return fmt.Errorf("invalid confidence threshold: %.2f (must be between 0.0 and 1.0)", cfg.detConf)
	}
	return nil
}

// validateOutputFormat validates the output format.
func validateOutputFormat(cfg *pdfConfig) error {
	validFormats := []string{"text", "json", "csv"}
	for _, f := range validFormats {
		if cfg.format == f {
			return nil
		}
	}
	return fmt.Errorf("invalid output format: %s (must be one of: %s)", cfg.format, strings.Join(validFormats, ", "))
}

// validatePageRangeConfig validates the page range configuration.
func validatePageRangeConfig(cfg *pdfConfig) error {
	if cfg.pages != "" {
		if err := validatePageRange(cfg.pages); err != nil {
			return fmt.Errorf("invalid page range: %w", err)
		}
	}
	return nil
}

// validateRecognitionHeight validates the recognition height.
func validateRecognitionHeight(cfg *pdfConfig) error {
	if cfg.recH < 0 {
		return fmt.Errorf("invalid recognition height: %d (must be positive)", cfg.recH)
	}
	return nil
}

// validateOrientationThreshold validates the orientation threshold.
func validateOrientationThreshold(cfg *pdfConfig) error {
	if cfg.orientThresh < 0 || cfg.orientThresh > 1 {
		return fmt.Errorf("invalid orientation threshold: %.2f (must be between 0.0 and 1.0)", cfg.orientThresh)
	}
	return nil
}

// validateTextlineThreshold validates the textline threshold.
func validateTextlineThreshold(cfg *pdfConfig) error {
	if cfg.textlineThresh < 0 || cfg.textlineThresh > 1 {
		return fmt.Errorf("invalid textline threshold: %.2f (must be between 0.0 and 1.0)", cfg.textlineThresh)
	}
	return nil
}

// validateRectifyThresholds validates rectify-related thresholds.
func validateRectifyThresholds(cfg *pdfConfig) error {
	if cfg.rectifyMask < 0 || cfg.rectifyMask > 1 {
		return fmt.Errorf("invalid rectify mask threshold: %.2f (must be between 0.0 and 1.0)", cfg.rectifyMask)
	}
	if cfg.rectifyHeight <= 0 {
		return fmt.Errorf("invalid rectify height: %d (must be positive)", cfg.rectifyHeight)
	}
	return nil
}

// buildPDFPipeline creates the OCR pipeline with the given configuration.
func buildPDFPipeline(cfg *pdfConfig) (*pipeline.Pipeline, error) {
	b := pipeline.NewBuilder().WithModelsDir(cfg.modelsDir).WithLanguage(cfg.lang)

	configureDetection(b, cfg)
	configureOrientation(b, cfg)
	configureRectification(b, cfg)
	configureModels(b, cfg)
	configureThresholds(b, cfg)

	return b.Build()
}

// configureDetection configures detection-related settings.
func configureDetection(b *pipeline.Builder, cfg *pdfConfig) {
	b.WithDetectorThresholds(pipeline.DefaultConfig().Detector.DbThresh, float32(cfg.detConf))
	if cfg.detModel != "" {
		b.WithDetectorModelPath(cfg.detModel)
	}
}

// configureOrientation configures orientation detection settings.
func configureOrientation(b *pipeline.Builder, cfg *pdfConfig) {
	if cfg.detectOrientation {
		b.WithOrientation(true)
	}
	if cfg.detectTextline {
		b.WithTextLineOrientation(true)
	}
	if cfg.orientThresh > 0 {
		b.WithOrientationThreshold(cfg.orientThresh)
	}
	if cfg.textlineThresh > 0 {
		b.WithTextLineOrientationThreshold(cfg.textlineThresh)
	}
}

// configureRectification configures rectification settings.
func configureRectification(b *pipeline.Builder, cfg *pdfConfig) {
	if cfg.rectify {
		b.WithRectification(true)
	}
	if cfg.rectifyModel != "" {
		b.WithRectifyModelPath(cfg.rectifyModel)
	}
	if cfg.rectifyMask > 0 {
		b.WithRectifyMaskThreshold(cfg.rectifyMask)
	}
	if cfg.rectifyHeight > 0 {
		b.WithRectifyOutputHeight(cfg.rectifyHeight)
	}
	if cfg.rectifyDebugDir != "" {
		b.WithRectifyDebugDir(cfg.rectifyDebugDir)
	}
}

// configureModels configures model paths and dictionaries.
func configureModels(b *pipeline.Builder, cfg *pdfConfig) {
	if cfg.recModel != "" {
		b.WithRecognizerModelPath(cfg.recModel)
	}
	if cfg.dictCSV != "" {
		b.WithDictionaryPaths(strings.Split(cfg.dictCSV, ","))
	}
	if cfg.dictLangs != "" {
		paths := models.GetDictionaryPathsForLanguages(cfg.modelsDir, strings.Split(cfg.dictLangs, ","))
		if len(paths) > 0 {
			b.WithDictionaryPaths(paths)
		}
	}
}

// configureThresholds configures image height and other thresholds.
func configureThresholds(b *pipeline.Builder, cfg *pdfConfig) {
	if cfg.recH > 0 {
		b.WithImageHeight(cfg.recH)
	}
}

// processPDFFile processes a single PDF file and returns the document result.
func processPDFFile(file string, pages string, pl *pipeline.Pipeline) (*pdf.DocumentResult, error) {
	pageImages, err := pdf.ExtractImages(file, pages)
	if err != nil {
		return nil, fmt.Errorf("failed to extract images from %s: %w", file, err)
	}

	doc := &pdf.DocumentResult{Filename: file}

	// Sort page numbers for consistent ordering
	pageNums := make([]int, 0, len(pageImages))
	for pageNum := range pageImages {
		pageNums = append(pageNums, pageNum)
	}
	sortPages(pageNums)

	for _, pageNum := range pageNums {
		pageRes, err := processPDFPage(pageNum, pageImages[pageNum], pl)
		if err != nil {
			return nil, fmt.Errorf("failed to process page %d of %s: %w", pageNum, file, err)
		}
		doc.Pages = append(doc.Pages, *pageRes)
	}

	doc.TotalPages = len(doc.Pages)
	return doc, nil
}

// sortPages performs a simple insertion sort on page numbers.
func sortPages(keys []int) {
	for i := 1; i < len(keys); i++ {
		v := keys[i]
		j := i - 1
		for j >= 0 && keys[j] > v {
			keys[j+1] = keys[j]
			j--
		}
		keys[j+1] = v
	}
}

// processPDFPage processes all images from a single PDF page.
func processPDFPage(pageNum int, images []image.Image, pl *pipeline.Pipeline) (*pdf.PageResult, error) {
	pageRes := pdf.PageResult{PageNumber: pageNum}

	for i, img := range images {
		if img == nil {
			continue
		}

		// Run full pipeline
		ocr, err := pl.ProcessImage(img)
		if err != nil {
			return nil, fmt.Errorf("OCR failed for page %d image %d: %w", pageNum, i, err)
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

	return &pageRes, nil
}

// processPDFs handles the main PDF processing logic.
func processPDFs(cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		return errors.New("no input files provided")
	}

	// Parse and validate flags
	cfg, err := parsePDFFlags(cmd)
	if err != nil {
		return err
	}

	fmt.Printf("Processing %d PDF(s): %v\n", len(args), args)

	// Build pipeline
	pl, err := buildPDFPipeline(cfg)
	if err != nil {
		return fmt.Errorf("failed to build OCR pipeline: %w", err)
	}
	defer func() { _ = pl.Close() }()

	// Process each PDF file
	results := make([]*pdf.DocumentResult, 0, len(args))
	for _, file := range args {
		doc, err := processPDFFile(file, cfg.pages, pl)
		if err != nil {
			return err
		}
		results = append(results, doc)
	}

	return outputResults(cmd, results, cfg.format, cfg.outputFile)
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
		if err := validatePagePart(part); err != nil {
			return err
		}
	}
	return nil
}

// validatePagePart validates a single page part (either a single number or a range).
func validatePagePart(part string) error {
	if strings.Contains(part, "-") {
		return validatePageRangePart(part)
	}
	return validateSinglePage(part)
}

// validatePageRangePart validates a range part like "1-5".
func validatePageRangePart(part string) error {
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
	return nil
}

// validateSinglePage validates a single page number.
func validateSinglePage(part string) error {
	pageNum, err := strconv.Atoi(part)
	if err != nil {
		return fmt.Errorf("invalid page number: %s", part)
	}
	if pageNum < 1 {
		return fmt.Errorf("page number must be positive: %d", pageNum)
	}
	return nil
}
