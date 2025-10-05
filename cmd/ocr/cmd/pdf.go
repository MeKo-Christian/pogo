package cmd

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/MeKo-Tech/pogo/internal/config"
	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/pdf"
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
	pdfCmd.Flags().String("filter-dict", "", "comma-separated filter dictionary paths (restricts output characters, e.g., latin_subset.txt)")
	pdfCmd.Flags().String("filter-dict-langs", "", "comma-separated language codes for filter dictionaries")
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

	// Enhanced PDF processing flags
	pdfCmd.Flags().Bool("enable-vector-text", true, "enable vector text extraction")
	pdfCmd.Flags().Bool("enable-hybrid", true, "enable hybrid processing (vector + OCR)")
	pdfCmd.Flags().Float64("vector-text-quality", 0.7, "minimum vector text quality threshold (0..1)")
	pdfCmd.Flags().Float64("vector-text-coverage", 0.8, "minimum vector text coverage for preference (0..1)")

	// Password-related flags
	pdfCmd.Flags().StringP("password", "p", "", "user password for encrypted PDFs")
	pdfCmd.Flags().String("owner-password", "", "owner password for encrypted PDFs")
	pdfCmd.Flags().Bool("allow-passwords", true, "allow processing of password-protected PDFs")
	pdfCmd.Flags().Bool("prompt-password", false, "prompt for password if PDF is encrypted and no password provided")

	// Multi-scale detection flags (parity with image command)
	pdfCmd.Flags().Bool("det-multiscale", false, "enable multi-scale detection (pyramid)")
	pdfCmd.Flags().Float64Slice("det-scales", []float64{1.0, 0.75, 0.5}, "relative scales for multi-scale detection (e.g., 1.0,0.75,0.5)")
	pdfCmd.Flags().Float64("det-merge-iou", 0.3, "IoU threshold for merging regions across scales")
	pdfCmd.Flags().Bool("det-ms-adaptive", false, "enable adaptive pyramid scaling (auto scales based on image size)")
	pdfCmd.Flags().Int("det-ms-max-levels", 3, "maximum pyramid levels when adaptive is enabled (including 1.0)")
	pdfCmd.Flags().Int("det-ms-min-side", 320, "stop adaptive scaling when min(image side * scale) <= this value")
	pdfCmd.Flags().Bool("det-ms-incremental-merge", true, "incrementally merge detections after each scale to reduce memory")

	// Barcode flags (optional; page-rendered images)
	pdfCmd.Flags().Bool("barcodes", false, "enable barcode detection on pages")
	pdfCmd.Flags().String("barcode-types", "", "comma-separated types to detect (e.g., qr,ean13,upca,code128,pdf417,datamatrix,aztec,ean8,upce,itf,codabar,code39)")
	pdfCmd.Flags().Int("barcode-min-size", 0, "minimum expected barcode size in pixels (hint; affects render DPI)")
	pdfCmd.Flags().Int("barcode-dpi", 150, "target DPI for barcode decoding (page image scaling)")
	pdfCmd.Flags().Int("pdf-workers", 0, "max worker goroutines for page processing (0=NumCPU)")
}

// Ensure pdf help mentions docs for multi-scale.
func init() {
	pdfCmd.SetHelpFunc(func(cmd *cobra.Command, args []string) {
		out := cmd.OutOrStdout()
		if _, err := fmt.Fprintln(out, cmd.Short); err != nil {
			return
		}
		if _, err := fmt.Fprintln(out, "Usage:"); err != nil {
			return
		}
		_, _ = fmt.Fprintln(out, cmd.UseLine())
		_, _ = fmt.Fprintln(out, "Flags:")
		_, _ = fmt.Fprintln(out, cmd.Flags().FlagUsages())
		_, _ = fmt.Fprintln(out, "Docs:")
		_, _ = fmt.Fprintln(out, "  See docs/multiscale.md for multi-scale detection options and tuning.")
	})
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
	filterDictCSV     string
	filterDictLangs   string
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

	// Enhanced PDF processing options
	enableVectorText   bool
	enableHybrid       bool
	vectorTextQuality  float64
	vectorTextCoverage float64

	// Password-related options
	userPassword   string
	ownerPassword  string
	allowPasswords bool
	promptPassword bool

	// Multi-scale detection
	multiScaleEnabled bool
	msScales          []float64
	msMergeIoU        float64
	msAdaptive        bool
	msMaxLevels       int
	msMinSide         int
	msIncremental     bool

	// Barcode (optional; currently captured only)
	barcodeEnabled bool
	barcodeTypes   string
	barcodeMinSize int
	barcodeDPI     int
	pdfWorkers     int
}

// configToPDFConfig maps centralized configuration to pdfConfig.
// CLI flags will override config file values through Viper's precedence system.
func configToPDFConfig(centralCfg *config.Config, cmd *cobra.Command) (*pdfConfig, error) {
	cfg := &pdfConfig{}

	// Helper functions to reduce cyclomatic complexity
	setFloat64WithFlag := func(configValue float64, flagName string, target *float64) {
		*target = configValue
		if cmd.Flags().Changed(flagName) {
			*target, _ = cmd.Flags().GetFloat64(flagName)
		}
	}

	setStringWithFlag := func(configValue, flagName string, target *string) {
		*target = configValue
		if cmd.Flags().Changed(flagName) {
			*target, _ = cmd.Flags().GetString(flagName)
		}
	}

	setIntWithFlag := func(configValue int, flagName string, target *int) {
		*target = configValue
		if cmd.Flags().Changed(flagName) {
			*target, _ = cmd.Flags().GetInt(flagName)
		}
	}

	setBoolWithFlag := func(configValue bool, flagName string, target *bool) {
		*target = configValue
		if cmd.Flags().Changed(flagName) {
			*target, _ = cmd.Flags().GetBool(flagName)
		}
	}

	// Core OCR settings
	setFloat64WithFlag(float64(centralCfg.Pipeline.Detector.DbBoxThresh), "confidence", &cfg.detConf)
	setStringWithFlag(centralCfg.ModelsDir, "", &cfg.modelsDir)
	setStringWithFlag(centralCfg.Output.Format, "format", &cfg.format)
	setStringWithFlag(centralCfg.Output.File, "output", &cfg.outputFile)
	setStringWithFlag(centralCfg.Pipeline.Recognizer.Language, "language", &cfg.lang)
	setStringWithFlag(centralCfg.Pipeline.Detector.ModelPath, "det-model", &cfg.detModel)
	setStringWithFlag(centralCfg.Pipeline.Recognizer.ModelPath, "rec-model", &cfg.recModel)
	setStringWithFlag(centralCfg.Pipeline.Recognizer.DictPath, "dict", &cfg.dictCSV)
	setStringWithFlag(centralCfg.Pipeline.Recognizer.DictLangs, "dict-langs", &cfg.dictLangs)
	setStringWithFlag(centralCfg.Pipeline.Recognizer.FilterDictPath, "filter-dict", &cfg.filterDictCSV)
	setStringWithFlag(centralCfg.Pipeline.Recognizer.FilterDictLangs, "filter-dict-langs", &cfg.filterDictLangs)
	setIntWithFlag(centralCfg.Pipeline.Recognizer.ImageHeight, "rec-height", &cfg.recH)

	// Orientation settings
	setBoolWithFlag(centralCfg.Features.OrientationEnabled, "detect-orientation", &cfg.detectOrientation)
	setFloat64WithFlag(centralCfg.Features.OrientationThreshold, "orientation-threshold", &cfg.orientThresh)
	setBoolWithFlag(centralCfg.Features.TextlineEnabled, "detect-textline", &cfg.detectTextline)
	setFloat64WithFlag(centralCfg.Features.TextlineThreshold, "textline-threshold", &cfg.textlineThresh)

	// Rectification settings
	setBoolWithFlag(centralCfg.Features.RectificationEnabled, "rectify", &cfg.rectify)
	setStringWithFlag(centralCfg.Features.RectificationModelPath, "rectify-model", &cfg.rectifyModel)
	setFloat64WithFlag(centralCfg.Features.RectificationThreshold, "rectify-mask-threshold", &cfg.rectifyMask)
	setIntWithFlag(centralCfg.Features.RectificationHeight, "rectify-height", &cfg.rectifyHeight)
	setStringWithFlag(centralCfg.Features.RectificationDebugDir, "rectify-debug-dir", &cfg.rectifyDebugDir)

	// PDF-specific flags (these don't have config file equivalents)
	cfg.pages, _ = cmd.Flags().GetString("pages")

	// Enhanced PDF processing flags
	cfg.enableVectorText, _ = cmd.Flags().GetBool("enable-vector-text")
	cfg.enableHybrid, _ = cmd.Flags().GetBool("enable-hybrid")
	cfg.vectorTextQuality, _ = cmd.Flags().GetFloat64("vector-text-quality")
	cfg.vectorTextCoverage, _ = cmd.Flags().GetFloat64("vector-text-coverage")

	// Password-related flags
	cfg.userPassword, _ = cmd.Flags().GetString("password")
	cfg.ownerPassword, _ = cmd.Flags().GetString("owner-password")
	cfg.allowPasswords, _ = cmd.Flags().GetBool("allow-passwords")
	cfg.promptPassword, _ = cmd.Flags().GetBool("prompt-password")

	// Barcode flags (captured for pipeline integration)
	cfg.barcodeEnabled, _ = cmd.Flags().GetBool("barcodes")
	cfg.barcodeTypes, _ = cmd.Flags().GetString("barcode-types")
	cfg.barcodeMinSize, _ = cmd.Flags().GetInt("barcode-min-size")
	cfg.barcodeDPI, _ = cmd.Flags().GetInt("barcode-dpi")
	// Concurrency controls
	// (Uses central config for defaults if present in future; currently CLI flag only)
	// Store into unused field via return path by extending buildEnhancedPDFProcessor
	// Will be passed to ProcessorConfig.MaxWorkers
	cfg.pdfWorkers, _ = cmd.Flags().GetInt("pdf-workers")

	// Multi-scale detection defaults from central config
	cfg.multiScaleEnabled = centralCfg.Pipeline.Detector.MultiScale.Enabled
	cfg.msScales = centralCfg.Pipeline.Detector.MultiScale.Scales
	cfg.msMergeIoU = centralCfg.Pipeline.Detector.MultiScale.MergeIoU
	// Override with CLI flags if provided
	if cmd.Flags().Changed("det-multiscale") {
		cfg.multiScaleEnabled, _ = cmd.Flags().GetBool("det-multiscale")
	}
	if cmd.Flags().Changed("det-scales") {
		cfg.msScales, _ = cmd.Flags().GetFloat64Slice("det-scales")
	}
	if cmd.Flags().Changed("det-merge-iou") {
		cfg.msMergeIoU, _ = cmd.Flags().GetFloat64("det-merge-iou")
	}
	if cmd.Flags().Changed("det-ms-adaptive") {
		cfg.msAdaptive, _ = cmd.Flags().GetBool("det-ms-adaptive")
	}
	if cmd.Flags().Changed("det-ms-max-levels") {
		cfg.msMaxLevels, _ = cmd.Flags().GetInt("det-ms-max-levels")
	}
	if cmd.Flags().Changed("det-ms-min-side") {
		cfg.msMinSide, _ = cmd.Flags().GetInt("det-ms-min-side")
	}
	if cmd.Flags().Changed("det-ms-incremental-merge") {
		cfg.msIncremental, _ = cmd.Flags().GetBool("det-ms-incremental-merge")
	}

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
		validateMultiScale,
	}

	for _, validator := range validators {
		if err := validator(cfg); err != nil {
			return err
		}
	}

	return nil
}

// validateMultiScale validates multi-scale parameters when enabled.
func validateMultiScale(cfg *pdfConfig) error {
	if !cfg.multiScaleEnabled {
		return nil
	}
	if cfg.msMergeIoU < 0 || cfg.msMergeIoU > 1 {
		return fmt.Errorf("invalid det-merge-iou: %.2f (must be between 0.0 and 1.0)", cfg.msMergeIoU)
	}
	for _, s := range cfg.msScales {
		if s <= 0 {
			return fmt.Errorf("invalid det-scales value: %.3f (must be > 0)", s)
		}
	}
	if cfg.msMaxLevels < 0 {
		return fmt.Errorf("invalid det-ms-max-levels: %d (must be >= 0)", cfg.msMaxLevels)
	}
	if cfg.msMinSide < 0 {
		return fmt.Errorf("invalid det-ms-min-side: %d (must be >= 0)", cfg.msMinSide)
	}
	// msIncremental is boolean; no further validation
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

// processPDFs handles the main PDF processing logic.
func processPDFs(cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		return errors.New("no input files provided")
	}

	// Get configuration from centralized system (includes CLI flags, config file, env vars, and defaults)
	centralCfg := GetConfig()

	// Map to PDF configuration
	cfg, err := configToPDFConfig(centralCfg, cmd)
	if err != nil {
		return err
	}

	fmt.Printf("Processing %d PDF(s): %v\n", len(args), args)

	// Build enhanced PDF processor
	processor, err := buildEnhancedPDFProcessor(cfg)
	if err != nil {
		return fmt.Errorf("failed to build enhanced PDF processor: %w", err)
	}
	defer func() { _ = processor.Close() }()

	// Setup password credentials if provided
	if cfg.userPassword != "" || cfg.ownerPassword != "" {
		creds := &pdf.PasswordCredentials{
			UserPassword:  cfg.userPassword,
			OwnerPassword: cfg.ownerPassword,
		}
		processor.SetPasswordCredentials(creds)
	}

	// Process each PDF file
	results := make([]*pdf.DocumentResult, 0, len(args))
	for _, file := range args {
		creds := &pdf.PasswordCredentials{
			UserPassword:  cfg.userPassword,
			OwnerPassword: cfg.ownerPassword,
		}
		doc, err := processor.ProcessFileWithCredentials(file, cfg.pages, creds)
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

	// Data rows (text regions)
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
				// Append barcode rows for this image (second section header per image)
				if len(img.Barcodes) > 0 {
					// Blank line and header for barcodes
					records = append(records, []string{})
					records = append(records, []string{"File", "Page", "Image", "Barcode", "Type", "Value", "Confidence", "Rotation", "X", "Y", "Width", "Height"})
					for j, b := range img.Barcodes {
						records = append(records, []string{
							doc.Filename,
							strconv.Itoa(page.PageNumber),
							strconv.Itoa(img.ImageIndex),
							strconv.Itoa(j + 1),
							b.Type,
							b.Value,
							strconv.FormatFloat(b.Confidence, 'f', 3, 64),
							strconv.FormatFloat(b.Rotation, 'f', 1, 64),
							strconv.Itoa(b.Box.X),
							strconv.Itoa(b.Box.Y),
							strconv.Itoa(b.Box.W),
							strconv.Itoa(b.Box.H),
						})
					}
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

// buildEnhancedPDFProcessor creates an enhanced PDF processor with the given configuration.
func buildEnhancedPDFProcessor(cfg *pdfConfig) (*pdf.Processor, error) {
	// Create detector configuration
	detectorConfig := detector.Config{
		ModelPath:    cfg.detModel,
		DbThresh:     0.3, // Default DB threshold
		DbBoxThresh:  float32(cfg.detConf),
		MaxImageSize: 960, // Default max image size
		NumThreads:   0,   // Auto-detect threads
	}

	// Multi-scale configuration
	if cfg.multiScaleEnabled {
		detectorConfig.MultiScale.Enabled = true
		if len(cfg.msScales) > 0 {
			detectorConfig.MultiScale.Scales = cfg.msScales
		}
		if cfg.msMergeIoU > 0 {
			detectorConfig.MultiScale.MergeIoU = cfg.msMergeIoU
		}
		detectorConfig.MultiScale.Adaptive = cfg.msAdaptive
		if cfg.msMaxLevels > 0 {
			detectorConfig.MultiScale.MaxLevels = cfg.msMaxLevels
		}
		if cfg.msMinSide > 0 {
			detectorConfig.MultiScale.MinSide = cfg.msMinSide
		}
		detectorConfig.MultiScale.IncrementalMerge = cfg.msIncremental
	}

	// If no specific model path, use models dir
	if detectorConfig.ModelPath == "" {
		detectorConfig.ModelPath = cfg.modelsDir
	}

	// Create a detector
	det, err := detector.NewDetector(detectorConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create detector: %w", err)
	}

	// Create processor configuration
	processorConfig := &pdf.ProcessorConfig{
		EnableVectorText:    cfg.enableVectorText,
		EnableHybrid:        cfg.enableHybrid,
		VectorTextQuality:   cfg.vectorTextQuality,
		VectorTextCoverage:  cfg.vectorTextCoverage,
		AllowPasswords:      cfg.allowPasswords,
		AllowPasswordPrompt: cfg.promptPassword,
		EnableBarcodes:      cfg.barcodeEnabled,
		BarcodeTypes:        cfg.barcodeTypes,
		BarcodeMinSize:      cfg.barcodeMinSize,
		BarcodeTargetDPI: func() int {
			if cfg.barcodeDPI > 0 {
				return cfg.barcodeDPI
			}
			return 150
		}(),
		MaxWorkers: cfg.pdfWorkers,
	}

	// Create enhanced processor
	processor := pdf.NewProcessorWithConfig(det, processorConfig)

	return processor, nil
}
