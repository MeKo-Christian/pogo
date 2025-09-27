package cmd

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"image/color"
	"image/png"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/spf13/cobra"
)

// batchCmd represents the batch command for parallel image processing.
var batchCmd = &cobra.Command{
	Use:   "batch [files...]",
	Short: "Process multiple images in parallel for OCR text detection and recognition",
	Long: `Process multiple image files in parallel to extract text using OCR.
This command is optimized for processing large numbers of images efficiently using
parallel workers and resource management.

Supported formats: JPEG, PNG, BMP, TIFF

Examples:
  pogo batch *.jpg *.png
  pogo batch images/ --recursive --workers 8
  pogo batch file1.jpg file2.png --format json --output results.json
  pogo batch images/ --progress --memory-limit 2GB`,
	Args:         cobra.MinimumNArgs(1),
	SilenceUsage: true,
	RunE:         runBatchCommand,
}

func init() {
	rootCmd.AddCommand(batchCmd)

	// Core OCR flags (reuse from image command)
	batchCmd.Flags().Float64("confidence", 0.0, "minimum detection confidence threshold (0.0-1.0)")
	batchCmd.Flags().String("det-model", "", "path to detection model (overrides default)")
	batchCmd.Flags().String("rec-model", "", "path to recognition model (overrides default)")
	batchCmd.Flags().String("language", "", "recognition language for post-processing")
	batchCmd.Flags().String("dict", "", "comma-separated dictionary file paths")
	batchCmd.Flags().String("dict-langs", "", "comma-separated language codes for dictionaries")
	batchCmd.Flags().Int("rec-height", 0, "recognition image height (default: model default)")
	batchCmd.Flags().Float64("min-rec-conf", 0.0, "minimum recognition confidence threshold")

	// Orientation flags
	batchCmd.Flags().Bool("detect-orientation", false, "enable document orientation detection")
	batchCmd.Flags().Float64("orientation-threshold", 0.0, "orientation confidence threshold")
	batchCmd.Flags().Bool("detect-textline", false, "enable text line orientation detection")
	batchCmd.Flags().Float64("textline-threshold", 0.0, "text line orientation confidence threshold")

	// Output flags
	batchCmd.Flags().StringP("format", "f", "text", "output format: text, json, csv")
	batchCmd.Flags().StringP("output", "o", "", "output file (default: stdout)")
	batchCmd.Flags().String("overlay-dir", "", "directory to save overlay images")

	// Rectification flags (experimental)
	batchCmd.Flags().Bool("rectify", false, "enable document rectification (experimental)")
	batchCmd.Flags().String("rectify-model", models.GetLayoutModelPath("", models.LayoutUVDoc), "override rectification model path")
	batchCmd.Flags().Float64("rectify-mask-threshold", 0.5, "rectification mask threshold (0..1)")
	batchCmd.Flags().Int("rectify-height", 1024, "rectified page output height (advisory)")
	batchCmd.Flags().String("rectify-debug-dir", "", "directory to write rectification debug images (mask, overlay, compare)")

	// Parallel processing flags
	batchCmd.Flags().IntP("workers", "w", 0, fmt.Sprintf("number of parallel workers (default: %d)", runtime.NumCPU()))
	batchCmd.Flags().Int("batch-size", 0, "batch size for micro-batching (0 = no batching)")
	batchCmd.Flags().String("memory-limit", "", "memory limit (e.g., 1GB, 512MB)")
	batchCmd.Flags().Int("max-goroutines", 0, "maximum concurrent goroutines")
	batchCmd.Flags().Float64("memory-threshold", 0.8, "memory pressure threshold (0.0-1.0)")

	// File discovery flags
	batchCmd.Flags().BoolP("recursive", "r", false, "recursively scan directories")
	batchCmd.Flags().StringSlice("include", []string{"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"}, "file patterns to include")
	batchCmd.Flags().StringSlice("exclude", []string{}, "file patterns to exclude")

	// Progress and monitoring flags
	batchCmd.Flags().Bool("progress", false, "show progress bar")
	batchCmd.Flags().Bool("quiet", false, "suppress progress output")
	batchCmd.Flags().Bool("stats", false, "show processing statistics")
	batchCmd.Flags().Duration("progress-interval", 500*time.Millisecond, "progress update interval")

	// Resource management flags
	batchCmd.Flags().Bool("adaptive-scaling", false, "enable adaptive worker scaling")
	batchCmd.Flags().Bool("backpressure", true, "enable backpressure when resources are constrained")
}

func runBatchCommand(cmd *cobra.Command, args []string) error {
	// Parse flags
	confFlag, _ := cmd.Flags().GetFloat64("confidence")
	modelsDir, _ := cmd.InheritedFlags().GetString("models-dir")
	detModel, _ := cmd.Flags().GetString("det-model")
	recModel, _ := cmd.Flags().GetString("rec-model")
	lang, _ := cmd.Flags().GetString("language")
	dictCSV, _ := cmd.Flags().GetString("dict")
	dictLangs, _ := cmd.Flags().GetString("dict-langs")
	recH, _ := cmd.Flags().GetInt("rec-height")
	minRecConf, _ := cmd.Flags().GetFloat64("min-rec-conf")
	overlayDir, _ := cmd.Flags().GetString("overlay-dir")
	format, _ := cmd.Flags().GetString("format")
	outputFile, _ := cmd.Flags().GetString("output")

	// Rectification flags
	rectify, _ := cmd.Flags().GetBool("rectify")
	rectifyModel, _ := cmd.Flags().GetString("rectify-model")
	rectifyMask, _ := cmd.Flags().GetFloat64("rectify-mask-threshold")
	rectifyHeight, _ := cmd.Flags().GetInt("rectify-height")
	rectifyDebugDir, _ := cmd.Flags().GetString("rectify-debug-dir")

	// Orientation flags
	detectOrientation, _ := cmd.Flags().GetBool("detect-orientation")
	orientThresh, _ := cmd.Flags().GetFloat64("orientation-threshold")
	detectTextline, _ := cmd.Flags().GetBool("detect-textline")
	textlineThresh, _ := cmd.Flags().GetFloat64("textline-threshold")

	// Parallel processing flags
	workers, _ := cmd.Flags().GetInt("workers")
	batchSize, _ := cmd.Flags().GetInt("batch-size")
	memoryLimitStr, _ := cmd.Flags().GetString("memory-limit")
	maxGoroutines, _ := cmd.Flags().GetInt("max-goroutines")
	memoryThreshold, _ := cmd.Flags().GetFloat64("memory-threshold")

	// File discovery flags
	recursive, _ := cmd.Flags().GetBool("recursive")
	includePatterns, _ := cmd.Flags().GetStringSlice("include")
	excludePatterns, _ := cmd.Flags().GetStringSlice("exclude")

	// Progress flags
	showProgress, _ := cmd.Flags().GetBool("progress")
	quiet, _ := cmd.Flags().GetBool("quiet")
	showStats, _ := cmd.Flags().GetBool("stats")
	progressInterval, _ := cmd.Flags().GetDuration("progress-interval")

	// Resource management flags
	adaptiveScaling, _ := cmd.Flags().GetBool("adaptive-scaling")
	backpressure, _ := cmd.Flags().GetBool("backpressure")

	// Parse memory limit
	var memoryLimitBytes uint64
	if memoryLimitStr != "" {
		limit, err := parseMemoryLimit(memoryLimitStr)
		if err != nil {
			return fmt.Errorf("invalid memory limit: %w", err)
		}
		memoryLimitBytes = limit
	}

	// Discover image files
	imageFiles, err := discoverImageFiles(args, recursive, includePatterns, excludePatterns)
	if err != nil {
		return fmt.Errorf("failed to discover image files: %w", err)
	}

	if len(imageFiles) == 0 {
		return errors.New("no image files found")
	}

	if !quiet {
		_, _ = fmt.Fprintf(cmd.OutOrStdout(), "Found %d image files to process\n", len(imageFiles))
	}

	// Load images
	images := make([]string, len(imageFiles))
	copy(images, imageFiles)

	// Set up progress callback
	var progressCallback pipeline.ProgressCallback
	if showProgress && !quiet {
		progressCallback = pipeline.NewConsoleProgressCallback(
			cmd.OutOrStdout(),
			"Processing: ",
		).WithUpdateInterval(progressInterval)
	}

	// Build OCR pipeline
	b := pipeline.NewBuilder().
		WithModelsDir(modelsDir).
		WithLanguage(lang).
		WithParallelWorkers(workers).
		WithBatchSize(batchSize).
		WithMemoryLimit(memoryLimitBytes).
		WithMaxGoroutines(maxGoroutines).
		WithResourceThreshold(memoryThreshold).
		WithAdaptiveScaling(adaptiveScaling).
		WithBackpressure(backpressure).
		WithProgressCallback(progressCallback)

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
	b = b.WithDetectorThresholds(pipeline.DefaultConfig().Detector.DbThresh, float32(confFlag))
	if detModel != "" {
		b = b.WithDetectorModelPath(detModel)
	}
	if recModel != "" {
		b = b.WithRecognizerModelPath(recModel)
	}
	if dictCSV != "" {
		parts := strings.Split(dictCSV, ",")
		b = b.WithDictionaryPaths(parts)
	}
	if dictLangs != "" {
		langs := strings.Split(dictLangs, ",")
		paths := models.GetDictionaryPathsForLanguages(modelsDir, langs)
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
	defer func() {
		if err := pl.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "Error closing pipeline: %v\n", err)
		}
	}()

	// Process images in parallel
	startTime := time.Now()
	results, err := processImagesParallel(pl, images, b.Config(), confFlag, minRecConf, overlayDir)
	duration := time.Since(startTime)

	if err != nil {
		return fmt.Errorf("batch processing failed: %w", err)
	}

	// Generate output
	output, err := formatBatchResults(results, images, format)
	if err != nil {
		return fmt.Errorf("failed to format results: %w", err)
	}

	// Write output
	if outputFile != "" {
		if err := os.WriteFile(outputFile, []byte(output), 0o600); err != nil {
			return fmt.Errorf("failed to write output file: %w", err)
		}
		if !quiet {
			_, _ = fmt.Fprintf(cmd.OutOrStdout(), "Results written to %s\n", outputFile)
		}
	} else {
		_, _ = fmt.Fprint(cmd.OutOrStdout(), output)
	}

	// Show statistics
	if showStats && !quiet {
		stats := pipeline.CalculateParallelStats(nil, results, duration, workers)
		_, _ = fmt.Fprintf(cmd.OutOrStdout(), "\nProcessing Statistics:\n")
		_, _ = fmt.Fprintf(cmd.OutOrStdout(), "  Total images: %d\n", stats.TotalImages)
		_, _ = fmt.Fprintf(cmd.OutOrStdout(), "  Processed: %d\n", stats.ProcessedImages)
		_, _ = fmt.Fprintf(cmd.OutOrStdout(), "  Failed: %d\n", stats.FailedImages)
		_, _ = fmt.Fprintf(cmd.OutOrStdout(), "  Workers: %d\n", stats.WorkerCount)
		_, _ = fmt.Fprintf(cmd.OutOrStdout(), "  Duration: %v\n", stats.TotalDuration.Round(time.Millisecond))
		_, _ = fmt.Fprintf(cmd.OutOrStdout(), "  Avg per image: %v\n", stats.AveragePerImage.Round(time.Millisecond))
		_, _ = fmt.Fprintf(cmd.OutOrStdout(), "  Throughput: %.1f images/sec\n", stats.ThroughputPerSec)
	}

	return nil
}

// processImagesParallel loads and processes images in parallel.
func processImagesParallel(pl *pipeline.Pipeline, imagePaths []string, config pipeline.Config, confFlag, minRecConf float64, overlayDir string) ([]*pipeline.OCRImageResult, error) {
	// Load images
	images := make([]string, len(imagePaths))
	copy(images, imagePaths)

	// Process paths to actual images
	imageResults := make([]*pipeline.OCRImageResult, len(images))
	cons := utils.DefaultImageConstraints()

	for i, pth := range images {
		if !utils.IsSupportedImage(pth) {
			return nil, fmt.Errorf("unsupported image format: %s", pth)
		}

		img, meta, err := utils.LoadImage(pth)
		if err != nil {
			return nil, fmt.Errorf("failed to load %s: %w", pth, err)
		}

		if err := utils.ValidateImageConstraints(img, cons); err != nil {
			slog.Warn("image does not meet constraints, skipping", "file", pth, "error", err)
		}

		res, err := pl.ProcessImage(img)
		if err != nil {
			return nil, fmt.Errorf("OCR failed for %s: %w", pth, err)
		}

		// Apply confidence filters
		if confFlag > 0 {
			filtered := make([]pipeline.OCRRegionResult, 0, len(res.Regions))
			var sum float64
			for _, r := range res.Regions {
				if r.DetConfidence >= confFlag {
					filtered = append(filtered, r)
					sum += r.DetConfidence
				}
			}
			res.Regions = filtered
			if len(filtered) > 0 {
				res.AvgDetConf = sum / float64(len(filtered))
			} else {
				res.AvgDetConf = 0
			}
		}

		if minRecConf > 0 {
			filtered := make([]pipeline.OCRRegionResult, 0, len(res.Regions))
			for _, r := range res.Regions {
				if r.RecConfidence >= minRecConf {
					filtered = append(filtered, r)
				}
			}
			res.Regions = filtered
		}

		// Generate overlay if requested
		if overlayDir == "" {
			imageResults[i] = res
			continue
		}

		ov := pipeline.RenderOverlay(img, res, color.RGBA{255, 0, 0, 255}, color.RGBA{0, 255, 0, 255})
		if ov == nil {
			imageResults[i] = res
			continue
		}

		if err := os.MkdirAll(overlayDir, 0o750); err != nil {
			imageResults[i] = res
			continue
		}

		base := filepath.Base(meta.Path)
		outPath := filepath.Join(overlayDir, strings.TrimSuffix(base, filepath.Ext(base))+"_overlay.png")
		if f, err := os.Create(outPath); err == nil {
			_ = png.Encode(f, ov)
			_ = f.Close()
		}

		imageResults[i] = res
	}

	return imageResults, nil
}

// formatBatchResults formats the batch processing results in the specified format.
func formatBatchResults(results []*pipeline.OCRImageResult, imagePaths []string, format string) (string, error) {
	switch format {
	case "json":
		batchResult := struct {
			Images []struct {
				File string                   `json:"file"`
				OCR  *pipeline.OCRImageResult `json:"ocr"`
			} `json:"images"`
		}{}

		batchResult.Images = make([]struct {
			File string                   `json:"file"`
			OCR  *pipeline.OCRImageResult `json:"ocr"`
		}, len(results))

		for i, res := range results {
			batchResult.Images[i] = struct {
				File string                   `json:"file"`
				OCR  *pipeline.OCRImageResult `json:"ocr"`
			}{
				File: imagePaths[i],
				OCR:  res,
			}
		}

		bts, err := json.MarshalIndent(batchResult, "", "  ")
		return string(bts), err

	case "csv":
		var csvData [][]string
		// Header
		csvData = append(csvData, []string{
			"file", "region_index", "text", "confidence", "det_confidence", "x", "y", "width", "height", "language",
		})

		for i, res := range results {
			if res == nil {
				continue
			}
			file := imagePaths[i]
			if len(res.Regions) == 0 {
				// Add empty row for files with no regions
				csvData = append(csvData, []string{file, "0", "", "0", "0", "0", "0", "0", "0", ""})
			} else {
				for j, region := range res.Regions {
					csvData = append(csvData, []string{
						file,
						strconv.Itoa(j),
						region.Text,
						fmt.Sprintf("%.3f", region.RecConfidence),
						fmt.Sprintf("%.3f", region.DetConfidence),
						strconv.Itoa(region.Box.X),
						strconv.Itoa(region.Box.Y),
						strconv.Itoa(region.Box.W),
						strconv.Itoa(region.Box.H),
						region.Language,
					})
				}
			}
		}

		var output strings.Builder
		writer := csv.NewWriter(&output)
		for _, row := range csvData {
			if err := writer.Write(row); err != nil {
				return "", err
			}
		}
		writer.Flush()
		return output.String(), nil

	default: // text
		var output strings.Builder
		for i, res := range results {
			if res == nil {
				continue
			}
			if i > 0 {
				output.WriteString("\n")
			}
			output.WriteString(fmt.Sprintf("# %s\n", imagePaths[i]))
			pipeline.SortRegionsTopLeft(res)
			text, err := pipeline.ToPlainTextImage(res)
			if err != nil {
				return "", err
			}
			output.WriteString(text)
		}
		return output.String(), nil
	}
}

// discoverImageFiles finds all image files matching the given patterns.
func discoverImageFiles(args []string, recursive bool, includePatterns, excludePatterns []string) ([]string, error) {
	var imageFiles []string

	for _, arg := range args {
		info, err := os.Stat(arg)
		if err != nil {
			return nil, fmt.Errorf("cannot access %s: %w", arg, err)
		}

		if info.IsDir() {
			files, err := discoverInDirectory(arg, recursive, includePatterns, excludePatterns)
			if err != nil {
				return nil, err
			}
			imageFiles = append(imageFiles, files...)
		} else if matchesPatterns(arg, includePatterns) && !matchesPatterns(arg, excludePatterns) {
			imageFiles = append(imageFiles, arg)
		}
	}

	return imageFiles, nil
}

// discoverInDirectory recursively discovers image files in a directory.
func discoverInDirectory(dir string, recursive bool, includePatterns, excludePatterns []string) ([]string, error) {
	var files []string

	walkFn := func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			if !recursive && path != dir {
				return filepath.SkipDir
			}
			return nil
		}

		if matchesPatterns(path, includePatterns) && !matchesPatterns(path, excludePatterns) {
			files = append(files, path)
		}

		return nil
	}

	return files, filepath.Walk(dir, walkFn)
}

// matchesPatterns checks if a file path matches any of the given patterns.
func matchesPatterns(path string, patterns []string) bool {
	if len(patterns) == 0 {
		return false
	}

	base := filepath.Base(path)
	for _, pattern := range patterns {
		if matched, _ := filepath.Match(pattern, base); matched {
			return true
		}
	}
	return false
}

// parseMemoryLimit parses a memory limit string (e.g., "1GB", "512MB") into bytes.
func parseMemoryLimit(limit string) (uint64, error) {
	limit = strings.TrimSpace(strings.ToUpper(limit))

	multipliers := map[string]uint64{
		"B":  1,
		"KB": 1024,
		"MB": 1024 * 1024,
		"GB": 1024 * 1024 * 1024,
		"TB": 1024 * 1024 * 1024 * 1024,
	}

	for suffix, multiplier := range multipliers {
		if strings.HasSuffix(limit, suffix) {
			numStr := strings.TrimSuffix(limit, suffix)
			num, err := strconv.ParseFloat(numStr, 64)
			if err != nil {
				return 0, err
			}
			return uint64(num * float64(multiplier)), nil
		}
	}

	// Try parsing as plain number (bytes)
	num, err := strconv.ParseUint(limit, 10, 64)
	return num, err
}
