package cmd

import (
	"fmt"
	"runtime"
	"time"

	"github.com/MeKo-Tech/pogo/internal/batch"
	"github.com/MeKo-Tech/pogo/internal/models"
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

// parseBatchFlags parses all command flags into a batch.Config struct.
func parseBatchFlags(cmd *cobra.Command) *batch.Config {
	config := &batch.Config{}

	// Core OCR settings
	config.Confidence, _ = cmd.Flags().GetFloat64("confidence")
	config.ModelsDir, _ = cmd.InheritedFlags().GetString("models-dir")
	config.DetModel, _ = cmd.Flags().GetString("det-model")
	config.RecModel, _ = cmd.Flags().GetString("rec-model")
	config.Language, _ = cmd.Flags().GetString("language")
	config.DictCSV, _ = cmd.Flags().GetString("dict")
	config.DictLangs, _ = cmd.Flags().GetString("dict-langs")
	config.RecHeight, _ = cmd.Flags().GetInt("rec-height")
	config.MinRecConf, _ = cmd.Flags().GetFloat64("min-rec-conf")
	config.OverlayDir, _ = cmd.Flags().GetString("overlay-dir")
	config.Format, _ = cmd.Flags().GetString("format")
	config.OutputFile, _ = cmd.Flags().GetString("output")

	// Rectification settings
	config.Rectify, _ = cmd.Flags().GetBool("rectify")
	config.RectifyModel, _ = cmd.Flags().GetString("rectify-model")
	config.RectifyMask, _ = cmd.Flags().GetFloat64("rectify-mask-threshold")
	config.RectifyHeight, _ = cmd.Flags().GetInt("rectify-height")
	config.RectifyDebugDir, _ = cmd.Flags().GetString("rectify-debug-dir")

	// Orientation settings
	config.DetectOrientation, _ = cmd.Flags().GetBool("detect-orientation")
	config.OrientThresh, _ = cmd.Flags().GetFloat64("orientation-threshold")
	config.DetectTextline, _ = cmd.Flags().GetBool("detect-textline")
	config.TextlineThresh, _ = cmd.Flags().GetFloat64("textline-threshold")

	// Parallel processing settings
	config.Workers, _ = cmd.Flags().GetInt("workers")
	config.BatchSize, _ = cmd.Flags().GetInt("batch-size")
	config.MemoryLimitStr, _ = cmd.Flags().GetString("memory-limit")
	config.MaxGoroutines, _ = cmd.Flags().GetInt("max-goroutines")
	config.MemoryThreshold, _ = cmd.Flags().GetFloat64("memory-threshold")

	// File discovery settings
	config.Recursive, _ = cmd.Flags().GetBool("recursive")
	config.IncludePatterns, _ = cmd.Flags().GetStringSlice("include")
	config.ExcludePatterns, _ = cmd.Flags().GetStringSlice("exclude")

	// Progress settings
	config.ShowProgress, _ = cmd.Flags().GetBool("progress")
	config.Quiet, _ = cmd.Flags().GetBool("quiet")
	config.ShowStats, _ = cmd.Flags().GetBool("stats")
	config.ProgressInterval, _ = cmd.Flags().GetDuration("progress-interval")

	// Resource management settings
	config.AdaptiveScaling, _ = cmd.Flags().GetBool("adaptive-scaling")
	config.Backpressure, _ = cmd.Flags().GetBool("backpressure")

	return config
}

func runBatchCommand(cmd *cobra.Command, args []string) error {
	// Parse configuration
	config := parseBatchFlags(cmd)

	if !config.Quiet {
		_, _ = fmt.Fprintf(cmd.OutOrStdout(), "Processing %d files...\n", len(args))
	}

	// Process batch
	result, err := batch.ProcessBatch(args, config)
	if err != nil {
		return fmt.Errorf("batch processing failed: %w", err)
	}

	// Save results
	if err := result.SaveResults(config.Format, config.OutputFile, config.Quiet); err != nil {
		return fmt.Errorf("failed to save results: %w", err)
	}

	// Print stats
	result.PrintStats(config.Quiet)

	return nil
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
	batchCmd.Flags().String("rectify-model",
		models.GetLayoutModelPath("", models.LayoutUVDoc),
		"override rectification model path")
	batchCmd.Flags().Float64("rectify-mask-threshold", 0.5, "rectification mask threshold (0..1)")
	batchCmd.Flags().Int("rectify-height", 1024, "rectified page output height (advisory)")
	batchCmd.Flags().String("rectify-debug-dir", "",
		"directory to write rectification debug images (mask, overlay, compare)")

	// Parallel processing flags
	batchCmd.Flags().IntP("workers", "w", 0, fmt.Sprintf("number of parallel workers (default: %d)", runtime.NumCPU()))
	batchCmd.Flags().Int("batch-size", 0, "batch size for micro-batching (0 = no batching)")
	batchCmd.Flags().String("memory-limit", "", "memory limit (e.g., 1GB, 512MB)")
	batchCmd.Flags().Int("max-goroutines", 0, "maximum concurrent goroutines")
	batchCmd.Flags().Float64("memory-threshold", 0.8, "memory pressure threshold (0.0-1.0)")

	// File discovery flags
	batchCmd.Flags().BoolP("recursive", "r", false, "recursively scan directories")
	batchCmd.Flags().StringSlice("include",
		[]string{"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"}, "file patterns to include")
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
