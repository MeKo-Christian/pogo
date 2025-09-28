package cmd

import (
	"fmt"
	"runtime"
	"time"

	"github.com/MeKo-Tech/pogo/internal/batch"
	"github.com/MeKo-Tech/pogo/internal/config"
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

// configToBatchConfig maps centralized configuration to batch.Config.
// CLI flags will override config file values through Viper's precedence system.
func configToBatchConfig(cfg *config.Config, cmd *cobra.Command) *batch.Config {
	batchConfig := &batch.Config{}

	// Core OCR settings - use centralized config with CLI flag overrides
	batchConfig.Confidence = float64(cfg.Pipeline.Detector.DbBoxThresh)
	if cmd.Flags().Changed("confidence") {
		batchConfig.Confidence, _ = cmd.Flags().GetFloat64("confidence")
	}

	batchConfig.ModelsDir = cfg.ModelsDir
	batchConfig.DetModel = cfg.Pipeline.Detector.ModelPath
	if cmd.Flags().Changed("det-model") {
		batchConfig.DetModel, _ = cmd.Flags().GetString("det-model")
	}

	batchConfig.RecModel = cfg.Pipeline.Recognizer.ModelPath
	if cmd.Flags().Changed("rec-model") {
		batchConfig.RecModel, _ = cmd.Flags().GetString("rec-model")
	}

	batchConfig.Language = cfg.Pipeline.Recognizer.Language
	if cmd.Flags().Changed("language") {
		batchConfig.Language, _ = cmd.Flags().GetString("language")
	}

	batchConfig.DictCSV = cfg.Pipeline.Recognizer.DictPath
	if cmd.Flags().Changed("dict") {
		batchConfig.DictCSV, _ = cmd.Flags().GetString("dict")
	}

	batchConfig.DictLangs = cfg.Pipeline.Recognizer.DictLangs
	if cmd.Flags().Changed("dict-langs") {
		batchConfig.DictLangs, _ = cmd.Flags().GetString("dict-langs")
	}

	batchConfig.RecHeight = cfg.Pipeline.Recognizer.ImageHeight
	if cmd.Flags().Changed("rec-height") {
		batchConfig.RecHeight, _ = cmd.Flags().GetInt("rec-height")
	}

	batchConfig.MinRecConf = cfg.Pipeline.Recognizer.MinConfidence
	if cmd.Flags().Changed("min-rec-conf") {
		batchConfig.MinRecConf, _ = cmd.Flags().GetFloat64("min-rec-conf")
	}

	batchConfig.OverlayDir = cfg.Output.OverlayDir
	if cmd.Flags().Changed("overlay-dir") {
		batchConfig.OverlayDir, _ = cmd.Flags().GetString("overlay-dir")
	}

	batchConfig.Format = cfg.Output.Format
	if cmd.Flags().Changed("format") {
		batchConfig.Format, _ = cmd.Flags().GetString("format")
	}

	batchConfig.OutputFile = cfg.Output.File
	if cmd.Flags().Changed("output") {
		batchConfig.OutputFile, _ = cmd.Flags().GetString("output")
	}

	// Rectification settings
	batchConfig.Rectify = cfg.Features.RectificationEnabled
	if cmd.Flags().Changed("rectify") {
		batchConfig.Rectify, _ = cmd.Flags().GetBool("rectify")
	}

	batchConfig.RectifyModel = cfg.Features.RectificationModelPath
	if cmd.Flags().Changed("rectify-model") {
		batchConfig.RectifyModel, _ = cmd.Flags().GetString("rectify-model")
	}

	batchConfig.RectifyMask = cfg.Features.RectificationThreshold
	if cmd.Flags().Changed("rectify-mask-threshold") {
		batchConfig.RectifyMask, _ = cmd.Flags().GetFloat64("rectify-mask-threshold")
	}

	batchConfig.RectifyHeight = cfg.Features.RectificationHeight
	if cmd.Flags().Changed("rectify-height") {
		batchConfig.RectifyHeight, _ = cmd.Flags().GetInt("rectify-height")
	}

	batchConfig.RectifyDebugDir = cfg.Features.RectificationDebugDir
	if cmd.Flags().Changed("rectify-debug-dir") {
		batchConfig.RectifyDebugDir, _ = cmd.Flags().GetString("rectify-debug-dir")
	}

	// Orientation settings
	batchConfig.DetectOrientation = cfg.Features.OrientationEnabled
	if cmd.Flags().Changed("detect-orientation") {
		batchConfig.DetectOrientation, _ = cmd.Flags().GetBool("detect-orientation")
	}

	batchConfig.OrientThresh = cfg.Features.OrientationThreshold
	if cmd.Flags().Changed("orientation-threshold") {
		batchConfig.OrientThresh, _ = cmd.Flags().GetFloat64("orientation-threshold")
	}

	batchConfig.DetectTextline = cfg.Features.TextlineEnabled
	if cmd.Flags().Changed("detect-textline") {
		batchConfig.DetectTextline, _ = cmd.Flags().GetBool("detect-textline")
	}

	batchConfig.TextlineThresh = cfg.Features.TextlineThreshold
	if cmd.Flags().Changed("textline-threshold") {
		batchConfig.TextlineThresh, _ = cmd.Flags().GetFloat64("textline-threshold")
	}

	// Parallel processing settings
	batchConfig.Workers = cfg.Batch.Workers
	if cmd.Flags().Changed("workers") {
		batchConfig.Workers, _ = cmd.Flags().GetInt("workers")
	}

	batchConfig.BatchSize = cfg.Pipeline.Parallel.BatchSize
	if cmd.Flags().Changed("batch-size") {
		batchConfig.BatchSize, _ = cmd.Flags().GetInt("batch-size")
	}

	// Memory limit string - this doesn't have a direct config equivalent, use flag
	batchConfig.MemoryLimitStr, _ = cmd.Flags().GetString("memory-limit")

	batchConfig.MaxGoroutines = cfg.Pipeline.Resource.MaxGoroutines
	if cmd.Flags().Changed("max-goroutines") {
		batchConfig.MaxGoroutines, _ = cmd.Flags().GetInt("max-goroutines")
	}

	// Memory threshold - no direct config equivalent, use flag
	batchConfig.MemoryThreshold, _ = cmd.Flags().GetFloat64("memory-threshold")

	// File discovery settings - these are typically CLI-only
	batchConfig.Recursive, _ = cmd.Flags().GetBool("recursive")
	batchConfig.IncludePatterns, _ = cmd.Flags().GetStringSlice("include")
	batchConfig.ExcludePatterns, _ = cmd.Flags().GetStringSlice("exclude")

	// Progress settings - these are typically CLI-only
	batchConfig.ShowProgress, _ = cmd.Flags().GetBool("progress")
	batchConfig.Quiet, _ = cmd.Flags().GetBool("quiet")
	batchConfig.ShowStats, _ = cmd.Flags().GetBool("stats")
	batchConfig.ProgressInterval, _ = cmd.Flags().GetDuration("progress-interval")

	// Resource management settings - these are typically CLI-only
	batchConfig.AdaptiveScaling, _ = cmd.Flags().GetBool("adaptive-scaling")
	batchConfig.Backpressure, _ = cmd.Flags().GetBool("backpressure")

	return batchConfig
}

func runBatchCommand(cmd *cobra.Command, args []string) error {
	// Get configuration from centralized system (includes CLI flags, config file, env vars, and defaults)
	cfg := GetConfig()

	// Map to batch configuration
	config := configToBatchConfig(cfg, cmd)

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
