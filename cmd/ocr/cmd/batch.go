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

	// Apply core OCR settings
	setCoreOCRSettings(cfg, batchConfig, setFloat64WithFlag, setStringWithFlag, setIntWithFlag)

	// Apply output settings
	setOutputSettings(cfg, batchConfig, setStringWithFlag)

	// Apply feature settings
	setFeatureSettings(cfg, batchConfig, setBoolWithFlag, setStringWithFlag, setFloat64WithFlag, setIntWithFlag)

	// Apply parallel processing settings
	setParallelProcessingSettings(cfg, batchConfig, setIntWithFlag)

	// Apply CLI-only settings
	setCLIOnlySettings(cmd, batchConfig)

	return batchConfig
}

// Helper function types for configuration mapping.
type (
	setFloat64Func func(float64, string, *float64)
	setStringFunc  func(string, string, *string)
	setIntFunc     func(int, string, *int)
	setBoolFunc    func(bool, string, *bool)
)

// setCoreOCRSettings configures core OCR parameters.
func setCoreOCRSettings(cfg *config.Config, batchConfig *batch.Config, setFloat64WithFlag setFloat64Func,
	setStringWithFlag setStringFunc, setIntWithFlag setIntFunc,
) {
	setFloat64WithFlag(float64(cfg.Pipeline.Detector.DbBoxThresh), "confidence", &batchConfig.Confidence)
	setStringWithFlag(cfg.ModelsDir, "", &batchConfig.ModelsDir)
	setStringWithFlag(cfg.Pipeline.Detector.ModelPath, "det-model", &batchConfig.DetModel)
	setStringWithFlag(cfg.Pipeline.Recognizer.ModelPath, "rec-model", &batchConfig.RecModel)
	setStringWithFlag(cfg.Pipeline.Recognizer.Language, "language", &batchConfig.Language)
	setStringWithFlag(cfg.Pipeline.Recognizer.DictPath, "dict", &batchConfig.DictCSV)
	setStringWithFlag(cfg.Pipeline.Recognizer.DictLangs, "dict-langs", &batchConfig.DictLangs)
	setIntWithFlag(cfg.Pipeline.Recognizer.ImageHeight, "rec-height", &batchConfig.RecHeight)
	setFloat64WithFlag(cfg.Pipeline.Recognizer.MinConfidence, "min-rec-conf", &batchConfig.MinRecConf)
}

// setOutputSettings configures output-related parameters.
func setOutputSettings(cfg *config.Config, batchConfig *batch.Config, setStringWithFlag func(string, string, *string)) {
	setStringWithFlag(cfg.Output.OverlayDir, "overlay-dir", &batchConfig.OverlayDir)
	setStringWithFlag(cfg.Output.Format, "format", &batchConfig.Format)
	setStringWithFlag(cfg.Output.File, "output", &batchConfig.OutputFile)
}

// setFeatureSettings configures feature flags and related parameters.
func setFeatureSettings(cfg *config.Config, batchConfig *batch.Config, setBoolWithFlag setBoolFunc,
	setStringWithFlag setStringFunc, setFloat64WithFlag setFloat64Func, setIntWithFlag setIntFunc,
) {
	// Rectification settings
	setBoolWithFlag(cfg.Features.RectificationEnabled, "rectify", &batchConfig.Rectify)
	setStringWithFlag(cfg.Features.RectificationModelPath, "rectify-model", &batchConfig.RectifyModel)
	setFloat64WithFlag(cfg.Features.RectificationThreshold, "rectify-mask-threshold", &batchConfig.RectifyMask)
	setIntWithFlag(cfg.Features.RectificationHeight, "rectify-height", &batchConfig.RectifyHeight)
	setStringWithFlag(cfg.Features.RectificationDebugDir, "rectify-debug-dir", &batchConfig.RectifyDebugDir)

	// Orientation settings
	setBoolWithFlag(cfg.Features.OrientationEnabled, "detect-orientation", &batchConfig.DetectOrientation)
	setFloat64WithFlag(cfg.Features.OrientationThreshold, "orientation-threshold", &batchConfig.OrientThresh)
	setBoolWithFlag(cfg.Features.TextlineEnabled, "detect-textline", &batchConfig.DetectTextline)
	setFloat64WithFlag(cfg.Features.TextlineThreshold, "textline-threshold", &batchConfig.TextlineThresh)
}

// setParallelProcessingSettings configures parallel processing parameters.
func setParallelProcessingSettings(cfg *config.Config, batchConfig *batch.Config,
	setIntWithFlag setIntFunc,
) {
	setIntWithFlag(cfg.Batch.Workers, "workers", &batchConfig.Workers)
	setIntWithFlag(cfg.Pipeline.Parallel.BatchSize, "batch-size", &batchConfig.BatchSize)
	setIntWithFlag(cfg.Pipeline.Resource.MaxGoroutines, "max-goroutines", &batchConfig.MaxGoroutines)
}

// setCLIOnlySettings configures CLI-only parameters that have no config file equivalent.
func setCLIOnlySettings(cmd *cobra.Command, batchConfig *batch.Config) {
	batchConfig.MemoryLimitStr, _ = cmd.Flags().GetString("memory-limit")
	batchConfig.MemoryThreshold, _ = cmd.Flags().GetFloat64("memory-threshold")
	batchConfig.Recursive, _ = cmd.Flags().GetBool("recursive")
	batchConfig.IncludePatterns, _ = cmd.Flags().GetStringSlice("include")
	batchConfig.ExcludePatterns, _ = cmd.Flags().GetStringSlice("exclude")
	batchConfig.ShowProgress, _ = cmd.Flags().GetBool("progress")
	batchConfig.Quiet, _ = cmd.Flags().GetBool("quiet")
	batchConfig.ShowStats, _ = cmd.Flags().GetBool("stats")
	batchConfig.ProgressInterval, _ = cmd.Flags().GetDuration("progress-interval")
	batchConfig.AdaptiveScaling, _ = cmd.Flags().GetBool("adaptive-scaling")
	batchConfig.Backpressure, _ = cmd.Flags().GetBool("backpressure")
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
