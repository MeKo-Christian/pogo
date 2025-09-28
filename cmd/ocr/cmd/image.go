package cmd

import (
	"encoding/json"
	"errors"
	"fmt"
	"image/color"
	"image/png"
	"os"
	"strconv"
	"strings"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/onnx"
	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

const (
	outputFormatJSON = "json"
	outputFormatCSV  = "csv"
	outputFormatText = "text"
)

// imageCmd represents the image command.
var imageCmd = &cobra.Command{
	Use:   "image",
	Short: "Process images for OCR text detection and recognition",
	Long: `Process one or more image files to extract text using OCR.

Supported formats: JPEG, PNG, BMP, TIFF

Examples:
  pogo image photo.jpg
  pogo image *.png --format json
  pogo image document.jpg --output results.json`,
	Args:         cobra.ArbitraryArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		// Help handling for tests
		if len(args) > 0 && (args[0] == "--help" || args[0] == "-h") {
			return cmd.Help()
		}
		if len(args) == 0 {
			return errors.New("no input files provided")
		}

		// Get configuration (includes CLI flags, config file, env vars, and defaults)
		cfg := GetConfig()

		// Extract values from configuration for backwards compatibility
		confFlag := float64(cfg.Pipeline.Detector.DbBoxThresh)
		modelsDir := cfg.ModelsDir
		detModel := cfg.Pipeline.Detector.ModelPath
		recModel := cfg.Pipeline.Recognizer.ModelPath
		polyMode := cfg.Pipeline.Detector.PolygonMode
		lang := cfg.Pipeline.Recognizer.Language
		dictCSV := cfg.Pipeline.Recognizer.DictPath
		dictLangs := cfg.Pipeline.Recognizer.DictLangs
		recH := cfg.Pipeline.Recognizer.ImageHeight
		minRecConf := cfg.Pipeline.Recognizer.MinConfidence
		overlayDir := cfg.Output.OverlayDir
		format := cfg.Output.Format
		outputFile := cfg.Output.File
		detectOrientation := cfg.Features.OrientationEnabled
		orientThresh := cfg.Features.OrientationThreshold
		detectTextline := cfg.Features.TextlineEnabled
		textlineThresh := cfg.Features.TextlineThreshold
		rectify := cfg.Features.RectificationEnabled
		rectifyModel := cfg.Features.RectificationModelPath
		rectifyMask := cfg.Features.RectificationThreshold
		rectifyHeight := cfg.Features.RectificationHeight
		rectifyDebugDir := cfg.Features.RectificationDebugDir
		useGPU := cfg.GPU.Enabled
		gpuDevice := cfg.GPU.Device
		gpuMemLimit := cfg.GPU.MemoryLimit

		// Validate confidence threshold
		if confFlag < 0 || confFlag > 1 {
			return fmt.Errorf("invalid confidence threshold: %.2f (must be between 0.0 and 1.0)", confFlag)
		}

		// Validate output format
		validFormats := []string{outputFormatText, outputFormatJSON, outputFormatCSV}
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

		if _, err := fmt.Fprintf(cmd.OutOrStdout(), "Processing %d image(s)\n", len(args)); err != nil {
			return fmt.Errorf("failed to write to stdout: %w", err)
		}

		// Parse GPU memory limit
		var gpuMemLimitBytes uint64
		if gpuMemLimit != "" && useGPU {
			if gpuMemLimit == "auto" {
				gpuMemLimitBytes = onnx.GetRecommendedGPUMemLimit()
			} else {
				// Parse memory limit (supports suffixes like "2GB", "512MB")
				memLimitBytes, err := parseMemorySize(gpuMemLimit)
				if err != nil {
					return fmt.Errorf("invalid GPU memory limit: %w", err)
				}
				gpuMemLimitBytes = memLimitBytes
			}
		}

		// Build OCR pipeline
		b := pipeline.NewBuilder().WithModelsDir(modelsDir).WithLanguage(lang)
		if useGPU {
			b = b.WithGPU(true).WithGPUDevice(gpuDevice).WithGPUMemoryLimit(gpuMemLimitBytes)
		}
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
		if detectOrientation {
			b = b.WithOrientation(true)
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
		// Configure detector polygon mode
		if polyMode != "" {
			b = b.WithDetectorPolygonMode(polyMode)
		}
		pl, err := b.Build()
		if err != nil {
			return fmt.Errorf("failed to build OCR pipeline: %w", err)
		}
		defer func() {
			if err := pl.Close(); err != nil {
				// Log the error but don't return it since we're in a defer
				fmt.Fprintf(os.Stderr, "Error closing pipeline: %v", err)
			}
		}()

		cons := utils.DefaultImageConstraints()
		var outputs []string
		for _, pth := range args {
			if !utils.IsSupportedImage(pth) {
				return fmt.Errorf("unsupported image format: %s", pth)
			}
			img, meta, err := utils.LoadImage(pth)
			if err != nil {
				return fmt.Errorf("failed to load %s: %w", pth, err)
			}
			if err := utils.ValidateImageConstraints(img, cons); err != nil {
				if _, err := fmt.Fprintf(cmd.OutOrStdout(), "warning: %s: %v", pth, err); err != nil {
					return fmt.Errorf("failed to write warning to stdout: %w", err)
				}
			}
			res, err := pl.ProcessImage(img)
			if err != nil {
				return fmt.Errorf("OCR failed for %s: %w", pth, err)
			}
			// Optional post-filter by detection confidence
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
			// Optional filter by recognition confidence
			if minRecConf > 0 {
				filtered := make([]pipeline.OCRRegionResult, 0, len(res.Regions))
				for _, r := range res.Regions {
					if r.RecConfidence >= minRecConf {
						filtered = append(filtered, r)
					}
				}
				res.Regions = filtered
			}
			// Optional overlay rendering
			if overlayDir != "" {
				ov := pipeline.RenderOverlay(img, res, color.RGBA{255, 0, 0, 255}, color.RGBA{0, 255, 0, 255})
				if ov != nil {
					if err := os.MkdirAll(overlayDir, 0o750); err == nil {
						base := meta.Path
						if idx := strings.LastIndex(base, "/"); idx >= 0 {
							base = base[idx+1:]
						}
						outPath := overlayDir + "/" + strings.TrimSuffix(base, ".png") + "_overlay.png"
						f, err := os.Create(outPath) //nolint:gosec // G304: Creating overlay output file with user-controlled path
						if err == nil {
							_ = png.Encode(f, ov)
							_ = f.Close()
							if _, err := fmt.Fprintf(cmd.OutOrStdout(), "Saved overlay: %s", outPath); err != nil {
								return fmt.Errorf("failed to write to stdout: %w", err)
							}
						}
					}
				}
			}
			switch format {
			case outputFormatJSON:
				obj := struct {
					File string                   `json:"file"`
					OCR  *pipeline.OCRImageResult `json:"ocr"`
				}{File: meta.Path, OCR: res}
				bts, err := json.MarshalIndent(obj, "", "  ")
				if err != nil {
					return fmt.Errorf("failed to marshal JSON: %w", err)
				}
				outputs = append(outputs, string(bts))
			case outputFormatCSV:
				s, err := pipeline.ToCSVImage(res)
				if err != nil {
					return fmt.Errorf("format csv failed: %w", err)
				}
				if len(args) > 1 {
					s = "# " + meta.Path + s
				}
				outputs = append(outputs, s)
			default:
				pipeline.SortRegionsTopLeft(res)
				s, err := pipeline.ToPlainTextImage(res)
				if err != nil {
					return fmt.Errorf("format text failed: %w", err)
				}
				s = fmt.Sprintf("%s:%s", meta.Path, s)
				outputs = append(outputs, s)
			}
		}
		final := strings.Join(outputs, "")
		if outputFile != "" {
			if err := os.WriteFile(outputFile, []byte(final), 0o600); err != nil {
				return fmt.Errorf("failed to write output file: %w", err)
			}
			if _, err := fmt.Fprintf(cmd.OutOrStdout(), "Results written to %s", outputFile); err != nil {
				return err
			}
		} else {
			if _, err := fmt.Fprintln(cmd.OutOrStdout(), final); err != nil {
				return fmt.Errorf("failed to write final output: %w", err)
			}
		}
		return nil
	},
}

// parseMemorySize parses memory size strings like "2GB", "512MB", "1024".
func parseMemorySize(s string) (uint64, error) {
	if s == "" {
		return 0, nil
	}

	// Convert to upper case and handle common suffixes
	s = strings.ToUpper(strings.TrimSpace(s))

	var multiplier uint64 = 1
	switch {
	case strings.HasSuffix(s, "KB"):
		multiplier = 1024
		s = s[:len(s)-2]
	case strings.HasSuffix(s, "MB"):
		multiplier = 1024 * 1024
		s = s[:len(s)-2]
	case strings.HasSuffix(s, "GB"):
		multiplier = 1024 * 1024 * 1024
		s = s[:len(s)-2]
	case strings.HasSuffix(s, "K"):
		multiplier = 1024
		s = s[:len(s)-1]
	case strings.HasSuffix(s, "M"):
		multiplier = 1024 * 1024
		s = s[:len(s)-1]
	case strings.HasSuffix(s, "G"):
		multiplier = 1024 * 1024 * 1024
		s = s[:len(s)-1]
	}

	value, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid memory size: %s", s)
	}

	return value * multiplier, nil
}

func addImageFlags(cmd *cobra.Command) {
	// Image-specific flags
	cmd.Flags().StringP("format", "f", "text", "output format (text, json, csv)")
	cmd.Flags().StringP("output", "o", "", "output file (default: stdout)")
	cmd.Flags().Float64("confidence", 0.5, "minimum confidence threshold")
	cmd.Flags().Bool("detect-orientation", false, "enable document orientation detection")
	cmd.Flags().Float64("orientation-threshold", 0.7, "orientation confidence threshold (0..1)")
	cmd.Flags().StringP("language", "l", "en", "recognition language")
	cmd.Flags().String("dict", "", "comma-separated dictionary file paths to merge for recognition")
	cmd.Flags().String("dict-langs", "", "comma-separated language codes to auto-select "+
		"dictionaries (e.g., en,de,fr)")
	cmd.Flags().Int("rec-height", 0, "recognizer input height (0=auto, typical: 32 or 48)")
	cmd.Flags().Float64("min-rec-conf", 0.0, "minimum recognition confidence (filter output)")
	cmd.Flags().String("overlay-dir", "", "directory to write overlay images (drawn boxes)")
	cmd.Flags().Bool("detect", true, "run detection (deprecated; pipeline runs full OCR)")
	cmd.Flags().String("det-model", "", "override detection model path (defaults to organized models path)")
	cmd.Flags().String("rec-model", "", "override recognition model path (defaults to organized models path)")
	cmd.Flags().Bool("detect-textline", false, "enable per-text-line orientation detection")
	cmd.Flags().Float64("textline-threshold", 0.6, "text line orientation confidence threshold (0..1)")

	// Rectification flags (minimal CPU-only)
	cmd.Flags().Bool("rectify", false, "enable document rectification (experimental)")
	cmd.Flags().String("rectify-model",
		models.GetLayoutModelPath("", models.LayoutUVDoc), "override rectification model path")
	cmd.Flags().Float64("rectify-mask-threshold", 0.5, "rectification mask threshold (0..1)")
	cmd.Flags().Int("rectify-height", 1024, "rectified page output height (advisory)")
	cmd.Flags().String("rectify-debug-dir", "", "directory to write rectification debug images (mask, overlay)")

	// GPU acceleration flags
	cmd.Flags().Bool("gpu", false, "enable GPU acceleration using CUDA")
	cmd.Flags().Int("gpu-device", 0, "CUDA device ID to use (default: 0)")
	cmd.Flags().String("gpu-mem-limit", "auto", "GPU memory limit "+
		"(e.g., '2GB', '512MB', 'auto' for recommended limit)")

	// Detection polygon mode: minrect (default) or contour
	cmd.Flags().String("det-polygon-mode", "minrect", "detector polygon mode: minrect or contour")

	// Morphological operations flags
	cmd.Flags().String("morph-op", "none", "morphological operation: none, dilate, erode, opening, closing, smooth")
	cmd.Flags().Int("morph-kernel-size", 3, "morphological operation kernel size (e.g., 3 for 3x3)")
	cmd.Flags().Int("morph-iterations", 1, "number of morphological operation iterations")

	// Adaptive threshold flags
	cmd.Flags().Bool("adaptive-thresholds", false, "enable adaptive threshold calculation")
	cmd.Flags().String("adaptive-thresh-method", "histogram", "adaptive threshold method: otsu, histogram, dynamic")
	cmd.Flags().Float32("adaptive-thresh-min-db", 0.1, "minimum allowed db_thresh value")
	cmd.Flags().Float32("adaptive-thresh-max-db", 0.8, "maximum allowed db_thresh value")
	cmd.Flags().Float32("adaptive-thresh-min-box", 0.3, "minimum allowed box_thresh value")
	cmd.Flags().Float32("adaptive-thresh-max-box", 0.9, "maximum allowed box_thresh value")
}

// bindImageFlags binds all flags to viper configuration keys.
func bindImageFlags(cmd *cobra.Command) {
	flagBindings := []struct {
		key  string
		flag string
	}{
		{"output.format", "format"},
		{"output.file", "output"},
		{"pipeline.detector.db_box_thresh", "confidence"},
		{"features.orientation_enabled", "detect-orientation"},
		{"features.orientation_threshold", "orientation-threshold"},
		{"pipeline.recognizer.language", "language"},
		{"pipeline.recognizer.dict_path", "dict"},
		{"pipeline.recognizer.dict_langs", "dict-langs"},
		{"pipeline.recognizer.image_height", "rec-height"},
		{"pipeline.recognizer.min_confidence", "min-rec-conf"},
		{"output.overlay_dir", "overlay-dir"},
		{"pipeline.detector.model_path", "det-model"},
		{"pipeline.recognizer.model_path", "rec-model"},
		{"features.textline_enabled", "detect-textline"},
		{"features.textline_threshold", "textline-threshold"},
		{"features.rectification_enabled", "rectify"},
		{"features.rectification_model_path", "rectify-model"},
		{"features.rectification_threshold", "rectify-mask-threshold"},
		{"features.rectification_height", "rectify-height"},
		{"features.rectification_debug_dir", "rectify-debug-dir"},
		{"gpu.enabled", "gpu"},
		{"gpu.device", "gpu-device"},
		{"gpu.memory_limit", "gpu-mem-limit"},
		{"pipeline.detector.polygon_mode", "det-polygon-mode"},
		{"pipeline.detector.morphology.operation", "morph-op"},
		{"pipeline.detector.morphology.kernel_size", "morph-kernel-size"},
		{"pipeline.detector.morphology.iterations", "morph-iterations"},
		{"pipeline.detector.adaptive_thresholds.enabled", "adaptive-thresholds"},
		{"pipeline.detector.adaptive_thresholds.method", "adaptive-thresh-method"},
		{"pipeline.detector.adaptive_thresholds.min_db_thresh", "adaptive-thresh-min-db"},
		{"pipeline.detector.adaptive_thresholds.max_db_thresh", "adaptive-thresh-max-db"},
		{"pipeline.detector.adaptive_thresholds.min_box_thresh", "adaptive-thresh-min-box"},
		{"pipeline.detector.adaptive_thresholds.max_box_thresh", "adaptive-thresh-max-box"},
	}

	for _, binding := range flagBindings {
		if err := viper.BindPFlag(binding.key, cmd.Flags().Lookup(binding.flag)); err != nil {
			panic(fmt.Sprintf("failed to bind flag %s: %v", binding.flag, err))
		}
	}
}

func init() {
	rootCmd.AddCommand(imageCmd)

	addImageFlags(imageCmd)
	bindImageFlags(imageCmd)

	// Ensure subcommand help prints expected sections when executed directly in tests
	imageCmd.SetHelpFunc(func(cmd *cobra.Command, args []string) {
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
	})
}

// GetImageCommand returns the image command for testing purposes.
func GetImageCommand() *cobra.Command {
	return imageCmd
}
