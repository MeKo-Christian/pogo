package cmd

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/spf13/cobra"
)

// rootCmd represents the base command when called without any subcommands.
var rootCmd = &cobra.Command{
	Use:   "pogo",
	Short: "OCR pipeline for text detection and recognition",
	Long: `A Go implementation of the OAR-OCR pipeline providing OCR (Optical Character Recognition) text detection
and recognition capabilities with support for images and PDFs.

This tool provides:
- Text detection in images using PaddleOCR models
- Text recognition with multiple language support
- PDF processing capabilities
- Both CLI and server modes
- High-performance inference with ONNX Runtime

Examples:
  pogo image input.jpg
  pogo pdf document.pdf --format json
  pogo serve --port 8080`,
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	err := rootCmd.Execute()
	if err != nil {
		os.Exit(1)
	}
}

// GetRootCommand returns the root command for testing purposes.
// This allows tests to execute commands without calling os.Exit().
func GetRootCommand() *cobra.Command {
	return rootCmd
}

func init() {
	// Global flags that apply to all commands
	rootCmd.PersistentFlags().StringP("config", "c", "", "config file (default is $HOME/.pogo.yaml)")
	rootCmd.PersistentFlags().BoolP("verbose", "v", false, "verbose output")

	// Set default models-dir from environment variable if available
	defaultModelsDir := models.DefaultModelsDir
	if envDir := os.Getenv(models.EnvModelsDir); envDir != "" {
		defaultModelsDir = envDir
	}
	rootCmd.PersistentFlags().String("models-dir", defaultModelsDir, "directory containing ONNX models (can also be set via GO_OAR_OCR_MODELS_DIR environment variable)")

	// Version flag for tests and usability
	rootCmd.PersistentFlags().Bool("version", false, "print version information and exit")
	rootCmd.PersistentPreRun = func(cmd *cobra.Command, args []string) {
		// Set up structured logging
		logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
			Level: slog.LevelInfo,
		}))
		slog.SetDefault(logger)

		v, _ := cmd.Flags().GetBool("version")
		if v {
			// Minimal version info; wired at build time in release builds
			fmt.Fprintln(cmd.OutOrStdout(), "pogo version")
		}
	}
}
