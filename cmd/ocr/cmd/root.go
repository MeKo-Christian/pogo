package cmd

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/MeKo-Tech/pogo/internal/config"
	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	// Global configuration loader.
	configLoader *config.Loader
	// Global configuration.
	globalConfig *config.Config
	// Configuration file path.
	cfgFile string
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
	RunE: func(cmd *cobra.Command, args []string) error {
		v, _ := cmd.PersistentFlags().GetBool("version")
		if v {
			// Print version info and return
			_, _ = fmt.Fprintln(cmd.OutOrStdout(), "pogo version dev")
			_, _ = fmt.Fprintln(cmd.OutOrStdout(), "Build: local development build")
			_, _ = fmt.Fprintln(cmd.OutOrStdout(), "Commit: local")
			_, _ = fmt.Fprintln(cmd.OutOrStdout(), "Date: development")
			return nil
		}
		// If no version flag, show help
		return cmd.Help()
	},
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
	// Initialize configuration loader
	cobra.OnInitialize(initConfig)

	// Global flags that apply to all commands
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is search in ., $HOME, $HOME/.config/pogo, /etc/pogo)")
	rootCmd.PersistentFlags().BoolP("verbose", "v", false, "verbose output (equivalent to --log-level=debug)")
	rootCmd.PersistentFlags().String("log-level", "info", "log level (debug, info, warn, error)")

	// Set default models-dir from environment variable if available
	defaultModelsDir := models.DefaultModelsDir
	if envDir := os.Getenv(models.EnvModelsDir); envDir != "" {
		defaultModelsDir = envDir
	}
	rootCmd.PersistentFlags().String("models-dir", defaultModelsDir,
		"directory containing ONNX models (can also be set via GO_OAR_OCR_MODELS_DIR environment variable)")

	// Version flag for tests and usability
	rootCmd.PersistentFlags().Bool("version", false, "print version information and exit")

	// Bind flags to viper
	viper.BindPFlag("verbose", rootCmd.PersistentFlags().Lookup("verbose"))
	viper.BindPFlag("log_level", rootCmd.PersistentFlags().Lookup("log-level"))
	viper.BindPFlag("models_dir", rootCmd.PersistentFlags().Lookup("models-dir"))

	rootCmd.PersistentPreRun = func(cmd *cobra.Command, args []string) {
		// Initialize configuration if not already done
		if globalConfig == nil {
			initConfig()
		}

		// Determine log level from config
		var logLevel slog.Level

		// Check verbose flag first for backward compatibility
		if globalConfig.Verbose {
			logLevel = slog.LevelDebug
		} else {
			// Parse log-level from config
			switch globalConfig.LogLevel {
			case "debug":
				logLevel = slog.LevelDebug
			case "info":
				logLevel = slog.LevelInfo
			case "warn":
				logLevel = slog.LevelWarn
			case "error":
				logLevel = slog.LevelError
			default:
				logLevel = slog.LevelInfo
			}
		}

		// Set up structured logging
		logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
			Level: logLevel,
		}))
		slog.SetDefault(logger)
	}
}

// initConfig reads in config file and ENV variables if set.
func initConfig() {
	configLoader = config.NewLoader()

	var err error
	if cfgFile != "" {
		// Use config file from the flag
		globalConfig, err = configLoader.LoadWithFile(cfgFile)
	} else {
		// Search for config in default locations
		globalConfig, err = configLoader.Load()
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading configuration: %v\n", err)
		os.Exit(1)
	}
}

// GetConfig returns the global configuration.
func GetConfig() *config.Config {
	if globalConfig == nil {
		initConfig()
	}

	// Reload configuration to ensure CLI flags are included
	// This is necessary because flag binding happens after initial config loading
	loader := GetConfigLoader()
	var cfg config.Config
	if err := loader.GetViper().Unmarshal(&cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Error unmarshaling updated configuration: %v\n", err)
		return globalConfig // Return the original config if unmarshal fails
	}

	return &cfg
}

// GetConfigLoader returns the global configuration loader.
func GetConfigLoader() *config.Loader {
	if configLoader == nil {
		configLoader = config.NewLoader()
	}
	return configLoader
}
