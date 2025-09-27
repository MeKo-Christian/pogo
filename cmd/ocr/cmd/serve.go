package cmd

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/MeKo-Tech/pogo/internal/server"
	"github.com/spf13/cobra"
)

// serveCmd represents the serve command.
var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start HTTP server for OCR API",
	Long: `Start an HTTP server that provides REST API endpoints for OCR processing.

The server provides the following endpoints:
  POST /ocr/image - Process uploaded images
  GET  /health    - Health check endpoint
  GET  /models    - List available models

Examples:
  pogo serve
  pogo serve --port 8080
  pogo serve --host 0.0.0.0 --port 3000`,
	RunE: func(cmd *cobra.Command, args []string) error {
		port, _ := cmd.Flags().GetInt("port")
		host, _ := cmd.Flags().GetString("host")
		corsOrigin, _ := cmd.Flags().GetString("cors-origin")
		maxUploadSize, _ := cmd.Flags().GetInt("max-upload-size")
		timeout, _ := cmd.Flags().GetInt("timeout")
		shutdownTimeout, _ := cmd.Flags().GetInt("shutdown-timeout")
		language, _ := cmd.Flags().GetString("language")
		detModel, _ := cmd.Flags().GetString("det-model")
		recModel, _ := cmd.Flags().GetString("rec-model")
		minDetConf, _ := cmd.Flags().GetFloat64("min-det-conf")
		overlayEnable, _ := cmd.Flags().GetBool("overlay-enable")
		overlayBox, _ := cmd.Flags().GetString("overlay-box-color")
		overlayPoly, _ := cmd.Flags().GetString("overlay-poly-color")
		modelsDir, _ := cmd.InheritedFlags().GetString("models-dir")
		orientEnable, _ := cmd.Flags().GetBool("detect-orientation")
		orientThresh, _ := cmd.Flags().GetFloat64("orientation-threshold")
		textlineEnable, _ := cmd.Flags().GetBool("detect-textline")
		textlineThresh, _ := cmd.Flags().GetFloat64("textline-threshold")
		dictCSV, _ := cmd.Flags().GetString("dict")
		dictLangs, _ := cmd.Flags().GetString("dict-langs")

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		// Create server configuration
		// Build pipeline config
		pCfg := pipeline.DefaultConfig()
		pCfg.ModelsDir = modelsDir
		pCfg.Recognizer.Language = language
		// Allow polygon mode selection
		polyMode, _ := cmd.Flags().GetString("det-polygon-mode")
		if polyMode != "" {
			pCfg.Detector.PolygonMode = polyMode
		}
		if detModel != "" {
			pCfg.Detector.ModelPath = detModel
		}
		if recModel != "" {
			pCfg.Recognizer.ModelPath = recModel
		}
		if dictCSV != "" {
			pCfg.Recognizer.DictPaths = strings.Split(dictCSV, ",")
		}
		if dictLangs != "" {
			pCfg.Recognizer.DictPaths = models.GetDictionaryPathsForLanguages(modelsDir, strings.Split(dictLangs, ","))
		}
		if minDetConf > 0 {
			pCfg.Detector.DbBoxThresh = float32(minDetConf)
		}
		pCfg.Orientation.Enabled = orientEnable
		if orientThresh > 0 {
			pCfg.Orientation.ConfidenceThreshold = orientThresh
		}
		pCfg.TextLineOrientation.Enabled = textlineEnable
		if textlineThresh > 0 {
			pCfg.TextLineOrientation.ConfidenceThreshold = textlineThresh
		}

		serverConfig := server.Config{
			Host:             host,
			Port:             port,
			CORSOrigin:       corsOrigin,
			MaxUploadMB:      int64(maxUploadSize),
			TimeoutSec:       timeout,
			PipelineConfig:   pCfg,
			OverlayEnabled:   overlayEnable,
			OverlayBoxColor:  overlayBox,
			OverlayPolyColor: overlayPoly,
		}

		// Initialize server
		ocrServer, err := server.NewServer(serverConfig)
		if err != nil {
			return fmt.Errorf("failed to initialize server: %w", err)
		}
		defer func() { _ = ocrServer.Close() }()

		mux := http.NewServeMux()
		ocrServer.SetupRoutes(mux)

		httpServer := &http.Server{
			Addr:              fmt.Sprintf("%s:%d", host, port),
			Handler:           mux,
			ReadHeaderTimeout: 5 * time.Second,
			ReadTimeout:       time.Duration(timeout) * time.Second,
			WriteTimeout:      time.Duration(timeout) * time.Second,
		}

		go func() {
			slog.Info("Starting OCR server", "host", host, "port", port)
			if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				slog.Error("Server error", "error", err)
				cancel()
			}
		}()

		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGTERM, syscall.SIGINT)

		select {
		case sig := <-sigChan:
			slog.Info("Received shutdown signal", "signal", sig.String())
		case <-ctx.Done():
			slog.Info("Context cancelled, initiating shutdown")
		}

		slog.Info("Starting graceful shutdown", "timeout", fmt.Sprintf("%ds", shutdownTimeout))

		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), time.Duration(shutdownTimeout)*time.Second)
		defer shutdownCancel()

		// Shutdown HTTP server first
		slog.Info("Shutting down HTTP server")
		if err := httpServer.Shutdown(shutdownCtx); err != nil {
			slog.Error("HTTP server shutdown error", "error", err)
		} else {
			slog.Info("HTTP server shutdown completed")
		}

		// Clean up OCR server resources
		slog.Info("Cleaning up server resources")
		if err := ocrServer.Close(); err != nil {
			slog.Error("Server cleanup error", "error", err)
		} else {
			slog.Info("Server cleanup completed")
		}

		slog.Info("Graceful shutdown completed")
		return nil
	},
}

func init() {
	rootCmd.AddCommand(serveCmd)
	serveCmd.Flags().StringP("host", "H", "localhost", "server host")
	serveCmd.Flags().IntP("port", "p", 8080, "server port")
	serveCmd.Flags().String("cors-origin", "*", "CORS allowed origins")
	serveCmd.Flags().Int("max-upload-size", 50, "maximum upload size in MB")
	serveCmd.Flags().Int("timeout", 30, "request timeout in seconds")
	serveCmd.Flags().Int("shutdown-timeout", 10, "shutdown timeout in seconds")
	// Pipeline/server customization flags
	serveCmd.Flags().String("language", "en", "recognizer language for text cleaning")
	serveCmd.Flags().String("det-model", "", "override detection model path")
	serveCmd.Flags().String("rec-model", "", "override recognition model path")
	serveCmd.Flags().Float64("min-det-conf", 0.5, "detector box threshold (db_box_thresh)")
	serveCmd.Flags().String("dict", "", "comma-separated dictionary file paths to merge for recognition")
	serveCmd.Flags().String("dict-langs", "", "comma-separated language codes to auto-select dictionaries (e.g., en,de,fr)")
	serveCmd.Flags().Bool("detect-orientation", false, "enable document orientation detection")
	serveCmd.Flags().Float64("orientation-threshold", 0.7, "orientation confidence threshold (0..1)")
	serveCmd.Flags().Bool("detect-textline", false, "enable per-text-line orientation detection")
	serveCmd.Flags().Float64("textline-threshold", 0.6, "text line orientation confidence threshold (0..1)")
	serveCmd.Flags().Bool("overlay-enable", true, "enable overlay image responses")
	serveCmd.Flags().String("overlay-box-color", "#FF0000", "overlay box color (hex)")
	serveCmd.Flags().String("overlay-poly-color", "#00FF00", "overlay polygon color (hex)")
	// Detection polygon mode flag
	serveCmd.Flags().String("det-polygon-mode", "minrect", "detector polygon mode: minrect or contour")
}
