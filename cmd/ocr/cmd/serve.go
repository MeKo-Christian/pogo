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
		// Get configuration from centralized system (includes CLI flags, config file, env vars, and defaults)
		cfg := GetConfig()

		// Extract server configuration with CLI flag overrides
		host := cfg.Server.Host
		if cmd.Flags().Changed("host") {
			host, _ = cmd.Flags().GetString("host")
		}

		port := cfg.Server.Port
		if cmd.Flags().Changed("port") {
			port, _ = cmd.Flags().GetInt("port")
		}

		corsOrigin := cfg.Server.CORSOrigin
		if cmd.Flags().Changed("cors-origin") {
			corsOrigin, _ = cmd.Flags().GetString("cors-origin")
		}

		maxUploadSize := cfg.Server.MaxUploadMB
		if cmd.Flags().Changed("max-upload-size") {
			maxUploadSize, _ = cmd.Flags().GetInt("max-upload-size")
		}

		timeout := cfg.Server.TimeoutSec
		if cmd.Flags().Changed("timeout") {
			timeout, _ = cmd.Flags().GetInt("timeout")
		}

		shutdownTimeout := cfg.Server.ShutdownTimeout
		if cmd.Flags().Changed("shutdown-timeout") {
			shutdownTimeout, _ = cmd.Flags().GetInt("shutdown-timeout")
		}

		overlayEnable := cfg.Server.OverlayEnabled
		if cmd.Flags().Changed("overlay-enable") {
			overlayEnable, _ = cmd.Flags().GetBool("overlay-enable")
		}

		overlayBox := cfg.Output.OverlayBoxColor
		if cmd.Flags().Changed("overlay-box-color") {
			overlayBox, _ = cmd.Flags().GetString("overlay-box-color")
		}

		overlayPoly := cfg.Output.OverlayPolyColor
		if cmd.Flags().Changed("overlay-poly-color") {
			overlayPoly, _ = cmd.Flags().GetString("overlay-poly-color")
		}

		// Extract rate limiting configuration
		rateLimitEnabled := cfg.Server.RateLimitEnabled
		if cmd.Flags().Changed("rate-limit-enabled") {
			rateLimitEnabled, _ = cmd.Flags().GetBool("rate-limit-enabled")
		}

		requestsPerMinute := cfg.Server.RequestsPerMinute
		if cmd.Flags().Changed("requests-per-minute") {
			requestsPerMinute, _ = cmd.Flags().GetInt("requests-per-minute")
		}

		requestsPerHour := cfg.Server.RequestsPerHour
		if cmd.Flags().Changed("requests-per-hour") {
			requestsPerHour, _ = cmd.Flags().GetInt("requests-per-hour")
		}

		maxRequestsPerDay := cfg.Server.MaxRequestsPerDay
		if cmd.Flags().Changed("max-requests-per-day") {
			maxRequestsPerDay, _ = cmd.Flags().GetInt("max-requests-per-day")
		}

		maxDataPerDay := cfg.Server.MaxDataPerDay
		if cmd.Flags().Changed("max-data-per-day") {
			maxDataPerDay, _ = cmd.Flags().GetInt64("max-data-per-day")
		}

		// Extract pipeline configuration with CLI flag overrides
		language := cfg.Pipeline.Recognizer.Language
		if cmd.Flags().Changed("language") {
			language, _ = cmd.Flags().GetString("language")
		}

		detModel := cfg.Pipeline.Detector.ModelPath
		if cmd.Flags().Changed("det-model") {
			detModel, _ = cmd.Flags().GetString("det-model")
		}

		recModel := cfg.Pipeline.Recognizer.ModelPath
		if cmd.Flags().Changed("rec-model") {
			recModel, _ = cmd.Flags().GetString("rec-model")
		}

		minDetConf := float64(cfg.Pipeline.Detector.DbBoxThresh)
		if cmd.Flags().Changed("min-det-conf") {
			minDetConf, _ = cmd.Flags().GetFloat64("min-det-conf")
		}

		// Multi-scale detection options
		msEnabled := cfg.Pipeline.Detector.MultiScale.Enabled
		if cmd.Flags().Changed("det-multiscale") {
			msEnabled, _ = cmd.Flags().GetBool("det-multiscale")
		}
		msScales := cfg.Pipeline.Detector.MultiScale.Scales
		if cmd.Flags().Changed("det-scales") {
			msScales, _ = cmd.Flags().GetFloat64Slice("det-scales")
		}
		msMergeIoU := cfg.Pipeline.Detector.MultiScale.MergeIoU
		if cmd.Flags().Changed("det-merge-iou") {
			msMergeIoU, _ = cmd.Flags().GetFloat64("det-merge-iou")
		}
		msAdaptive := cfg.Pipeline.Detector.MultiScale.Adaptive
		if cmd.Flags().Changed("det-ms-adaptive") {
			msAdaptive, _ = cmd.Flags().GetBool("det-ms-adaptive")
		}
		msMaxLevels := cfg.Pipeline.Detector.MultiScale.MaxLevels
		if cmd.Flags().Changed("det-ms-max-levels") {
			msMaxLevels, _ = cmd.Flags().GetInt("det-ms-max-levels")
		}
		msMinSide := cfg.Pipeline.Detector.MultiScale.MinSide
		if cmd.Flags().Changed("det-ms-min-side") {
			msMinSide, _ = cmd.Flags().GetInt("det-ms-min-side")
		}
		msIncr := cfg.Pipeline.Detector.MultiScale.IncrementalMerge
		if cmd.Flags().Changed("det-ms-incremental-merge") {
			msIncr, _ = cmd.Flags().GetBool("det-ms-incremental-merge")
		}

		orientEnable := cfg.Features.OrientationEnabled
		if cmd.Flags().Changed("detect-orientation") {
			orientEnable, _ = cmd.Flags().GetBool("detect-orientation")
		}

		orientThresh := cfg.Features.OrientationThreshold
		if cmd.Flags().Changed("orientation-threshold") {
			orientThresh, _ = cmd.Flags().GetFloat64("orientation-threshold")
		}

		textlineEnable := cfg.Features.TextlineEnabled
		if cmd.Flags().Changed("detect-textline") {
			textlineEnable, _ = cmd.Flags().GetBool("detect-textline")
		}

		textlineThresh := cfg.Features.TextlineThreshold
		if cmd.Flags().Changed("textline-threshold") {
			textlineThresh, _ = cmd.Flags().GetFloat64("textline-threshold")
		}

		dictCSV := cfg.Pipeline.Recognizer.DictPath
		if cmd.Flags().Changed("dict") {
			dictCSV, _ = cmd.Flags().GetString("dict")
		}

		dictLangs := cfg.Pipeline.Recognizer.DictLangs
		if cmd.Flags().Changed("dict-langs") {
			dictLangs, _ = cmd.Flags().GetString("dict-langs")
		}

		// Validate port number
		if port < 1 || port > 65535 {
			return fmt.Errorf("invalid port number: %d (must be between 1 and 65535)", port)
		}

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		// Create server configuration
		// Build pipeline config using centralized configuration
		pCfg := pipeline.DefaultConfig()
		pCfg.ModelsDir = cfg.ModelsDir
		pCfg.Recognizer.Language = language
		// Barcode config from flags/env
		if cmd.Flags().Changed("barcodes") || cmd.Flags().Changed("barcode-types") || cmd.Flags().Changed("barcode-min-size") || cfg.Features.BarcodeEnabled || cfg.Features.BarcodeTypes != "" || cfg.Features.BarcodeMinSize > 0 {
			pCfg.Barcode.Enabled = cfg.Features.BarcodeEnabled
			if cmd.Flags().Changed("barcodes") {
				pCfg.Barcode.Enabled, _ = cmd.Flags().GetBool("barcodes")
			}
			var typesCSV string
			if cmd.Flags().Changed("barcode-types") {
				typesCSV, _ = cmd.Flags().GetString("barcode-types")
			} else {
				typesCSV = cfg.Features.BarcodeTypes
			}
			if strings.TrimSpace(typesCSV) != "" {
				pCfg.Barcode.Types = strings.Split(typesCSV, ",")
			}
			if cmd.Flags().Changed("barcode-min-size") {
				pCfg.Barcode.MinSize, _ = cmd.Flags().GetInt("barcode-min-size")
			} else if cfg.Features.BarcodeMinSize > 0 {
				pCfg.Barcode.MinSize = cfg.Features.BarcodeMinSize
			}
		}

		// Allow polygon mode selection
		polyMode := cfg.Pipeline.Detector.PolygonMode
		if cmd.Flags().Changed("det-polygon-mode") {
			polyMode, _ = cmd.Flags().GetString("det-polygon-mode")
		}
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
			pCfg.Recognizer.DictPaths = models.GetDictionaryPathsForLanguages(cfg.ModelsDir, strings.Split(dictLangs, ","))
		}
		if minDetConf > 0 {
			pCfg.Detector.DbBoxThresh = float32(minDetConf)
		}
		// Apply multi-scale detection configuration
		pCfg.Detector.MultiScale.Enabled = msEnabled
		if len(msScales) > 0 {
			pCfg.Detector.MultiScale.Scales = msScales
		}
		if msMergeIoU > 0 {
			pCfg.Detector.MultiScale.MergeIoU = msMergeIoU
		}
		pCfg.Detector.MultiScale.Adaptive = msAdaptive
		if msMaxLevels > 0 {
			pCfg.Detector.MultiScale.MaxLevels = msMaxLevels
		}
		if msMinSide > 0 {
			pCfg.Detector.MultiScale.MinSide = msMinSide
		}
		pCfg.Detector.MultiScale.IncrementalMerge = msIncr
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
			RateLimit: server.RateLimitConfig{
				Enabled:           rateLimitEnabled,
				RequestsPerMinute: requestsPerMinute,
				RequestsPerHour:   requestsPerHour,
				MaxRequestsPerDay: maxRequestsPerDay,
				MaxDataPerDay:     maxDataPerDay,
			},
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
	serveCmd.Flags().String("dict-langs", "",
		"comma-separated language codes to auto-select dictionaries (e.g., en,de,fr)")
	serveCmd.Flags().Bool("detect-orientation", false, "enable document orientation detection")
	serveCmd.Flags().Float64("orientation-threshold", 0.7, "orientation confidence threshold (0..1)")
	serveCmd.Flags().Bool("detect-textline", false, "enable per-text-line orientation detection")
	serveCmd.Flags().Float64("textline-threshold", 0.6, "text line orientation confidence threshold (0..1)")
	serveCmd.Flags().Bool("overlay-enable", true, "enable overlay image responses")
	serveCmd.Flags().String("overlay-box-color", "#FF0000", "overlay box color (hex)")
	serveCmd.Flags().String("overlay-poly-color", "#00FF00", "overlay polygon color (hex)")
	// Detection polygon mode flag
	serveCmd.Flags().String("det-polygon-mode", "minrect", "detector polygon mode: minrect or contour")
	// Multi-scale detection flags (parity with image/pdf)
	serveCmd.Flags().Bool("det-multiscale", false, "enable multi-scale detection (pyramid)")
	serveCmd.Flags().Float64Slice("det-scales", []float64{1.0, 0.75, 0.5}, "relative scales for multi-scale detection (e.g., 1.0,0.75,0.5)")
	serveCmd.Flags().Float64("det-merge-iou", 0.3, "IoU threshold for merging regions across scales")
	serveCmd.Flags().Bool("det-ms-adaptive", false, "enable adaptive pyramid scaling (auto scales based on image size)")
	serveCmd.Flags().Int("det-ms-max-levels", 3, "maximum pyramid levels when adaptive is enabled (including 1.0)")
	serveCmd.Flags().Int("det-ms-min-side", 320, "stop adaptive scaling when min(image side * scale) <= this value")
	// Rate limiting flags
	serveCmd.Flags().Bool("rate-limit-enabled", false, "enable rate limiting")
	serveCmd.Flags().Int("requests-per-minute", 60, "maximum requests per minute per client")
	serveCmd.Flags().Int("requests-per-hour", 1000, "maximum requests per hour per client")
	serveCmd.Flags().Int("max-requests-per-day", 5000, "maximum requests per day per client")
	serveCmd.Flags().Int64("max-data-per-day", 100*1024*1024, "maximum data processed per day per client (bytes)")

	// Barcode flags (optional; server-wide defaults)
	serveCmd.Flags().Bool("barcodes", false, "enable barcode detection in server pipeline")
	serveCmd.Flags().String("barcode-types", "", "comma-separated types to detect (e.g., qr,ean13,upca,code128,pdf417,datamatrix,aztec,ean8,upce,itf,codabar,code39)")
	serveCmd.Flags().Int("barcode-min-size", 0, "minimum expected barcode size in pixels (hint)")
}

// Ensure server help mentions docs for multi-scale
func init() {
	serveCmd.SetHelpFunc(func(cmd *cobra.Command, args []string) {
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
