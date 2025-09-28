package batch

// Package batch provides batch processing functionality for OCR operations.

import (
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
)

// ProcessBatch processes a batch of images with the given configuration.
func ProcessBatch(imagePaths []string, config *Config) (*Result, error) {
	// Discover image files
	files, err := discoverImageFiles(imagePaths, config.Recursive, config.IncludePatterns, config.ExcludePatterns)
	if err != nil {
		return nil, fmt.Errorf("failed to discover image files: %w", err)
	}

	if len(files) == 0 {
		return nil, errors.New("no image files found")
	}

	// Set up progress callback
	var progressCallback pipeline.ProgressCallback
	if config.ShowProgress && !config.Quiet {
		progressCallback = pipeline.NewConsoleProgressCallback(
			os.Stdout,
			"Processing: ",
		).WithUpdateInterval(config.ProgressInterval)
	}

	// Build OCR pipeline
	pl, err := buildPipeline(config, progressCallback)
	if err != nil {
		return nil, fmt.Errorf("failed to build OCR pipeline: %w", err)
	}
	defer func() {
		if err := pl.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "Error closing pipeline: %v\n", err)
		}
	}()

	// Process images in parallel
	startTime := time.Now()
	results, err := processImagesParallel(pl, files, config.Confidence, config.MinRecConf, config.OverlayDir)
	duration := time.Since(startTime)

	if err != nil {
		return nil, fmt.Errorf("batch processing failed: %w", err)
	}

	return &Result{
		Results:     results,
		ImagePaths:  files,
		Duration:    duration,
		WorkerCount: config.Workers,
	}, nil
}
