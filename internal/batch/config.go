package batch

import (
	"fmt"
	"os"
	"time"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
)

// Config holds all configuration for batch processing.
type Config struct {
	// Core OCR settings
	Confidence float64
	ModelsDir  string
	DetModel   string
	RecModel   string
	Language   string
	DictCSV    string
	DictLangs  string
	RecHeight  int
	MinRecConf float64
	OverlayDir string
	Format     string
	OutputFile string

	// Rectification settings
	Rectify         bool
	RectifyModel    string
	RectifyMask     float64
	RectifyHeight   int
	RectifyDebugDir string

	// Orientation settings
	DetectOrientation bool
	OrientThresh      float64
	DetectTextline    bool
	TextlineThresh    float64

	// Parallel processing settings
	Workers         int
	BatchSize       int
	MemoryLimitStr  string
	MaxGoroutines   int
	MemoryThreshold float64

	// File discovery settings
	Recursive       bool
	IncludePatterns []string
	ExcludePatterns []string

	// Progress settings
	ShowProgress     bool
	Quiet            bool
	ShowStats        bool
	ProgressInterval time.Duration

	// Resource management settings
	AdaptiveScaling bool
	Backpressure    bool
}

// Result holds the result of batch processing.
type Result struct {
	Results     []*pipeline.OCRImageResult
	ImagePaths  []string
	Duration    time.Duration
	WorkerCount int
}

// FormatResults formats the batch processing results in the specified format.
func (r *Result) FormatResults(format string) (string, error) {
	return formatBatchResults(r.Results, r.ImagePaths, format)
}

// SaveResults saves the formatted results to a file or stdout.
func (r *Result) SaveResults(format, outputFile string, quiet bool) error {
	output, err := r.FormatResults(format)
	if err != nil {
		return fmt.Errorf("failed to format results: %w", err)
	}

	if outputFile != "" {
		if err := os.WriteFile(outputFile, []byte(output), 0o600); err != nil {
			return fmt.Errorf("failed to write output file: %w", err)
		}
		if !quiet {
			_, _ = fmt.Fprintf(os.Stdout, "Results written to %s\n", outputFile)
		}
	} else {
		_, _ = fmt.Fprint(os.Stdout, output)
	}

	return nil
}

// PrintStats prints processing statistics.
func (r *Result) PrintStats(quiet bool) {
	if !quiet {
		stats := pipeline.CalculateParallelStats(nil, r.Results, r.Duration, r.WorkerCount)
		_, _ = fmt.Fprintf(os.Stdout, "\nProcessing Statistics:\n")
		_, _ = fmt.Fprintf(os.Stdout, "  Total images: %d\n", len(r.ImagePaths))
		_, _ = fmt.Fprintf(os.Stdout, "  Processed: %d\n", stats.ProcessedImages)
		_, _ = fmt.Fprintf(os.Stdout, "  Failed: %d\n", stats.FailedImages)
		_, _ = fmt.Fprintf(os.Stdout, "  Workers: %d\n", stats.WorkerCount)
		_, _ = fmt.Fprintf(os.Stdout, "  Duration: %v\n", stats.TotalDuration.Round(time.Millisecond))
		_, _ = fmt.Fprintf(os.Stdout, "  Avg per image: %v\n", stats.AveragePerImage.Round(time.Millisecond))
		_, _ = fmt.Fprintf(os.Stdout, "  Throughput: %.1f images/sec\n", stats.ThroughputPerSec)
	}
}
