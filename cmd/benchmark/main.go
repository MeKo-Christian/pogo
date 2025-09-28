package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/MeKo-Tech/pogo/internal/benchmark"
)

func main() {
	var (
		modelsDir  = flag.String("models", "models", "Directory containing ONNX models")
		iterations = flag.Int("iterations", 3, "Number of iterations per benchmark")
		outputFile = flag.String("output", "", "Output file for results (optional)")
		verbose    = flag.Bool("verbose", false, "Verbose output")
	)
	flag.Parse()

	fmt.Println("pogo GPU vs CPU Performance Benchmark")
	fmt.Println("============================================")

	// Check if models directory exists
	if _, err := os.Stat(*modelsDir); os.IsNotExist(err) {
		log.Fatalf("Models directory not found: %s", *modelsDir)
	}

	// Create benchmark
	gpuBench := benchmark.NewGPUVSCPUBenchmark(*modelsDir)

	// Add additional test images if they exist
	additionalImages := []struct {
		path, desc, size string
	}{
		{"testdata/images/multiline/multiline_text.png", "Multiline text", "Medium"},
		{"testdata/images/rotated/rotated_text.png", "Rotated text", "Medium"},
		{"testdata/images/scanned/scanned_document.png", "Scanned document", "Large"},
	}

	for _, img := range additionalImages {
		if fileExists(img.path) {
			gpuBench.AddTestImage(img.path, img.desc, img.size)
			if *verbose {
				fmt.Printf("Added test image: %s\n", img.path)
			}
		}
	}

	// Run benchmarks
	fmt.Printf("Running benchmarks with %d iterations per test...\n\n", *iterations)

	results, err := gpuBench.RunBenchmark(*iterations)
	if err != nil {
		log.Fatalf("Benchmark failed: %v", err)
	}

	// Print detailed results
	gpuBench.PrintDetailedResults()

	// Save results to file if requested
	if *outputFile != "" {
		if err := saveResultsToFile(*outputFile, results); err != nil {
			log.Printf("Failed to save results to file: %v", err)
		} else {
			fmt.Printf("Results saved to: %s\n", *outputFile)
		}
	}
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return !os.IsNotExist(err)
}

func saveResultsToFile(filename string, results []benchmark.GPUVSCPUBenchmarkResult) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer func() { _ = file.Close() }()

	// Write header
	_, _ = fmt.Fprintln(file, "pogo GPU vs CPU Benchmark Results")
	_, _ = fmt.Fprintln(file, "======================================")
	_, _ = fmt.Fprintln(file)

	// Write individual results
	for _, result := range results {
		_, _ = fmt.Fprintf(file, "%s\n", result.String())
	}

	_, _ = fmt.Fprintln(file)
	_, _ = fmt.Fprintln(file, "CSV Format:")
	_, _ = fmt.Fprintln(file, "Image,Size,CPU_Duration_ms,GPU_Duration_ms,Speedup,Memory_Diff_KB,GPU_Available")

	for _, result := range results {
		cpuMs := float64(result.CPUResult.Duration.Nanoseconds()) / 1e6
		gpuMs := float64(0)
		if result.GPUAvailable {
			gpuMs = float64(result.GPUResult.Duration.Nanoseconds()) / 1e6
		}

		_, _ = fmt.Fprintf(file, "%s,%s,%.2f,%.2f,%.2f,%d,%t\n",
			filepath.Base(result.ImagePath),
			result.ImageSize,
			cpuMs,
			gpuMs,
			result.SpeedupFactor,
			result.MemoryDiff,
			result.GPUAvailable,
		)
	}

	return nil
}
