package benchmark

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/MeKo-Tech/pogo/internal/testutil"
)

// Timer provides simple timing utilities for benchmarking.
type Timer struct {
	start    time.Time
	name     string
	duration time.Duration
}

// NewTimer creates a new timer with the given name.
func NewTimer(name string) *Timer {
	return &Timer{
		name:  name,
		start: time.Now(),
	}
}

// Stop stops the timer and returns the elapsed duration.
func (t *Timer) Stop() time.Duration {
	t.duration = time.Since(t.start)
	return t.duration
}

// Duration returns the recorded duration (only valid after Stop()).
func (t *Timer) Duration() time.Duration {
	return t.duration
}

// String returns a formatted string representation of the timer.
func (t *Timer) String() string {
	return fmt.Sprintf("%s: %v", t.name, t.duration)
}

// MemoryStats holds memory usage statistics.
type MemoryStats struct {
	AllocBytes      uint64  // Currently allocated bytes
	TotalAllocBytes uint64  // Total allocated bytes (cumulative)
	SysBytes        uint64  // Total bytes from system
	NumGC           uint32  // Number of GC runs
	GCCPUFraction   float64 // Fraction of CPU time spent in GC
}

// GetMemoryStats returns current memory statistics.
func GetMemoryStats() MemoryStats {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return MemoryStats{
		AllocBytes:      m.Alloc,
		TotalAllocBytes: m.TotalAlloc,
		SysBytes:        m.Sys,
		NumGC:           m.NumGC,
		GCCPUFraction:   m.GCCPUFraction,
	}
}

// String returns a formatted string representation of memory stats.
func (m MemoryStats) String() string {
	return fmt.Sprintf("Alloc: %d KB, Total: %d KB, Sys: %d KB, GC: %d (%.2f%% CPU)",
		m.AllocBytes/1024,
		m.TotalAllocBytes/1024,
		m.SysBytes/1024,
		m.NumGC,
		m.GCCPUFraction*100)
}

// BenchmarkResult holds the result of a benchmark run.
type BenchmarkResult struct {
	Name         string
	Duration     time.Duration
	MemoryBefore MemoryStats
	MemoryAfter  MemoryStats
	Iterations   int
	Error        error
}

// String returns a formatted string representation of the benchmark result.
func (br BenchmarkResult) String() string {
	if br.Error != nil {
		return fmt.Sprintf("%s: ERROR - %v", br.Name, br.Error)
	}

	memDiff := br.MemoryAfter.AllocBytes - br.MemoryBefore.AllocBytes
	avgDuration := br.Duration / time.Duration(br.Iterations)

	return fmt.Sprintf("%s: %d iterations, avg: %v, total: %v, mem: +%d KB",
		br.Name, br.Iterations, avgDuration, br.Duration, int64(memDiff)/1024) //nolint:gosec // G115: Safe conversion for memory display
}

// Benchmark represents a benchmark function.
type Benchmark struct {
	Name string
	Func func() error
}

// BenchmarkSuite manages multiple benchmarks.
type BenchmarkSuite struct {
	benchmarks []Benchmark
	results    []BenchmarkResult
	mu         sync.Mutex
}

// NewBenchmarkSuite creates a new benchmark suite.
func NewBenchmarkSuite() *BenchmarkSuite {
	return &BenchmarkSuite{
		benchmarks: make([]Benchmark, 0),
		results:    make([]BenchmarkResult, 0),
	}
}

// Add adds a benchmark to the suite.
func (bs *BenchmarkSuite) Add(name string, fn func() error) {
	bs.benchmarks = append(bs.benchmarks, Benchmark{
		Name: name,
		Func: fn,
	})
}

// Run runs a single benchmark with the specified number of iterations.
func (bs *BenchmarkSuite) Run(name string, iterations int) BenchmarkResult {
	var benchmark Benchmark
	found := false
	for _, b := range bs.benchmarks {
		if b.Name == name {
			benchmark = b
			found = true
			break
		}
	}

	if !found {
		return BenchmarkResult{
			Name:  name,
			Error: fmt.Errorf("benchmark '%s' not found", name),
		}
	}

	return bs.runBenchmark(benchmark, iterations)
}

// RunAll runs all benchmarks in the suite.
func (bs *BenchmarkSuite) RunAll(iterations int) []BenchmarkResult {
	bs.mu.Lock()
	defer bs.mu.Unlock()

	bs.results = make([]BenchmarkResult, 0, len(bs.benchmarks))

	for _, benchmark := range bs.benchmarks {
		result := bs.runBenchmark(benchmark, iterations)
		bs.results = append(bs.results, result)
	}

	return bs.results
}

// runBenchmark executes a single benchmark.
func (bs *BenchmarkSuite) runBenchmark(benchmark Benchmark, iterations int) BenchmarkResult {
	// Force garbage collection before measuring
	runtime.GC()
	memBefore := GetMemoryStats()

	timer := NewTimer(benchmark.Name)
	var err error

	for range iterations {
		if e := benchmark.Func(); e != nil {
			err = e
			break
		}
	}

	duration := timer.Stop()
	memAfter := GetMemoryStats()

	return BenchmarkResult{
		Name:         benchmark.Name,
		Duration:     duration,
		MemoryBefore: memBefore,
		MemoryAfter:  memAfter,
		Iterations:   iterations,
		Error:        err,
	}
}

// Results returns the last run results.
func (bs *BenchmarkSuite) Results() []BenchmarkResult {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	return bs.results
}

// PrintResults prints formatted benchmark results.
func (bs *BenchmarkSuite) PrintResults() {
	results := bs.Results()
	fmt.Println("\nBenchmark Results:")
	fmt.Println("==================")
	for _, result := range results {
		fmt.Println(result.String())
	}
	fmt.Println()
}

// OCRPipelineBenchmark provides specialized benchmarking for OCR operations.
type OCRPipelineBenchmark struct {
	*BenchmarkSuite
}

// NewOCRPipelineBenchmark creates a new OCR-specific benchmark suite.
func NewOCRPipelineBenchmark() *OCRPipelineBenchmark {
	return &OCRPipelineBenchmark{
		BenchmarkSuite: NewBenchmarkSuite(),
	}
}

// AddImageProcessingBenchmark adds an image processing benchmark.
func (ocr *OCRPipelineBenchmark) AddImageProcessingBenchmark(name string, fn func() error) {
	ocr.Add("ImageProcessing_"+name, fn)
}

// AddDetectionBenchmark adds a text detection benchmark.
func (ocr *OCRPipelineBenchmark) AddDetectionBenchmark(name string, fn func() error) {
	ocr.Add("Detection_"+name, fn)
}

// AddRecognitionBenchmark adds a text recognition benchmark.
func (ocr *OCRPipelineBenchmark) AddRecognitionBenchmark(name string, fn func() error) {
	ocr.Add("Recognition_"+name, fn)
}

// AddPipelineBenchmark adds an end-to-end pipeline benchmark.
func (ocr *OCRPipelineBenchmark) AddPipelineBenchmark(name string, fn func() error) {
	ocr.Add("Pipeline_"+name, fn)
}

// GPUVSCPUBenchmarkResult holds comparison results between GPU and CPU processing.
type GPUVSCPUBenchmarkResult struct {
	ImagePath     string
	ImageSize     string
	CPUResult     BenchmarkResult
	GPUResult     BenchmarkResult
	SpeedupFactor float64
	MemoryDiff    int64 // GPU memory usage - CPU memory usage (KB)
	GPUAvailable  bool
}

// String returns a formatted representation of the GPU vs CPU comparison.
func (r GPUVSCPUBenchmarkResult) String() string {
	if !r.GPUAvailable {
		return fmt.Sprintf("%s (%s): GPU not available, CPU only: %v",
			r.ImagePath, r.ImageSize, r.CPUResult.Duration)
	}

	speedupStr := "slower"
	if r.SpeedupFactor > 1.0 {
		speedupStr = fmt.Sprintf("%.2fx faster", r.SpeedupFactor)
	} else if r.SpeedupFactor < 1.0 {
		speedupStr = fmt.Sprintf("%.2fx slower", 1.0/r.SpeedupFactor)
	}

	return fmt.Sprintf("%s (%s): CPU: %v, GPU: %v (%s), Mem diff: %+d KB",
		r.ImagePath, r.ImageSize, r.CPUResult.Duration, r.GPUResult.Duration, speedupStr, r.MemoryDiff)
}

// GPUVSCPUBenchmark performs comprehensive GPU vs CPU performance comparison.
type GPUVSCPUBenchmark struct {
	modelsDir  string
	testImages []TestImage
	results    []GPUVSCPUBenchmarkResult
}

// TestImage represents a test image with metadata.
type TestImage struct {
	Path         string
	Description  string
	SizeCategory string
}

// NewGPUVSCPUBenchmark creates a new GPU vs CPU benchmark.
func NewGPUVSCPUBenchmark(modelsDir string) *GPUVSCPUBenchmark {
	return &GPUVSCPUBenchmark{
		modelsDir: modelsDir,
		testImages: []TestImage{
			{"testdata/images/simple_text.png", "Simple text", "Small"},
			{"testdata/images/english_text.png", "English text", "Small"},
			{"testdata/images/german_text.png", "German text", "Small"},
			{"testdata/images/complex_layout.png", "Complex layout", "Medium"},
		},
		results: make([]GPUVSCPUBenchmarkResult, 0),
	}
}

// AddTestImage adds a custom test image to the benchmark.
func (b *GPUVSCPUBenchmark) AddTestImage(path, description, sizeCategory string) {
	b.testImages = append(b.testImages, TestImage{
		Path:         path,
		Description:  description,
		SizeCategory: sizeCategory,
	})
}

// RunBenchmark executes the complete GPU vs CPU benchmark.
func (b *GPUVSCPUBenchmark) RunBenchmark(iterations int) ([]GPUVSCPUBenchmarkResult, error) {
	b.results = make([]GPUVSCPUBenchmarkResult, 0, len(b.testImages))

	for _, img := range b.testImages {
		fmt.Printf("Benchmarking: %s (%s)\n", img.Description, img.SizeCategory)

		result, err := b.benchmarkImage(img, iterations)
		if err != nil {
			fmt.Printf("  Error: %v\n", err)
			continue
		}

		b.results = append(b.results, result)
		fmt.Printf("  %s\n", result.String())
	}

	return b.results, nil
}

// benchmarkImage runs CPU and GPU benchmarks for a single image.
func (b *GPUVSCPUBenchmark) benchmarkImage(img TestImage, iterations int) (GPUVSCPUBenchmarkResult, error) {
	// Check if image exists
	imgPath := filepath.Join(".", img.Path)
	if !testutil.FileExists(imgPath) {
		return GPUVSCPUBenchmarkResult{}, fmt.Errorf("image not found: %s", imgPath)
	}

	// Get image size info
	sizeInfo, err := b.getImageSizeInfo(imgPath)
	if err != nil {
		sizeInfo = "unknown"
	}

	result := GPUVSCPUBenchmarkResult{
		ImagePath:    img.Path,
		ImageSize:    sizeInfo,
		GPUAvailable: true, // Will be set to false if GPU setup fails
	}

	// Benchmark CPU processing
	cpuResult, err := b.benchmarkCPU(imgPath, iterations)
	if err != nil {
		return result, fmt.Errorf("CPU benchmark failed: %w", err)
	}
	result.CPUResult = cpuResult

	// Benchmark GPU processing
	gpuResult, err := b.benchmarkGPU(imgPath, iterations)
	if err != nil {
		// GPU not available or failed
		result.GPUAvailable = false
		return result, err
	}
	result.GPUResult = gpuResult

	// Calculate performance metrics
	if result.GPUAvailable {
		result.SpeedupFactor = float64(cpuResult.Duration.Nanoseconds()) / float64(gpuResult.Duration.Nanoseconds())
		cpuDiff := cpuResult.MemoryAfter.AllocBytes - cpuResult.MemoryBefore.AllocBytes
		gpuDiff := gpuResult.MemoryAfter.AllocBytes - gpuResult.MemoryBefore.AllocBytes
		cpuMemDiff := int64(cpuDiff)
		if cpuDiff > math.MaxInt64 {
			cpuMemDiff = math.MaxInt64
		}
		gpuMemDiff := int64(gpuDiff)
		if gpuDiff > math.MaxInt64 {
			gpuMemDiff = math.MaxInt64
		}
		result.MemoryDiff = (gpuMemDiff - cpuMemDiff) / 1024 // Convert to KB
	}

	return result, nil
}

// benchmarkCPU runs CPU-only OCR benchmark.
func (b *GPUVSCPUBenchmark) benchmarkCPU(imagePath string, iterations int) (BenchmarkResult, error) {
	// Create CPU pipeline
	pipeline, err := pipeline.NewBuilder().
		WithModelsDir(b.modelsDir).
		Build()
	if err != nil {
		return BenchmarkResult{}, fmt.Errorf("failed to create CPU pipeline: %w", err)
	}
	defer func() { _ = pipeline.Close() }()

	// Load the image once for reuse
	img, err := testutil.LoadImageFile(imagePath)
	if err != nil {
		return BenchmarkResult{}, fmt.Errorf("failed to load image: %w", err)
	}

	// Warmup
	_, _ = pipeline.ProcessImage(img)

	// Create benchmark function
	benchmarkFunc := func() error {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		_, err := pipeline.ProcessImageContext(ctx, img)
		return err
	}

	// Run benchmark
	suite := NewBenchmarkSuite()
	suite.Add("CPU_OCR", benchmarkFunc)

	return suite.Run("CPU_OCR", iterations), nil
}

// benchmarkGPU runs GPU-accelerated OCR benchmark.
func (b *GPUVSCPUBenchmark) benchmarkGPU(imagePath string, iterations int) (BenchmarkResult, error) {
	// Create GPU pipeline
	pipeline, err := pipeline.NewBuilder().
		WithModelsDir(b.modelsDir).
		WithGPU(true).
		WithGPUDevice(0).
		WithGPUMemoryLimit(2 * 1024 * 1024 * 1024). // 2GB in bytes
		Build()
	if err != nil {
		return BenchmarkResult{}, fmt.Errorf("failed to create GPU pipeline: %w", err)
	}
	defer func() { _ = pipeline.Close() }()

	// Load the image once for reuse
	img, err := testutil.LoadImageFile(imagePath)
	if err != nil {
		return BenchmarkResult{}, fmt.Errorf("failed to load image: %w", err)
	}

	// Warmup
	_, _ = pipeline.ProcessImage(img)

	// Create benchmark function
	benchmarkFunc := func() error {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		_, err := pipeline.ProcessImageContext(ctx, img)
		return err
	}

	// Run benchmark
	suite := NewBenchmarkSuite()
	suite.Add("GPU_OCR", benchmarkFunc)

	return suite.Run("GPU_OCR", iterations), nil
}

// getImageSizeInfo returns formatted image size information.
func (b *GPUVSCPUBenchmark) getImageSizeInfo(imagePath string) (string, error) {
	img, err := testutil.LoadImageFile(imagePath)
	if err != nil {
		return "", err
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	megapixels := float64(width*height) / 1000000.0

	return fmt.Sprintf("%dx%d (%.1fMP)", width, height, megapixels), nil
}

// PrintDetailedResults prints comprehensive benchmark results.
func (b *GPUVSCPUBenchmark) PrintDetailedResults() {
	if len(b.results) == 0 {
		fmt.Println("No benchmark results available")
		return
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("GPU vs CPU OCR Performance Benchmark Results")
	fmt.Println(strings.Repeat("=", 80))

	// System info
	fmt.Printf("System Information:\n")
	fmt.Printf("  GOOS: %s\n", runtime.GOOS)
	fmt.Printf("  GOARCH: %s\n", runtime.GOARCH)
	fmt.Printf("  NumCPU: %d\n", runtime.NumCPU())
	fmt.Printf("  Go Version: %s\n", runtime.Version())
	fmt.Println()

	// Individual results
	fmt.Println("Individual Image Results:")
	fmt.Println(strings.Repeat("-", 50))
	for _, result := range b.results {
		fmt.Printf("• %s\n", result.String())
	}
	fmt.Println()

	// Summary statistics
	b.printSummaryStatistics()

	// Recommendations
	b.printRecommendations()
}

// printSummaryStatistics calculates and prints summary stats.
func (b *GPUVSCPUBenchmark) printSummaryStatistics() {
	var cpuTotalTime, gpuTotalTime time.Duration
	var speedups []float64
	var memoryDiffs []int64
	gpuSuccessCount := 0

	for _, result := range b.results {
		cpuTotalTime += result.CPUResult.Duration
		if result.GPUAvailable {
			gpuTotalTime += result.GPUResult.Duration
			speedups = append(speedups, result.SpeedupFactor)
			memoryDiffs = append(memoryDiffs, result.MemoryDiff)
			gpuSuccessCount++
		}
	}

	fmt.Println("Summary Statistics:")
	fmt.Println(strings.Repeat("-", 25))
	fmt.Printf("  Total CPU Time: %v\n", cpuTotalTime)
	if gpuSuccessCount > 0 {
		fmt.Printf("  Total GPU Time: %v\n", gpuTotalTime)
		fmt.Printf("  Overall Speedup: %.2fx\n", float64(cpuTotalTime.Nanoseconds())/float64(gpuTotalTime.Nanoseconds()))

		// Average speedup
		avgSpeedup := 0.0
		for _, s := range speedups {
			avgSpeedup += s
		}
		avgSpeedup /= float64(len(speedups))
		fmt.Printf("  Average Speedup: %.2fx\n", avgSpeedup)

		// Average memory difference
		avgMemDiff := int64(0)
		for _, m := range memoryDiffs {
			avgMemDiff += m
		}
		avgMemDiff /= int64(len(memoryDiffs))
		fmt.Printf("  Average Memory Diff: %+d KB\n", avgMemDiff)
	} else {
		fmt.Printf("  GPU: Not available or failed\n")
	}
	fmt.Printf("  GPU Success Rate: %d/%d (%.1f%%)\n",
		gpuSuccessCount, len(b.results), float64(gpuSuccessCount)*100.0/float64(len(b.results)))
	fmt.Println()
}

// printRecommendations provides usage recommendations based on results.
func (b *GPUVSCPUBenchmark) printRecommendations() {
	fmt.Println("Recommendations:")
	fmt.Println(strings.Repeat("-", 20))

	if len(b.results) == 0 {
		fmt.Println("  No results to analyze")
		return
	}

	gpuFasterCount := 0
	for _, result := range b.results {
		if result.GPUAvailable && result.SpeedupFactor > 1.0 {
			gpuFasterCount++
		}
	}

	switch {
	case gpuFasterCount == 0:
		fmt.Println("  • Use CPU processing for better performance")
		fmt.Println("  • GPU acceleration shows overhead for small images")
		fmt.Println("  • Consider GPU for batch processing or larger images")
	case gpuFasterCount < len(b.results)/2:
		fmt.Println("  • Mixed results: GPU benefits depend on image size/complexity")
		fmt.Println("  • Use CPU for small/simple images")
		fmt.Println("  • Use GPU for large/complex images or batch processing")
	default:
		fmt.Println("  • GPU acceleration recommended for most use cases")
		fmt.Println("  • Consistent performance improvements observed")
	}

	fmt.Println("  • Consider warmup costs for single image processing")
	fmt.Println("  • GPU memory limit may need tuning for large images")
	fmt.Println()
}

// GetResults returns the benchmark results.
func (b *GPUVSCPUBenchmark) GetResults() []GPUVSCPUBenchmarkResult {
	return b.results
}
