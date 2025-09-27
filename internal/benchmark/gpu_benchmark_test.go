package benchmark

import (
	"context"
	"testing"
	"time"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/MeKo-Tech/pogo/internal/testutil"
)

// Benchmark test functions for Go testing framework.
func BenchmarkOCR_CPU_Simple(b *testing.B) {
	benchmarkOCRMode(b, "testdata/images/simple_text.png", false)
}

func BenchmarkOCR_GPU_Simple(b *testing.B) {
	benchmarkOCRMode(b, "testdata/images/simple_text.png", true)
}

func BenchmarkOCR_CPU_Complex(b *testing.B) {
	benchmarkOCRMode(b, "testdata/images/complex_layout.png", false)
}

func BenchmarkOCR_GPU_Complex(b *testing.B) {
	benchmarkOCRMode(b, "testdata/images/complex_layout.png", true)
}

// benchmarkOCRMode is a helper for Go benchmark tests.
func benchmarkOCRMode(b *testing.B, imagePath string, useGPU bool) {
	b.Helper()

	if !testutil.FileExists(imagePath) {
		b.Skipf("Test image not found: %s", imagePath)
	}

	// Load the image once
	img, err := testutil.LoadImageFile(imagePath)
	if err != nil {
		b.Fatalf("Failed to load image %s: %v", imagePath, err)
	}

	// Create pipeline
	builder := pipeline.NewBuilder().
		WithDetectorModelPath("").
		WithRecognizerModelPath("").
		WithModelsDir("models")

	if useGPU {
		// 2GB in bytes
		gpuMemLimit := uint64(2 * 1024 * 1024 * 1024)
		builder = builder.WithGPU(true).WithGPUDevice(0).WithGPUMemoryLimit(gpuMemLimit)
	}

	p, err := builder.Build()
	if err != nil {
		b.Fatalf("Failed to create pipeline: %v", err)
	}
	defer p.Close()

	// Warmup
	_, _ = p.ProcessImage(img)

	b.ResetTimer()
	for range b.N {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		_, err := p.ProcessImageContext(ctx, img)
		cancel()
		if err != nil {
			b.Fatalf("OCR processing failed: %v", err)
		}
	}
}
