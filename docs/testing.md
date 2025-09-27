# Testing Framework Documentation

This document describes the comprehensive testing framework set up for the pogo project as part of Phase 1.4 of the development plan.

## Overview

The testing framework provides:

- **Synthetic test image generation** for consistent OCR testing
- **Comprehensive test utilities** for common testing operations
- **Benchmark framework** for performance testing
- **ONNX Runtime smoke tests** for dependency verification
- **Test data management** with automated generation and fixtures
- **CLI command testing** infrastructure

## Test Structure

```
testdata/
├── images/
│   ├── simple/           # Single-word test images
│   ├── multiline/        # Multi-line text documents
│   ├── rotated/          # Rotated text samples (0°, 90°, 180°, 270°, ±45°)
│   └── scanned/          # Simulated scanned documents with noise
├── synthetic/            # Generated test images
├── fixtures/             # Test fixtures with expected results
├── documents/            # Test PDF documents
└── manifest.json         # Test data manifest
```

## Testing Packages

### internal/testutil

Core testing utilities providing:

- **Project navigation**: `GetProjectRoot()`, `GetTestDataDir()`
- **Image generation**: `GenerateTextImage()`, synthetic test image creation
- **Image utilities**: Save, load, and compare images
- **Fixtures management**: Load and save test fixtures with expected results
- **File utilities**: Directory creation, file existence checks

**Coverage**: 96.8%

#### Key Functions

```go
// Generate synthetic test images
config := testutil.DefaultTestImageConfig()
config.Text = "Hello World"
img, err := testutil.GenerateTextImage(config)

// Save and load images
testutil.SaveImage(t, img, "testdata/images/test.png")
loadedImg := testutil.LoadImage(t, "testdata/images/test.png")

// Load test fixtures
fixture := testutil.LoadFixture(t, "simple_hello")
```

### internal/benchmark

Performance testing framework providing:

- **Timer utilities**: High-precision timing for operations
- **Memory tracking**: Monitor memory allocation and GC
- **Benchmark suites**: Organize and run multiple benchmarks
- **OCR-specific benchmarks**: Specialized for OCR pipeline testing

**Coverage**: 89.7%

#### Key Features

```go
// Create benchmark suite
suite := benchmark.NewBenchmarkSuite()
suite.Add("test_operation", func() error {
    // Your operation here
    return nil
})

// Run benchmarks
results := suite.RunAll(100) // 100 iterations
suite.PrintResults()

// OCR-specific benchmark suite
ocrBench := benchmark.NewOCRPipelineBenchmark()
ocrBench.AddDetectionBenchmark("text_detection", detectionFunc)
ocrBench.AddRecognitionBenchmark("text_recognition", recognitionFunc)
```

### internal/onnx

ONNX Runtime testing and verification:

- **Library detection**: Automatic ONNX Runtime library discovery
- **Smoke tests**: Basic functionality verification
- **Environment validation**: CGO and library path testing

**Coverage**: 56.1% (higher when ONNX Runtime is installed)

#### Mocking ONNX Outputs (No Runtime Required)

For unit tests that should not depend on ONNX Runtime, use the synthetic generators under `internal/onnx/mock`:

- `mock.NewUniformMap(w, h, value)` produces a flat probability map for detection post-processing tests.
- `mock.NewCenteredBlobMap(w, h, peak, sigma)` produces a Gaussian-like region to simulate text heatmaps.
- `mock.NewTextStripeMap(w, h, lineHeight, gap, hi, lo)` simulates line-like activations.
- `mock.NewGreedyPathLogits(indices, classes, classesFirst, high, low)` builds logits matching a target greedy path for recognition decoding tests (works directly with `DecodeCTCGreedy`).

These helpers allow testing detection post-processing and recognition decoding logic deterministically without initializing ONNX Runtime or loading models.

## Test Data Generation

### Automatic Generation

Run tests to generate synthetic test data:

```bash
# Using Go tests
go test ./internal/testutil -run TestGenerateTestImages

# Using dedicated program
just test-data-generate

# Using shell script
./scripts/download-test-data.sh --synthetic
```

### Generated Test Images

1. **Simple Images**: Single words (Hello, World, OCR, Test, 123, Sample)
2. **Multiline Images**: Multi-line text documents
3. **Rotated Images**: Text at various angles (0°, 90°, 180°, 270°, ±45°)
4. **Scanned Images**: Simulated scanned documents with noise

### Test Fixtures

JSON fixtures with expected OCR results:

```json
{
  "name": "simple_hello",
  "description": "Simple single word 'Hello' detection and recognition",
  "input_file": "images/simple/simple_1_Hello.png",
  "expected": {
    "text_regions": [
      {
        "text": "Hello",
        "confidence": 0.95,
        "bounding_box": {
          "x": 130,
          "y": 115,
          "width": 60,
          "height": 15
        }
      }
    ],
    "full_text": "Hello",
    "confidence": 0.95
  }
}
```

## Available Test Commands

### Basic Testing

```bash
# Run all tests
just test

# Run with coverage
just test-coverage

# Run unit tests only
just test-unit

# Run integration tests
just test-integration

# Run with race detection
just test-race
```

### Specialized Testing

```bash
# Run benchmarks
just test-benchmark

# Run ONNX Runtime tests
just test-onnx

# Run all test variants
just test-all
```

### Test Data Management

```bash
# Generate synthetic test data
just test-data-generate

# Setup all test data (generate + download)
just test-data

# Clean test data
just test-data-clean
```

## CLI Command Testing

Test infrastructure for CLI commands with utilities for:

- Command execution with output capture
- Flag validation
- Help text verification
- Error handling validation

Example:

```go
func TestImageCommand(t *testing.T) {
    output, err := executeCommandAndCaptureOutput(t, imageCmd, []string{"--help"})
    require.NoError(t, err)
    assert.Contains(t, output, "Process images")
}
```

## Test Coverage Goals

- **Unit tests**: >90% coverage for core components
- **Integration tests**: >80% coverage for workflows
- **Benchmark tests**: Performance regression detection
- **Smoke tests**: Dependency verification

## Current Status

✅ **Completed (Phase 1.4)**:

- Comprehensive test directory structure
- Synthetic test image generation
- Test utility packages (96.8% coverage)
- Benchmark framework (89.7% coverage)
- ONNX Runtime smoke tests
- CLI command testing infrastructure
- Test data management scripts
- Automated test data generation

## Integration with CI/CD

The testing framework integrates with the existing CI/CD pipeline:

- Tests run automatically on push/PR
- Code coverage reporting
- Lint and format checking
- Build verification

## Best Practices

1. **Use synthetic data**: Consistent, reproducible test results
2. **Test fixtures**: Define expected outputs for validation
3. **Benchmark regularly**: Track performance over time
4. **Mock dependencies**: Use test doubles for external dependencies
5. **Clean up**: Properly manage test resources and temporary files

## Future Enhancements

As OCR functionality is implemented:

1. Add model-specific test fixtures
2. Implement accuracy benchmarks against reference implementations
3. Add stress testing for large document processing
4. Create visual regression tests for UI components
5. Add integration tests with real OCR models

## Usage Examples

### Running Specific Tests

```bash
# Test specific package
go test ./internal/testutil -v

# Test with pattern
go test ./... -run TestImage

# Benchmark specific functions
go test ./internal/benchmark -bench=BenchmarkTimer
```

### Generating Test Data

```bash
# Generate all test data
just test-data

# Generate only synthetic images
./scripts/download-test-data.sh --synthetic

# Clean and regenerate
just test-data-clean
just test-data
```

### Custom Test Development

```go
func TestCustomOCROperation(t *testing.T) {
    // Load test image
    img := testutil.LoadImage(t, "testdata/images/simple/simple_1_Hello.png")

    // Process with your OCR pipeline
    result, err := yourOCRFunction(img)
    require.NoError(t, err)

    // Load expected results
    fixture := testutil.LoadFixture(t, "simple_hello")
    expected := fixture.Expected.(testutil.OCRExpectedResult)

    // Validate results
    assert.Equal(t, expected.FullText, result.Text)
    assert.GreaterOrEqual(t, result.Confidence, expected.Confidence-0.05)
}
```

This testing framework provides a solid foundation for developing and validating the OCR functionality throughout the development process.
