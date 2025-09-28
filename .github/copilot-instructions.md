# AI Coding Agent Instructions for pogo

## Project Overview

This is a high-performance OCR (Optical Character Recognition) pipeline in Go that reimplements OAR-OCR functionality. It processes images and PDFs through three main stages:

1. **Text Detection** (`internal/detector/`) - Uses PaddleOCR DB-style models to locate text regions
2. **Text Recognition** (`internal/recognizer/`) - CTC-based recognition with dictionary support
3. **Optional Processing** - Orientation classification and document rectification (UVDoc)

The pipeline supports both CLI (`cmd/ocr/`) and HTTP server modes with ONNX Runtime for inference.

## Architecture Patterns

### Pipeline Builder Pattern
Use the fluent builder pattern for pipeline configuration. Always start with `pipeline.NewBuilder()` and chain configuration methods:

```go
pipeline := pipeline.NewBuilder().
    WithModelsDir("/path/to/models").
    WithDetectorModelPath("custom_det.onnx").
    WithDictionaryPaths([]string{"en.txt", "de.txt"}).
    Build()
```

### Model Path Resolution
Models are organized hierarchically under `models/` directory. Use `internal/models/paths.go` functions for path resolution:

```go
// Get paths using the centralized resolver
detPath := models.GetDetectionModelPath(modelsDir, useServerModel)
recPath := models.GetRecognitionModelPath(modelsDir, useServerModel)
dictPath := models.GetDictionaryPath(modelsDir, models.DictionaryPPOCRKeysV1)
```

### ONNX Runtime Integration
ONNX models require proper environment setup. The project uses direnv for automatic CGO configuration:

```go
// GPU configuration follows this pattern
gpuConfig := onnx.GPUConfig{
    Enabled: true,
    DeviceID: 0,
    MemLimit: 1 << 30, // 1GB
}
```

### Error Handling
Follow Go idioms with structured errors. Use `fmt.Errorf` with `%w` verb for error wrapping:

```go
if err := validateInput(image); err != nil {
    return fmt.Errorf("input validation failed: %w", err)
}
```

## Development Workflow

### Build Commands
```bash
just build          # Production build with version info
just build-dev      # Fast development build
just run -- <args>  # Run from source (e.g., just run image input.jpg)
```

### Testing Strategy
```bash
just test                    # All tests
just test-unit              # Unit tests only (-short)
just test-integration       # Integration tests
just test-benchmark         # Performance benchmarks
just test-onnx              # ONNX Runtime smoke tests
```

### Code Quality
```bash
just fmt           # Format with treefmt (gofumpt, gci, prettier)
just lint          # golangci-lint (85+ rules enabled)
just check         # All quality checks
```

## Critical Setup Requirements

### ONNX Runtime Environment
The project requires ONNX Runtime. Use direnv for automatic setup:

```bash
direnv allow  # Enables automatic CGO_CFLAGS/CGO_LDFLAGS/LD_LIBRARY_PATH
```

Without direnv, manually set:
```bash
export CGO_CFLAGS="-I$(pwd)/onnxruntime/include"
export CGO_LDFLAGS="-L$(pwd)/onnxruntime/lib -lonnxruntime"
export LD_LIBRARY_PATH="$(pwd)/onnxruntime/lib:$LD_LIBRARY_PATH"
```

### Model Organization
Models must be placed in `models/` directory with this structure:
```
models/
├── detection/
│   ├── mobile/PP-OCRv5_mobile_det.onnx
│   └── server/PP-OCRv5_server_det.onnx
├── recognition/
│   ├── mobile/PP-OCRv5_mobile_rec.onnx
│   └── server/PP-OCRv5_server_rec.onnx
├── layout/
│   ├── pplcnet_x1_0_doc_ori.onnx
│   └── ...
└── dictionaries/
    └── ppocr_keys_v1.txt
```

## Component-Specific Patterns

### CLI Commands (`cmd/ocr/cmd/`)
- Use Cobra framework with subcommands
- Global config loaded via `internal/config`
- Follow existing flag patterns (e.g., `--models-dir`, `--format json|text|csv`)

### Detector (`internal/detector/`)
- Processes images through DB thresholding and polygon extraction
- Supports both `minrect` and `contour` polygon modes
- NMS (Non-Maximum Suppression) with adaptive thresholds

### Recognizer (`internal/recognizer/`)
- CTC decoding with character dictionary lookup
- Supports multiple dictionaries for different languages
- Image preprocessing with width padding to multiples of 8

### Pipeline (`internal/pipeline/`)
- Orchestrates detector → orientation → recognizer → rectification
- Parallel processing with worker pools and resource limits
- Results aggregation with confidence scoring

### ONNX Integration (`internal/onnx/`)
- Tensor creation/destruction with proper cleanup
- Session management with GPU support
- Memory pool patterns for float32 tensors

## Testing Patterns

### Unit Tests
Place `*_test.go` files alongside implementation. Use table-driven tests:

```go
func TestProcessImage(t *testing.T) {
    tests := []struct {
        name     string
        input    image.Image
        expected string
        wantErr  bool
    }{
        {"valid image", testImage, "expected text", false},
        {"nil image", nil, "", true},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result, err := processImage(tt.input)
            if tt.wantErr {
                assert.Error(t, err)
            } else {
                assert.NoError(t, err)
                assert.Equal(t, tt.expected, result)
            }
        })
    }
}
```

### Integration Tests
Use `test/integration/` for full pipeline tests. Mock external dependencies when possible.

## Code Style Conventions

### Package Organization
- `internal/` for private packages (Go convention)
- Short, lowercase package names (e.g., `detector`, not `textdetector`)
- CLI wiring only in `cmd/`, business logic in `internal/`

### Function Length
- Soft 120-char line limit
- Avoid functions >50 lines (enforced by `funlen` linter)
- Break down complex functions into smaller helpers

### Documentation
- All exported functions/types need doc comments
- Use complete sentences starting with the name being documented
- Reference related types/functions where relevant

### Imports
- Grouped and sorted by `gci` (standard → internal → external)
- No unused imports (enforced by `goimports`)

## Common Pitfalls

### ONNX Runtime Issues
- Always check `LD_LIBRARY_PATH` when runtime fails
- GPU support requires CUDA providers to be enabled at session creation
- Tensor shapes must match model expectations exactly

### Memory Management
- ONNX tensors require explicit destruction
- Use memory pools (`internal/mempool/`) for float32 arrays
- Watch for goroutine leaks in parallel processing

### Model Path Resolution
- Use `internal/models/paths.go` functions instead of string concatenation
- Handle both flat and hierarchical model directory layouts
- Validate model files exist before creating ONNX sessions

### Pipeline Configuration
- Always call `UpdateModelPath()` after changing `ModelsDir`
- Builder methods are chainable but not idempotent
- Validate configuration before pipeline creation

## Performance Considerations

- Use server models for production (higher accuracy, slower)
- Mobile models for real-time applications (faster, slightly less accurate)
- GPU acceleration when available (significant speedup for large batches)
- Parallel processing scales with CPU cores but watch memory usage

## Debugging Tips

- Enable debug logging with `--log-level debug`
- Use `--overlay-dir` to visualize detection results
- Rectification debug images save to `--rectify-debug-dir`
- Benchmark tests help identify performance bottlenecks

Reference files for patterns:
- `internal/pipeline/pipeline.go` - Builder pattern and configuration
- `internal/models/paths.go` - Model path resolution
- `internal/detector/detector.go` - ONNX session management
- `cmd/ocr/cmd/root.go` - CLI structure and global config</content>
<parameter name="filePath">/home/christian/Code/pogo/.github/copilot-instructions.md