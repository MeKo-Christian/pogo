# POGO - Blazing Fast OCR Engine

**OCR pipeline engineered in Go. Extract text from images and PDFs at lightning speed.**

![Go](https://img.shields.io/badge/Go-1.25+-00ADD8?style=flat&logo=go)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat&logo=onnx)
![Performance](https://img.shields.io/badge/Performance-GPU%20Ready-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-Open%20Source-blue?style=flat)

## Why POGO?

**POGO delivers high-quality OCR performance with minimal overhead.** Built from the ground up in Go, it combines the accuracy of PaddleOCR models with the speed of ONNX Runtime inference.

### Core Capabilities

- **Lightning Fast**: ONNX Runtime + Go performance optimization
- **Precision Detection**: PaddleOCR DB-style text detection
- **Smart Recognition**: CTC-based text recognition with dictionary support
- **Auto-Correction**: Document orientation and rectification (UVDoc)
- **Batch Power**: Parallel processing with intelligent resource management
- **Production Ready**: CLI tool + HTTP server for any workflow

## Power Features

### Performance & Scale

- **Multi-Model Support**: Mobile & server variants for any use case
- **Parallel Processing**: Intelligent worker pools with resource caps
- **GPU Acceleration**: CUDA-powered inference where available
- **Batch Operations**: Process thousands of documents efficiently
- **Multi-Scale Detection**: Optional image pyramid with IoU-based result fusion

### Document Intelligence

- **PDF Mastery**: Full PDF extraction + OCR pipeline via pdfcpu
- **Smart Orientation**: Auto-detect document rotation (0°/90°/180°/270°)
- **Line-Level Correction**: Per-text-line skew correction
- **Auto-Rectification**: Advanced page quad detection + homography warping

### Output Excellence

- **Multiple Formats**: Plain text, JSON, CSV - choose your weapon
- **Flexible Integration**: CLI for automation, HTTP server for applications
- **Language Support**: Configurable dictionaries and multi-language detection
- **Debug Visualization**: Optional overlay and debug image generation

### Enterprise Ready

- **Battle-Tested**: Comprehensive unit tests across all components
- **Quality Gates**: Smart rectification with automatic quality validation
- **Configurable**: Fine-tune models, thresholds, and processing parameters

> **Deep Dive**: Check out `GOAL.md` for project vision and `PLAN.md` for development roadmap

## Quick Start

### Requirements

- **Go 1.25+** - Modern Go for optimal performance
- **ONNX Runtime** - CPU or CUDA acceleration
- **Pre-trained Models** - Placed under `models/` directory

### Lightning Setup

**One-command installation:**

```bash
# Install ONNX Runtime
./scripts/setup-onnxruntime.sh

# Enable auto-environment (set once, forget forever)
direnv allow
```

**Smart Environment**: POGO uses direnv for zero-config development - all environment variables (`CGO_CFLAGS`, `CGO_LDFLAGS`, `LD_LIBRARY_PATH`) are auto-configured when you enter the directory!

**Manual Setup** (if needed):

```bash
source scripts/setup-env.sh
```

> **Pro Tip**: Linux x64 users get ONNX Runtime bundled in `onnxruntime/` - just run and go!

## AI Models Arsenal

**Organized model hierarchy under `models/`:**

```plain
models/
├── detection/
│   ├── mobile/     → PP-OCRv5_mobile_det.onnx    (fast, efficient)
│   └── server/     → PP-OCRv5_server_det.onnx    (high accuracy)
├── recognition/
│   ├── mobile/     → PP-OCRv5_mobile_rec.onnx    (lightweight)
│   └── server/     → PP-OCRv5_server_rec.onnx    (precision)
├── layout/
│   ├── pplcnet_x1_0_doc_ori.onnx       (document orientation)
│   ├── pplcnet_x0_25_textline_ori.onnx (textline - fast)
│   ├── pplcnet_x1_0_textline_ori.onnx  (textline - accurate)
│   └── uvdoc.onnx                      (rectification)
└── dictionaries/
    └── ppocr_keys_v1.txt               (default dictionary)
```

**Custom Models**: Override any path with flags or set `GO_OAR_OCR_MODELS_DIR`. The intelligent path resolver in `internal/models/paths.go` handles both organized trees and flat legacy layouts.

## Build & Deploy

### Lightning Commands (with `just`)

```bash
# Production Build
just build                  # → bin/pogo (optimized + version info)

# Development Speed
just build-dev              # → Fast local build
just run -- image doc.jpg   # → Run from source instantly

# Quality Assurance
just test                   # → Full test suite
just test-coverage          # → Generate coverage.html report
just fmt                    # → Auto-format (treefmt + gofumpt + gci)
just lint                   # → Comprehensive linting
just lint-fix              # → Auto-fix lint issues
```

### Traditional Build

```bash
# Direct Go commands
go build -o bin/pogo ./cmd/ocr
go test -v ./...
```

> **Pro Tip**: Use `just` for the ultimate developer experience with optimized build flags and integrated tooling!

### Barcode (Optional)

Barcode detection is optional and disabled by default to keep builds lean. To enable decoding via the pure-Go ZXing port (gozxing):

```bash
# Add dependency (once)
go get github.com/makiuchi-d/gozxing@latest

# Build with barcode support enabled
just build-dev -- -tags=barcode_gozxing
# or
go build -tags=barcode_gozxing -o bin/pogo ./cmd/ocr
```

This enables the internal `barcode` package backed by `gozxing`. The interface is pluggable for future backends.

## CLI Power User Guide

### Instant Results

```bash
# Quick Start Commands
pogo test                              # → Self-test system health
pogo image input.jpg --format json    # → Single image OCR
pogo batch images/*.png --format text # → Batch processing
pogo pdf scan.pdf --format json       # → Full PDF extraction
```

### Advanced Configuration

**Model Control:**

- `--models-dir <dir>` → Custom model directory

**Detection Tuning:**

- `--det-model <path>` → Custom detector model
- `--confidence <0..1>` → Detection confidence threshold
- `--det-polygon-mode minrect|contour` → Polygon extraction mode
- `--det-multiscale` → Enable multi-scale detection (image pyramid)
- `--det-scales 1.0,0.75,0.5` → Relative scales used when multi-scale is enabled
- `--det-merge-iou <0..1>` → IoU threshold for merging detections across scales
  (Advanced) Adaptive pyramid controls:
  - `--det-ms-adaptive` → Auto-generate scales based on image size
  - `--det-ms-max-levels <n>` → Max pyramid levels when adaptive (default: 3)
  - `--det-ms-min-side <px>` → Stop when min(image side × scale) <= this value (default: 320)
  - `--det-ms-incremental-merge` → Merge after each scale to reduce memory (default: true)

**Recognition Power:**

- `--rec-model <path>` → Custom recognizer model
- `--rec-height <32|48>` → Input height optimization
- `--dict <paths,comma>` → Custom dictionaries
- `--dict-langs <en,de,...>` → Language-specific processing

**Intelligence Features:**

- `--detect-orientation` → Auto document rotation
- `--orientation-threshold <0..1>` → Orientation confidence
- `--detect-textline` → Per-line skew correction
- `--textline-threshold <0..1>` → Textline confidence

**Rectification (Experimental):**

- `--rectify` → Enable page rectification
- `--rectify-model <path>` → Custom rectification model
- `--rectify-mask-threshold <0..1>` → Mask sensitivity
- `--rectify-height <pixels>` → Processing height
- `--rectify-debug-dir <dir>` → Debug visualization export

**Output Mastery:**

- `--format text|json|csv` → Choose your format
- `--output <file>` → Save to file
- `--overlay-dir <dir>` → Visual debugging overlays

**Debugging:**

- `--log-level debug|info|warn|error` → Logging verbosity
- `--verbose` → Full debug output (alias for `--log-level=debug`)

### Real-World Examples

```bash
# Single Image → Perfect JSON
pogo image doc.jpg --format json --detect-orientation --rectify

# Batch Processing Powerhouse
pogo batch images/ --recursive \
  --detect-orientation --rectify \
  --overlay-dir .tmp/overlay --rectify-debug-dir .tmp/rectify

# Multi-Language PDF Processing
pogo pdf scan.pdf --format json \
  --pages 1-5 --rectify \
  --dict-langs en,de

# Multi-Scale Detection (improved small text sensitivity)
pogo image doc.jpg --det-multiscale --det-scales 1.0,0.8,0.6 --format json
pogo pdf scan.pdf --det-multiscale --det-merge-iou 0.35 --format text
```

### Debug Visualization

When using `--rectify-debug-dir`, POGO generates these debug artifacts:

- `rect_mask_<ts>.png` → UVDoc mask heatmap with threshold visualization
- `rect_overlay_<ts>.png` → Original image with detected page quad overlay
- `rect_compare_<ts>.png` → Before/after comparison (original+quad vs rectified)

> **Smart Quality Gates**: Rectification only applies when quality metrics pass (mask coverage, area ratio, aspect bounds) to prevent harmful transformations.

## HTTP Server - Production Ready

### Launch Your OCR Service

```bash
# Start production server
pogo serve --port 8080 --language en --detect-orientation
```

### API Endpoints

| Endpoint     | Method | Purpose                             |
| ------------ | ------ | ----------------------------------- |
| `/ocr/image` | POST   | Process uploaded images (multipart) |
| `/ocr/pdf`   | POST   | Extract text from PDF files         |
| `/health`    | GET    | System health check                 |
| `/models`    | GET    | List available AI models            |

> **Server Configuration**: All CLI pipeline flags work identically (det/rec models, orientation, textline, multi-scale). Visual overlays supported in responses.
> Includes multi-scale detection flags: `--det-multiscale`, `--det-scales`, `--det-merge-iou`.
> Adaptive scaling: `--det-ms-adaptive`, `--det-ms-max-levels`, `--det-ms-min-side`. Memory: `--det-ms-incremental-merge`.

Barcode options (server API per-request overrides):
- `barcodes=true|false` → enable barcode detection stage
- `barcode-types=qr,ean13,code128,...` → restrict symbologies
- `barcode-min-size=<pixels>` → size hint to skip tiny candidates
- `barcode-dpi=<int>` → target DPI for page scaling (enhanced path; default 150)

PDF barcode mapping
- PDF responses include `page_box` for each barcode, in page coordinates (points) with origin at bottom-left.
- The server uses DPI heuristics (target ~150 DPI) to improve barcode decode robustness on page images.

OpenAPI
- See `docs/openapi.yaml` for a machine-readable API specification (OpenAPI 3.0).

### API Examples

```bash
# Health Check
curl -s http://localhost:8080/health | jq

# Model Status
curl -s http://localhost:8080/models | jq

# Image OCR → JSON
curl -s -F image=@doc.jpg http://localhost:8080/ocr/image | jq

# Enable multi-scale via server start flags
# pogo serve --det-multiscale --det-scales 1.0,0.75,0.5 --det-merge-iou 0.3
# Use adaptive multi-scale with incremental merge (default on)
# pogo serve --det-multiscale --det-ms-adaptive --det-ms-max-levels 4 --det-ms-min-side 320

## How Adaptive Multi-Scale Works

POGO can auto-generate pyramid scales per image to improve small-text detection while controlling memory:

- Always includes scale 1.0, then adds smaller scales until reaching `--det-ms-max-levels` or `--det-ms-min-side`.
- Performs per-scale detection, maps regions back to original coordinates, and merges across scales using NMS.
- With `--det-ms-incremental-merge` (default), performs a merge after each scale to bound memory usage.

See docs/multiscale.md for details and tuning guidance.

# Image OCR → Plain Text
curl -s -F image=@doc.jpg -F format=text http://localhost:8080/ocr/image

# Image OCR → Visual Overlay
curl -s -o overlay.png -F image=@doc.jpg -F format=overlay \
  -F box=#FF0000 -F poly=#00FF00 http://localhost:8080/ocr/image

# PDF OCR → JSON (with page selection)
curl -s -F pdf=@scan.pdf -F pages=1-3 http://localhost:8080/ocr/pdf | jq

# PDF OCR → Plain Text
curl -s -F pdf=@scan.pdf -F format=text http://localhost:8080/ocr/pdf
```

## Docker Deployment - Container Ready

### Quick Start with Docker

**Single Command Deployment:**

```bash
# Build and run with docker-compose
cd deployment/
docker-compose up --build

# Access your OCR service
curl -s -F image=@../testdata/images/simple_text.png http://localhost:8080/ocr/image | jq
```

### Docker Configuration

**Basic Usage:**

```bash
# Build the image
docker build -t pogo-ocr -f deployment/Dockerfile .

# Run the container
docker run -p 8080:8080 pogo-ocr serve --host 0.0.0.0
```

**Production Deployment:**

All deployment files are organized in the `deployment/` directory:

```
deployment/
├── Dockerfile          # Multi-stage Docker build
├── docker-compose.yml  # Production configuration
├── nginx.conf          # Reverse proxy setup
└── README.md           # Deployment guide
```

```yaml
# deployment/docker-compose.yml excerpt
services:
  pogo-ocr:
    build:
      context: ..
      dockerfile: deployment/Dockerfile
    ports:
      - "8080:8080"
    environment:
      - POGO_SERVER_HOST=0.0.0.0
      - POGO_MODELS_DIR=/usr/share/pogo/models
      - POGO_LOG_LEVEL=info
    volumes:
      # Optional: Custom models
      - ../custom-models:/usr/share/pogo/models:ro
    restart: unless-stopped
```

**Environment Variables:**

| Variable                            | Default                  | Description          |
| ----------------------------------- | ------------------------ | -------------------- |
| `POGO_SERVER_HOST`                  | `0.0.0.0`                | Server bind address  |
| `POGO_SERVER_PORT`                  | `8080`                   | Server port          |
| `POGO_MODELS_DIR`                   | `/usr/share/pogo/models` | Models directory     |
| `POGO_LOG_LEVEL`                    | `info`                   | Logging level        |
| `POGO_PIPELINE_RECOGNIZER_LANGUAGE` | `en`                     | Recognition language |

**Custom Models:**
Mount your custom models directory to override the built-in models:

```bash
docker run -p 8080:8080 \
  -v ./my-models:/usr/share/pogo/models:ro \
  pogo-ocr serve --host 0.0.0.0
```

**Health & Monitoring:**
Built-in health checks and resource management ensure production reliability:

```bash
# Check container health
docker ps  # Look for "healthy" status

# View logs (from deployment/ directory)
docker-compose logs pogo-ocr

# Scale for high availability
docker-compose up --scale pogo-ocr=3
```

> **Pro Tip**: Use the included nginx profile (`docker-compose --profile proxy up`) for load balancing and SSL termination!

## Project Status - Battle-Tested & Ready

### Mission Complete

**Core OCR Engine:**

- ✓ **ONNX Runtime Integration** - Blazing fast detector & recognizer
- ✓ **Smart Orientation** - Document + per-text-line classifiers
- ✓ **Auto-Rectification** - UVDoc with quality gating & homography warping
- ✓ **Production Pipeline** - Full batch & PDF processing workflows

**Interface Excellence:**

- ✓ **CLI Mastery** - Complete command-line interface
- ✓ **HTTP Server** - Production-ready API endpoints
- ✓ **Comprehensive Testing** - Utils, ONNX setup, pipeline validation

> **Roadmap**: See `PLAN.md` for detailed progress tracking and upcoming enhancements!

## Developer Experience

### Essential Commands

```bash
just fmt             # Auto-format (treefmt + gofumpt + gci)
just lint            # Comprehensive linting (golangci-lint)
just test            # Full test suite
just test-coverage   # Generate coverage reports
```

**GPU Acceleration**: Configure CUDA providers in ONNX Runtime and use `--gpu` flags for maximum performance!

### Architecture Overview

```
cmd/ocr/          # Cobra CLI interface
internal/
├── detector/     # ONNX text detection + post-processing
├── recognizer/   # ONNX text recognition + CTC decoding
├── orientation/  # Document & textline orientation classifiers
├── rectify/      # Advanced document rectification (UVDoc)
├── pipeline/     # Orchestration + parallel processing + results
├── pdf/          # PDF image extraction engine
├── server/       # Production HTTP server
└── utils/        # Image processing, tensors, geometry utilities
models/           # AI model arsenal (see Models section)
scripts/          # ONNX Runtime setup automation
```

## Troubleshooting Guide

### ONNX Runtime Issues

**Problem**: `ONNX Runtime not found`
**Solution**:

```bash
./scripts/setup-onnxruntime.sh && source scripts/setup-env.sh
```

> **Linux users**: `onnxruntime/lib` is auto-detected!

### Model Loading Issues

**Problem**: `Model not found`
**Solution**: Verify files in `models/` or use `--models-dir` override

### Empty Recognition Results

**Problem**: `Recognition returns empty text`
**Solution**: Check dictionary paths and language cleaning rules

### Rectification Not Working

**Problem**: `Rectification not applied`
**Solution**: Enable debug mode with `--rectify-debug-dir` and adjust `--rectify-mask-threshold`

---

## Credits & Inspiration

**Built on the shoulders of giants:**

- **OAR-OCR & PaddleOCR** - Pioneering OCR model architectures
- **pdfcpu** - Blazing fast PDF image extraction engine

---

### Ready to extract text at lightning speed? Let's go!

```bash
# One command to rule them all
just run -- image your-document.jpg --format json --detect-orientation
```
