# pogo

Reimplementation of the OAR-OCR pipeline in Go, focused on inference with pre‑trained ONNX models. Ships a fast CLI for images and PDFs and an HTTP server mode. Uses ONNX Runtime via cgo and high‑quality image processing (imaging).

This project integrates three main stages:

- Text detection (PaddleOCR DB-style detector)
- Text recognition (CTC-based recognizer with dictionary support)
- Optional orientation and document rectification (UVDoc)

It supports single images, batches, and PDFs (image extraction + OCR), and can run as a long‑running HTTP service.

## Features

- Detector and recognizer with mobile and server model variants
- Batch processing with parallel workers and resource caps
- PDF image extraction via pdfcpu and full OCR pipeline on the extracted images
- Orientation classifiers:
  - Whole-document orientation (0/90/180/270)
  - Per‑text‑line orientation for skewed lines
- Document rectification (experimental): mask-based page quad + homography warp, with quality gating, optional debug image dumps
- Multiple outputs: plain text, JSON, and CSV
- CLI and HTTP server with configurable models, thresholds, and language/dictionary options
- GPU support (ONNX Runtime CUDA providers) where available
- Extensive unit tests for core utilities, detector/recognizer setup, pipeline wiring, and CLI ergonomics

See GOAL.md for the original long-form project rationale and background. See PLAN.md for the development plan and progress tracking (Phase 6.3 rectification implemented).

## Requirements

- Go 1.25+
- ONNX Runtime shared library (CPU or CUDA)
- Pre‑trained ONNX models placed under `models/` (see below)

Install ONNX Runtime and set up environment:

```bash
# Install ONNX Runtime
./scripts/setup-onnxruntime.sh

# Enable automatic environment loading (one-time setup)
direnv allow
```

The project uses direnv to automatically configure environment variables (`CGO_CFLAGS`, `CGO_LDFLAGS`, `LD_LIBRARY_PATH`) when entering the directory. Alternatively, you can source the environment manually:

```bash
source scripts/setup-env.sh
```

Locally, the repo includes `onnxruntime/` with a Linux x64 distribution.

## Models

Organized under `models/`:

- detection/
  - mobile/ `PP-OCRv5_mobile_det.onnx`
  - server/ `PP-OCRv5_server_det.onnx`
- recognition/
  - mobile/ `PP-OCRv5_mobile_rec.onnx`
  - server/ `PP-OCRv5_server_rec.onnx`
- layout/
  - Document orientation: `pplcnet_x1_0_doc_ori.onnx`
  - Textline orientation (light): `pplcnet_x0_25_textline_ori.onnx`
  - Textline orientation: `pplcnet_x1_0_textline_ori.onnx`
  - Rectification (UVDoc): `uvdoc.onnx`
- dictionaries/
  - Default dictionary: `ppocr_keys_v1.txt`

You can override model/dictionary paths with flags or by setting `GO_OAR_OCR_MODELS_DIR`. Internally, `internal/models/paths.go` resolves filenames in either the organized tree or a flat legacy layout.

## Build, Run, and Test

Convenience tasks (requires `just`):

- `just build` – Build CLI with ldflags into `bin/pogo`
- `just build-dev` – Fast local build
- `just run -- <args>` – Run from source; example: `just run image input.jpg`
- `just test` – Run tests (`go test -v ./...`)
- `just test-coverage` – Create `coverage.out` and `coverage.html`
- `just fmt` – Format via treefmt (gofumpt + gci)
- `just lint` / `just lint-fix` – Run golangci-lint

Without `just`:

```
go build -o bin/pogo ./cmd/ocr
go test -v ./...
```

## CLI Usage

Basic:

```
pogo test
pogo image input.jpg --format json
pogo batch images/*.png --format text
pogo pdf scan.pdf --format json
```

Common flags:

- `--models-dir <dir>` – Root directory for models
- Detection: `--det-model <path>`, `--confidence <0..1>`, `--det-polygon-mode minrect|contour`
- Recognition: `--rec-model <path>`, `--rec-height <32|48>`, `--dict <paths,comma>`, `--dict-langs <en,de,...>`
- Orientation: `--detect-orientation`, `--orientation-threshold <0..1>`
- Textline orientation: `--detect-textline`, `--textline-threshold <0..1>`
- Rectification (experimental): `--rectify`, `--rectify-model`, `--rectify-mask-threshold`, `--rectify-height`, `--rectify-debug-dir <dir>`
- Output: `--format text|json|csv`, `--output <file>`, `--overlay-dir <dir>`
- Logging: `--log-level debug|info|warn|error`, `--verbose` (equivalent to `--log-level=debug`)

Examples:

```
# Image → OCR JSON
pogo image doc.jpg --format json --detect-orientation --rectify

# Batch images, save overlays and rectification debug images
pogo batch images/ --recursive \
  --detect-orientation --rectify \
  --overlay-dir .tmp/overlay --rectify-debug-dir .tmp/rectify

# PDF OCR with rectification and language dictionaries
pogo pdf scan.pdf --format json \
  --pages 1-5 --rectify \
  --dict-langs en,de
```

Rectification debug outputs (when `--rectify-debug-dir` is set):

- `rect_mask_<ts>.png` – UVDoc mask heatmap with threshold highlighting
- `rect_overlay_<ts>.png` – Original image with estimated page quad overlay
- `rect_compare_<ts>.png` – Side‑by‑side original+quad (left) and rectified preview (right)

Rectification applies only if quality gates pass (mask coverage, area ratio, and aspect bounds) to avoid harmful warps.

## HTTP Server

Start a local server:

```
pogo serve --port 8080 --language en --detect-orientation
```

Endpoints:

- `POST /ocr/image` – Process uploaded images (multipart)
- `GET /health` – Liveness check
- `GET /models` – List available models

Server flags mirror the pipeline flags (det/rec models, orientation, textline, dicts). Overlays can be enabled in responses.

Basic examples:

```
# Health
curl -s http://localhost:8080/health | jq

# Models
curl -s http://localhost:8080/models | jq

# OCR image (JSON)
curl -s -F image=@doc.jpg http://localhost:8080/ocr/image | jq

# OCR image (plain text)
curl -s -F image=@doc.jpg -F format=text http://localhost:8080/ocr/image

# OCR image (overlay PNG)
curl -s -o overlay.png -F image=@doc.jpg -F format=overlay \
  -F box=#FF0000 -F poly=#00FF00 http://localhost:8080/ocr/image

# OCR PDF (JSON). Pages can be passed via 'pages' like '1-3' or '1,4'.
curl -s -F pdf=@scan.pdf -F pages=1-3 http://localhost:8080/ocr/pdf | jq

# OCR PDF (plain text)
curl -s -F pdf=@scan.pdf -F format=text http://localhost:8080/ocr/pdf
```

## Project Status

Major pieces implemented:

- Detector and recognizer integration with ONNX Runtime
- Orientation classifiers (document + per‑text‑line)
- Document rectification (UVDoc) with quality gating and homography warp
- Batch and PDF flows now use the full pipeline
- CLI + HTTP server
- Tests across utils, ONNX setup, detector/recognizer init, pipeline wiring

See PLAN.md for detailed progress and upcoming enhancements.

## Development

- Format: `just fmt`
- Lint: `just lint` (120‑char soft limit; Go idioms enforced by golangci‑lint)
- Tests: `just test` / `just test-coverage`
- GPU: configure CUDA providers in ONNX Runtime and use `--gpu` flags (where supported in the codebase)

Repo structure:

```
cmd/ocr           # Cobra CLI
internal/
  detector/       # ONNX detector + post-processing
  recognizer/     # ONNX recognizer + CTC decoding
  orientation/    # Orientation classifiers
  rectify/        # Document rectification (UVDoc)
  pipeline/       # Orchestration + parallel processing + results
  pdf/            # PDF image extraction
  server/         # HTTP server
  utils/          # Imaging, tensors, geometry, etc.
models/           # See Models section
scripts/          # ONNX runtime setup helpers
```

## Troubleshooting

- ONNX Runtime not found
  - Ensure the shared library is installed and exported. Use `./scripts/setup-onnxruntime.sh` and source `scripts/setup-env.sh`.
  - Local path `onnxruntime/lib` is auto-detected on Linux.
- Model not found
  - Verify files under `models/` or pass `--models-dir`/per‑model overrides.
- Recognition returns empty text
  - Check the dictionary paths and language cleaning rules.
- Rectification not applied
  - Inspect debug images, adjust `--rectify-mask-threshold` and gates via code defaults if necessary.

## Acknowledgements

- Inspired by OAR‑OCR and PaddleOCR models.
- pdfcpu for PDF image extraction.
