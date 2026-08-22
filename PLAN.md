# pogo — Plan

## What pogo is

pogo is a Go library and CLI that runs ONNX text-detection and text-recognition
models over images and returns boxes with text and per-line confidence.

**Models are described by a spec file, not by Go code.** Swapping the recognizer
for a different one is a YAML edit, not a patch.

## What pogo is not

Each names the package deleted for it, so the boundary is traceable.

| Non-goal                                        | Package removed                                          |
| ----------------------------------------------- | -------------------------------------------------------- |
| An HTTP server                                  | `internal/server`                                        |
| Barcode decoding                                | `internal/barcode`, `pipeline/barcodes*.go`              |
| PDF input and output                            | `internal/pdf` (returns later, with a fixture)           |
| Kubernetes, cloud-native anything               | `deployment/`, `.goreleaser.yaml`                        |
| Handwriting; layout, tables, formulas; training | — never implemented, and PP-OCR is a printed-text engine |

If one of these comes back, it comes back with a test that proves it works.

## Why this plan replaces the old one

The previous PLAN.md was 1,104 lines and 20 phases, running from "Phase 0:
Critical Bug Fixes" through Kubernetes, "Enterprise Features" and "Innovation
Metrics", closing on _247 remaining tasks_. None of Phase 0 was done. The
recognizer still cannot read the word "Hello" — it returns `Hellg`.

Current state, measured:

- **24,329 lines** of source across 18 `internal/` packages, plus **33,873** of
  Go tests and **5,241** of godog tests — a 1.6:1 ratio that never caught a
  wrong normalization constant.
- **`pkg/ocr/` is an empty directory.** pogo cannot be imported as a Go library
  at all, which is the one thing it exists to be.
- **No `Detector` or `Recognizer` interface exists.** `pipeline.Pipeline` holds
  concrete structs (`internal/pipeline/pipeline.go:512-521`). The only such
  interfaces in the tree are dead test scaffolding
  (`internal/pipeline/parallel_test.go:322,330`), declared and never used.
- **~4,000–5,000 lines are dead** or unreachable from the CLI.
- **99 config fields, ~272 CLI flag declarations.** One knob touches 3–5 files.
- **229 MB of ONNX weights committed to git**, no LFS — `.git` is 274 MB. The
  ignore rule `/models/*.onnx` never matched `models/detection/…`, which is
  exactly how they got in.
- **`just check` fails on all four of its steps**: 44 files not treefmt-clean
  (fixed in Phase 0), 131 lint findings, 3 failing test packages, and
  `go mod tidy -diff` failing because of an unbuildable barcode backend.

## The defects that define credibility

Two bugs, and the reason neither was ever caught. This is all of Phase 1.

### a. The charset is one token short

`models/dictionaries/ppocrv5_dict.txt` has 18,383 lines.
`internal/recognizer/inference.go:188` computes `classes = charset.Size() + 1`
= 18,384. Both bundled recognizer graphs were loaded and inspected directly:
their output `fetch_name_0` has shape `[dyn, dyn, 18385]`. PaddleOCR appends the
space token itself under `use_space_char=True`; pogo never does, so every space
is silently dropped — `Rotatedlext`.

The mechanism is exact. `inference.go:229` decodes with
`charset.LookupToken(idx - 1)`; index 18384 — PaddleOCR's space — resolves to
`LookupToken(18383)`, which is out of range, and `dictionary.go:150-158` returns
`""` for that. No error, no warning. `dictionary.go:75` also skips empty lines,
so the fix cannot be a text-file edit: it has to be code.

The important part is not the missing line. It is that **nothing checks.**
Nowhere is `charset.Size()+1` validated against the model's real class count —
a grep for any class-count validation across `internal/**/*.go` returns nothing.
The mismatch instead flows into `determineClassesFirst` (`inference.go:527-548`),
which infers the tensor layout by matching a dimension against the expected
class count. Today, with 18,384 against `[N, T, 18385]`, **neither arm matches**
and it falls through to its `return false` default — which happens to be the
correct NTC answer. The layout is right by luck, not by verification. That is worse
than being wrong, because it will hold until the day a model makes it not hold.

So the fix is an assertion at model load, not an edit to a text file:
**a dictionary that does not match the model must be a startup error, not
garbage output.** And the layout must be declared, not inferred.

### b. Recognition input is normalized to the wrong range

PaddleOCR feeds the recognizer `(x/255 - 0.5) / 0.5`, i.e. `[-1, 1]`.
`NormalizeImage` (`internal/utils/image_processing.go:154`) produces `[0, 1]`
and stops — wrong mean, half the dynamic range. That is the signature of the
residual character errors.

The same arithmetic is shared by the detector, which needs ImageNet mean/std
instead, so detection is mis-normalized too. There are three copies (`:154`,
`:197`, `:233`) whose bodies are byte-for-byte identical, differing only in how
they allocate. A grep for `0.485|0.456|0.406|imagenet` across the entire tree
returns **zero hits**, and `internal/config` has **no normalization field at
all** (`structs.go` carries 99 `json:` tags; none of them is a mean or a std).

The old plan investigated exactly this, tested the ImageNet hypothesis against
the _recognizer_, got 0% confidence, and concluded "recognition models use
simple `/255` scaling" — the right experiment, the wrong control, filed as
"✅ INVESTIGATION COMPLETED — expected behavior documented".

> An investigation that ends in a rationalization is worse than no investigation.

### c. And the test suite was built so it could be tuned

`internal/pipeline/accuracy_test.go` is a real CER/WER harness — Levenshtein
similarity, character accuracy, word accuracy. But it reads its **pass
thresholds from the same JSON file as its ground truth**
(`testdata/fixtures/ocr_accuracy.json`), so any failing case is fixed by editing
its own row. All 14 cases are synthetic renders, and the bars were walked down
case by case: 0.8, 0.7, 0.65, 0.55, 0.5. Every `min_avg_conf` is `0.0`, so that
assertion has never once executed.

`Hello` is gated at `min_similarity: 0.8`. `Hellg` against `Hello` scores
exactly 0.8, and the assertion is `GreaterOrEqual`. It passes by construction.

> A threshold that lives next to the data it judges is not a threshold.

Two things this document previously got wrong, and they matter:

- The suite is **not currently green**. A full run fails 11 of 14 cases,
  producing `Hellg`, `Hor1c`, `""`, `Samele`, `Rotatedlext`, `Rot.at.ectext.`,
  `scannecoccument.`, `Haoetr`. The fixture PNGs were regenerated at some point
  and the walked-down bars no longer cover the damage. The design is what is
  broken; the current status is merely failing.
- `german_text.png` is **not** an untestable case. Its similarity/CAR/WAR bars
  are all `0.0` and vacuous, but it also carries `contains_any` with
  `min_contains: 1`, asserted at `accuracy_test.go:163` — and that assertion
  fails today. It needs a hand-keyed `expected` string, not deletion.

Also measured, and unrelated to accuracy: **`pogo pdf` does not run at all.**
`cmd/ocr/cmd/pdf.go:687-689` assigns the models _directory_ to
`detectorConfig.ModelPath`, which reaches ONNX as a model file and fails with
`Protobuf parsing failed`. Every PDF scenario in the godog suite fails for that
one line. It will not be fixed; `internal/pdf` is scheduled for deletion, and
PDF returns later on top of the library API with a fixture that proves it works.

## Architecture: the model is a spec file

### Public API — `pkg/pogo`

Small enough to read in one screen.

```go
type Engine struct{ ... }

func Open(set ModelSet, opts ...Option) (*Engine, error)
func (e *Engine) Read(img image.Image) (Result, error)
func (e *Engine) Close() error

type Result struct{ Lines []Line }

type Line struct {
    Box        Quad
    Text       string
    Confidence float32
}

// The seams that make models swappable.
type Detector interface {
    Detect(image.Image) ([]Quad, error)
    Close() error
}

type Recognizer interface {
    Recognize([]image.Image) ([]Line, error)
    Close() error
}
```

### ModelSpec — a YAML sidecar next to each `.onnx`

Everything currently hardcoded moves here.

```yaml
kind: recognizer # detector | recognizer
decode: ctc # ctc | db
input:
  layout: NCHW
  channels: 3
  resize: { height: 48, max_width: 320, pad_multiple: 8 }
  color: RGB
  scale: 0.00392156862 # 1/255
  mean: [0.5, 0.5, 0.5]
  std: [0.5, 0.5, 0.5] # -> [-1,1]; the detector spec carries ImageNet values
ctc:
  blank_index: 0
  layout: NTC # NTC | NCT — declared, not inferred
charset:
  file: ppocrv5_dict.txt
  append_space: true # the dropped-space bug, expressed as data
  assert_classes: 18385 # verified against the model at load; mismatch = error
```

Adding PP-OCRv6, a docTR CRNN, or a digits-only fine-tune becomes: drop in the
`.onnx`, write ~15 lines of YAML. No Go changes, no recompile.

What this deletes:

- The three `NormalizeImage*` copies collapse into one preprocessor driven by
  `input:`.
- `determineClassesFirst`'s layout inference goes, replaced by the declared
  `ctc.layout`.
- `blankIndex := 0` (`inference.go:190`, `:501`) and the literal `3` channel
  count (`preprocess.go:188`, `:203`, `detector.go:143`) become spec fields.
- `internal/models/paths.go` — hardcoded model filenames (`:13-29`), the
  walk-up-for-`go.mod` directory heuristic (`:53-75`, which breaks for installed
  binaries), and `ResolveModelPath`'s silent fallback to a flat path it never
  stats (`:108-128`), which is what turns a missing model into a protobuf error.

### Model store

A manifest maps model name to URL + sha256. `pogo models pull` fetches into
`~/.cache/pogo/models`. Weights leave the working tree; git history keeps them —
**no history rewrite**, by decision, so no one's clone breaks.

This is also what makes third-party models first-class instead of second-class:
a model you did not commit is loaded the same way as one you did.

## What is worth keeping

Roughly 2,500 lines carry forward.

- **DB post-processing and polygon geometry** (`internal/detector`) — contour
  tracing, connected components, polygon expansion, quad fitting. Real
  algorithmic work and the hardest part to rewrite. This includes
  `multiscale.go` and `adaptive_threshold.go`, both live.
- **The CTC decoder** (`internal/recognizer/ctc.go`), including beam search.
  Correct; it was fed bad tensors.
- **Homography and warping** from `internal/rectify`, salvaged into
  `internal/detect` for crop rectification.
- **The accuracy harness** in `accuracy_test.go`. The measurement is right; it
  needs its thresholds taken away from it, not a rewrite.
- **`internal/onnx`** — session and tensor handling; thin, and required for any
  model to run at all.
- **Build tooling** — `justfile`, `.golangci.toml`, the three GitHub workflows.

## What gets deleted

About 15,000 lines of source.

| Package                                                   |   Src | Reason                                                                                                                                                                                                                                                                 |
| --------------------------------------------------------- | ----: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `internal/server`                                         | 2,699 | Out of scope. Drags in gorilla/websocket and Prometheus; its `PipelineCache` is an unbounded map of live ONNX sessions                                                                                                                                                 |
| `internal/pdf`                                            | 2,834 | Two divergent implementations (`pipeline.ProcessPDF` vs `pdf.Processor`) with ~1,400 lines of overlap, no fixture proving either works, and a one-line bug that makes the command fail on every input                                                                  |
| `internal/rectify`                                        |   978 | UVDoc off by default; the DocTR path is dead — its model file does not exist and its own comment admits "the exact format may vary"                                                                                                                                    |
| `internal/orientation`                                    |   902 | Off by default, four overlapping escape hatches                                                                                                                                                                                                                        |
| `internal/batch`                                          |   646 | A `for` loop over `Read()` belongs in `cmd/`                                                                                                                                                                                                                           |
| `internal/benchmark` + `cmd/benchmark`                    |   657 | Replaced by `pogo eval`, which measures accuracy rather than speed. Duplicates `detector/benchmark.go`                                                                                                                                                                 |
| `internal/barcode` + `pipeline/barcodes*.go`              |   574 | Cannot compile — the backend is behind `-tags=barcode_gozxing`, `gozxing` is not in `go.mod`, and the package it imports (`gozxing/pdf417`) does not exist in any published version. This is why `go mod tidy` fails                                                   |
| `internal/mempool`                                        |   130 | 130 lines of source, 1,007 lines of tests, pooling buffers on a path that produces wrong answers                                                                                                                                                                       |
| Dead code in `pipeline/`                                  |  ~900 | `parallel.go`, `profile.go`, `monitor.go`, `ProcessImages`, `AdaptiveWorkerPool` — zero non-test callers. `ResourceManager` and `MemoryMonitor` in the same file stay: they are wired in `pipeline.go`                                                                 |
| Dead code in `detector/`                                  |  ~350 | `batch.go`'s `RunBatchInference`, `benchmark.go`, `DefaultAdaptiveNMSThresholds`. The adaptive and size-aware NMS paths themselves stay — they are reachable via YAML config (`config/loader.go:271-272`, `detector/postprocess_onnx.go:66-96`), just never via a flag |
| `deployment/`, `.goreleaser.yaml`                         |     — | Premature; `--version` prints a hardcoded string regardless                                                                                                                                                                                                            |
| `pkg/ocr/`, `internal/version/`, `cmd/simple-model-test/` |     0 | Empty directories                                                                                                                                                                                                                                                      |

**Deletion is not loss.** Everything stays on `main` and stays recoverable.

---

# Phases

No week estimates. The old plan's 23-week schedule was fiction. Each task names a
file or a command, and each carries an acceptance test that describes what must
be true **after** it — not what happens to be true now. An acceptance test that
cannot fail is not one.

## Phase 0 — A clean base

Done. Recorded here because the rest builds on it.

- **T0.1** Commit the treefmt formatting pass as its own `style:` commit.
  _Accept:_ `just check-formatted` exits 0.
- **T0.2** Enable revive's `file-length-limit` at 1500 in `.golangci.toml`.
  _Accept:_ the limit produces findings rather than being inert.
- **T0.3** Point the orientation scenarios at `testdata/images/rotated/rotated_45.png`;
  `testdata/images/rotated_text.png` does not exist.
  _Accept:_ the "Orientation detection configuration" scenario passes.
- **T0.4** Resolve testdata from the project root instead of the package CWD,
  reusing `testutil.GetProjectRoot` / `GetFixturesDir`; delete the two untracked
  symlinks that were propping this up.
  _Accept:_ `go test ./internal/pipeline -run TestOCRAccuracy` finds its fixtures
  with no symlink present anywhere in the tree.
- **T0.5** Ignore coverage artifacts and `/overlays/`; delete the superseded
  `GOAL.md`, `COMPARISON.md`, `WORK_INVOICE.md`, `QWEN.md`.
  _Accept:_ `git status --porcelain` is empty on a clean checkout.

_Exit: working tree clean, `just build` green, `just check-formatted` green._

## Phase 1 — Make it read "Hello"

### 1.1 Make the harness able to fail first

- **T1.1** Raise every case in `ocr_accuracy.json` to exact match
  (`min_similarity`, `min_car`, `min_war` all `1.0`).
  _Accept:_ the suite fails, printing actual vs expected for every wrong case.
  This failure is the baseline the rest of the phase is measured against.
- **T1.2** Hand-key a real `expected` string for `german_text.png`, replacing
  `""` and the `contains_any` fallback.
  _Accept:_ the case asserts a concrete string; no case in the file has an empty
  `expected`.
- **T1.3** Default `accuracy_test.go:89` to the mobile models; keep the server
  models behind `POGO_ACCURACY_MODELS=server`.
  _Accept:_ `go test ./internal/pipeline` finishes in under 60 s (it takes 715 s
  today, almost all of it this one test).

### 1.2 Assert the class count at load

- **T1.4** Read the recognizer's real output class count from the ONNX output
  shape at init. `internal/onnx` already exposes I/O info; the seam is
  `recognizer.go:185-203`.
  _Accept:_ a unit test asserts 18385 for both bundled recognizer models, read
  from the graph rather than hardcoded.
- **T1.5** Fail recognizer construction when
  `charset size + specials != model classes`, naming both numbers in the error.
  _Accept:_ loading PP-OCRv5 rec with `ppocr_keys_v1.txt` (142 lines) returns an
  error containing both `18385` and `143`; loading it with `ppocrv5_dict.txt`
  returns no error. A dictionary/model mismatch can no longer reach inference.

### 1.3 The space token

- **T1.6** Add an explicit append-space step to charset loading in
  `dictionary.go`, off by default and on for PP-OCRv5.
  _Accept:_ charset size becomes 18384 and T1.5 passes with the real dictionary.
- **T1.7** Verify the decode index mapping end to end.
  _Accept:_ `rotated/rotated_0.png` recognizes as `Rotated Text`, with the space.
  A test asserts on the space specifically, so its loss cannot regress silently.

### 1.4 One normalizer, parameterized

- **T1.8** Collapse `NormalizeImage` / `NormalizeImageIntoBuffer` /
  `NormalizeImagePooled` (`image_processing.go:154/197/233`) into one
  implementation taking scale, mean and std; keep the buffer and pooled variants
  as thin wrappers over it.
  _Accept:_ a table test checks one known pixel through both a `[0,1]` and a
  `[-1,1]` parameterization; the three entry points produce identical values.
- **T1.9** Detector passes ImageNet mean/std; recognizer passes `0.5/0.5`.
  Update the `rectify` and `orientation` call sites too.
  _Accept:_ per-consumer tests assert the resulting tensor's range — recognizer
  input reaches negative values, detector input is ImageNet-centred.
- **T1.10** Replace the hardcoded channel literal `3` (`preprocess.go:188,203`,
  `detector.go:143`) with a value carried on the config path.
  _Accept:_ `grep -rn "NewImageTensor(.*, 3," internal/` returns nothing.

### 1.5 Declare, don't infer

- **T1.11** Replace `determineClassesFirst` (`inference.go:527`) with a declared
  `NTC`/`NCT` layout on the recognizer config, and make `blankIndex` a field
  rather than the literal at `inference.go:190` and `:501`.
  _Accept:_ synthetic-tensor tests decode correctly under both layouts, and a
  declared layout that contradicts the model's actual shape is a load-time error
  rather than a silent fallback.

_Phase 1 exit: all 14 fixtures pass at exact match. `TestProcessImage_Smoke`
reads `hello`, not `hel1o`. Both failing `TestRecognizeBatch` tests are green.
`go test ./...` passes apart from the PDF scenarios Phase 3 deletes, and no
package takes over 60 s._

## Phase 2 — Measure something real

### 2.1 Take the thresholds away from the data

- **T2.1** Split `ocr_accuracy.json` into a corpus manifest carrying only
  `image` and `expected`.
  _Accept:_ the file contains no `min_*` key. Grepping for one returns nothing.
- **T2.2** Define one corpus-level gate in code: mean CER, mean WER, exact-match
  count.
  _Accept:_ degrading any single case fails the corpus gate, and there is no
  per-case knob that can silence it.

### 2.2 `pogo eval`

- **T2.3** Promote the harness in `accuracy_test.go` into
  `pogo eval <corpus-dir>`, printing per-case and aggregate CER/WER.
  _Accept:_ `pogo eval ./testdata/corpus` prints a table and exits 0.
- **T2.4** Commit a baseline per corpus; `eval` compares against it and exits
  non-zero on regression.
  _Accept:_ deliberately truncating the dictionary makes `pogo eval` exit 1 and
  name the metric that moved.

### 2.3 A corpus that isn't synthetic

- **T2.5** Add at least 10 real scans, German included, with hand-keyed ground
  truth under `testdata/corpus/real/`.
  _Accept:_ every image is a real capture rather than a render, and each has
  ground truth committed beside it.
- **T2.6** Give the real corpus its own committed baseline.
  _Accept:_ `pogo eval ./testdata/corpus/real` exits 0 and its CER is recorded
  in the repo, so the next change has something to be compared against.

_Phase 2 exit: one committed CER/WER baseline per corpus. Nothing merges
afterwards that moves CER the wrong way._

## Phase 3 — Cut

Strictly leaf-first — the order below is the dependency order, and taking it out
of order means editing code twice. Each task is one commit and ends with the same
three checks: `go build ./...`, `go test ./...`, and `pogo eval` showing CER
unchanged from the Phase 2 baseline. Only the extra acceptance criteria are
listed per task.

- **T3.1** Empty directories (`pkg/ocr/`, `internal/version/`,
  `cmd/simple-model-test/`), plus `deployment/` and `.goreleaser.yaml`.
  _Accept:_ no build reference to any of them remains.
- **T3.2** `cmd/benchmark` + `internal/benchmark` (657 src). One caller each.
  _Accept:_ the binary is gone and nothing imports the package.
- **T3.3** `cmd/ocr/cmd/batch.go` + `internal/batch` (646). Re-home
  `pipeline.CalculateParallelStats`, which `internal/batch/config.go` is the only
  external user of.
  _Accept:_ `pogo --help` no longer lists `batch`.
- **T3.4** `cmd/ocr/cmd/serve.go` + `internal/server` (2,699), the four
  `server_*.feature` files, and `support/server_*.go`.
  _Accept:_ gorilla/websocket and the Prometheus client leave `go.mod`.
- **T3.5** `cmd/ocr/cmd/pdf.go` + `internal/pdf` (2,834) +
  `pipeline/process_pdf.go` + `pdf_processing.feature` + `testdata/documents/`.
  _Accept:_ pdfcpu leaves `go.mod`.
- **T3.6** `internal/barcode` + `pipeline/barcodes*.go` + the 12 barcode CLI
  flags.
  _Accept:_ **`go mod tidy -diff` exits 0.** This is the task that repairs
  `just check-tidy`.
- **T3.7** Pipeline dead code: `parallel.go`, `profile.go`, `monitor.go`,
  `ProcessImages` (keep `ProcessImagesContext`), and `AdaptiveWorkerPool` from
  `resources.go`. `ResourceManager` and `MemoryMonitor` stay — `pipeline.go`
  wires them.
  _Accept:_ `resources_test.go` shrinks rather than breaking.
- **T3.8** Detector dead code: `batch.go`'s `RunBatchInference`, `benchmark.go`,
  `DefaultAdaptiveNMSThresholds`. `multiscale.go`, `adaptive_threshold.go` and
  the NMS variants stay.
  _Accept:_ the `--det-multiscale` flag still works and the YAML-configured
  adaptive NMS path still has a test.
- **T3.9** `internal/rectify` (978) + `internal/orientation` (902), their config
  fields, and the three `internal/recognizer` call sites.
  _Accept:_ the config struct loses every rectify and orientation field, and no
  flag referencing them survives in `--help`.
- **T3.10** `internal/mempool` (130 src, 1,007 test) — last, because it has 12
  call sites across detector, recognizer and utils.
  _Accept:_ `pogo eval` wall-clock is not materially worse than before the
  removal; if it is, that is a real finding and pooling comes back with a
  benchmark that justifies it.
- **T3.11** Godog: delete the features for deleted subsystems, write step
  definitions for what survives, set `godog.Options{Strict: true}`, and fix
  `just test-integration` — it runs `-run "Integration"` while the entry point
  is `TestFeatures`, so today it runs no scenarios at all.
  _Accept:_ zero undefined and zero failing scenarios, and removing any single
  step definition turns the suite red instead of silently passing. (Today: 179
  scenarios, 76 pass, 64 fail, 39 undefined and invisible.)
- **T3.12** Clear the remaining golangci-lint findings in the surviving code.
  _Accept:_ `just lint` exits 0.

_Phase 3 exit: `internal/` source under 6,000 lines. **`just check` exits 0** —
all four steps, for the first time. CER unchanged from Phase 2._

## Phase 4 — The library exists

- **T4.1** Define the `Detector` and `Recognizer` interfaces in `pkg/pogo`.
  _Accept:_ a fake detector and fake recognizer drive the pipeline in a unit test
  with no ONNX runtime loaded at all.
- **T4.2** Make `pipeline.Pipeline` hold those interfaces instead of the concrete
  structs at `pipeline.go:512-521`.
  _Accept:_ CER unchanged; the fakes from T4.1 substitute cleanly.
- **T4.3** Implement `Engine` — `Open` / `Read` / `Close` — plus `Result`,
  `Line` and `Quad`.
  _Accept:_ `go doc ./pkg/pogo` fits on one screen.
- **T4.4** Reduce `cmd/pogo` to a thin consumer: roughly 12 flags instead of
  ~272, and a config struct of roughly 15 fields instead of 99.
  _Accept:_ `pogo image --help` fits in one terminal page and every remaining
  flag has a test that exercises it.
- **T4.5** Add `example_test.go` and a scratch-module smoke check.
  _Accept:_ a 20-line `main.go` in a module **outside** this repo imports pogo
  and prints boxes with text and confidence.

_Phase 4 exit: that scratch module compiles and prints correct text._

## Phase 5 — The model is swappable

- **T5.1** `ModelSpec` type and YAML loader (`kind`, `decode`, `input`, `ctc`,
  `charset`).
  _Accept:_ a spec whose `assert_classes` does not match the model is a load-time
  error naming both numbers — this subsumes the hand-rolled check from T1.5.
- **T5.2** Drive preprocessing from `input:` and decoding from `ctc:`.
  _Accept:_ changing `mean` or `std` in the YAML changes the produced tensor,
  with no recompile.
- **T5.3** Write sidecar specs for the four bundled models and delete
  `internal/models/paths.go`.
  _Accept:_ a missing model produces an error naming the path that was searched,
  not `Protobuf parsing failed`; and an installed binary run outside a Go module
  resolves its models correctly.
- **T5.4** Model store: a manifest of name → URL + sha256, and
  `pogo models pull` fetching into `~/.cache/pogo/models`.
  _Accept:_ with an empty cache, `pogo models pull && pogo image x.png` works;
  a corrupted download fails its checksum and is not cached.
- **T5.5** Untrack the weights, fix the ignore pattern to `/models/**/*.onnx`,
  and teach CI to pull. No history rewrite.
  _Accept:_ `git ls-files models | grep '\.onnx$'` is empty, and a fresh clone
  plus `pogo models pull` passes `pogo eval`.
- **T5.6** Prove swappability.
  _Accept:_ a second, structurally different recognizer runs end to end via a new
  YAML file and **zero Go changes**.

_Phase 5 exit: T5.6 passes and no weights remain in the working tree._

## Verification

From a cold clone:

```sh
just build                      # every phase
just check                      # format + lint + test + tidy; green from Phase 3
pogo models pull                # Phase 5
pogo eval ./testdata/corpus     # Phase 2: prints CER/WER, compares to baseline
```

For Phase 4, in a scratch module outside the repo:

```go
f, _ := os.Open("scan.png")
img, _ := png.Decode(f)

e, err := pogo.Open(pogo.DefaultModels())
if err != nil { panic(err) }
defer e.Close()

res, _ := e.Read(img)
for _, l := range res.Lines {
    fmt.Printf("%v %q %.2f\n", l.Box, l.Text, l.Confidence)
}
```

If that compiles and prints correct text, pogo is what it set out to be.
