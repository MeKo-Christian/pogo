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

No week estimates. The old plan's 23-week schedule was fiction. Every task names
a file or a command, breaks into checkable subtasks, and carries an acceptance
test that describes what must be true **after** it — not what happens to be true
now. An acceptance test that cannot fail is not one.

Checkboxes are the working state: `[x]` is done and committed, `[ ]` is not.

## Phase 0 — A clean base

Done. Recorded because the rest builds on it, and because two of its tasks were
not foreseen.

### Task 0.1 — Commit the treefmt formatting pass

- [x] Stage the ~40 files touched only by gofumpt, gci and prettier
- [x] Keep them apart from the intentional edits so later diffs stay readable
- [x] Commit as `style: apply treefmt across the tree` (`5ab2473`)

**Accept:** the formatting churn is one commit and contains no behaviour change.

### Task 0.2 — Enable revive's file-length limit

- [x] Add `revive` to the enabled linters in `.golangci.toml`
- [x] Configure `file-length-limit` at 1500 lines, counting comments and blanks
- [x] Confirm the rule actually fires (`internal/pdf/hybrid_test.go`, 1623 lines)

**Accept:** the limit produces a finding rather than being inert.

### Task 0.3 — Point the orientation scenarios at an existing fixture

- [x] Replace `testdata/images/rotated_text.png` — which exists nowhere in the
      repo — with `testdata/images/rotated/rotated_45.png` in
      `configuration.feature` and `orientation_integration.feature`
- [x] Note in the commit that the `orientation_integration.feature` edit is
      inert until Task 3.11 writes the missing step definitions

**Accept:** the "Orientation detection configuration" scenario passes.

### Task 0.4 — Commit the regenerated overlay fixture

- [x] Commit `overlays/simple_text_overlay.png` (`63dbfc7`)

**Accept:** the working tree carries no unexplained binary diff.

### Task 0.5 — Resolve testdata from the project root, not the CWD

- [x] Reuse the existing `testutil.GetProjectRoot` and `testutil.GetFixturesDir`
      rather than adding a new helper
- [x] Join `ocr_accuracy.json` and each fixture image path against the root in
      `internal/pipeline/accuracy_test.go`
- [x] Delete `internal/pipeline/testdata` and the self-referential
      `testdata/testdata` symlink that were propping the old paths up
- [x] Run one accuracy subtest to prove the paths resolve without them

**Accept:** `go test ./internal/pipeline -run TestOCRAccuracy` finds its fixtures
with no symlink present anywhere in the tree.

### Task 0.6 — Ignore coverage artifacts and generated overlays

- [x] Add `coverage_*.out`, `*_coverage.out` and `/overlays/` to `.gitignore`
- [x] Delete the nine stale coverage files from the working tree

**Accept:** a full test-and-coverage run leaves `git status --porcelain` empty.

### Task 0.7 — Remove superseded planning documents

- [x] Delete `GOAL.md` (56 KB), `COMPARISON.md`, `WORK_INVOICE.md`
- [x] Delete the 0-byte `QWEN.md`

**Accept:** no document in the repo describes scope the project has dropped.

### Task 0.8 — Repair `docs/openapi.yaml`

Unforeseen. Found while checking Phase 0's own exit criterion.

- [x] Fix the mis-indented `properties` block under the `/ocr/pdf` request body
      (line 88), which made the file invalid YAML
- [x] Understand the blast radius: yamlfmt aborts its entire pass on one parse
      error, so **no YAML file in the repo had ever been formatted**
- [x] Let the formatter reflow the four other YAML files

**Accept:** `python3 -c "import yaml; yaml.safe_load(open('docs/openapi.yaml'))"`
succeeds, and yamlfmt processes the YAML set instead of bailing out.

### Task 0.9 — Let yamlfmt own YAML formatting

Unforeseen. `just check-formatted` still failed on a byte-identical tree.

- [x] Diagnose: prettier and yamlfmt both claimed `*.yaml`/`*.yml`, so each run
      rewrote `docs/openapi.yaml` and tripped `--fail-on-change`
- [x] Drop YAML from prettier's `includes` in `treefmt.toml`
- [x] Verify idempotency by running the check three times

**Accept:** `just check-formatted` exits 0, and exits 0 again immediately after.

_Phase 0 exit: working tree clean, `just build` green, `just check-formatted`
green and idempotent._

## Phase 1 — Make it read "Hello"

The harness is made able to fail first, then the two defects are fixed against
it. Tasks 1.1–1.3 must land before 1.4, or there is nothing to measure against.

### Task 1.1 — Tier the corpus and raise the upright cases to exact match

Exact match for all 14 was the original intent, but `rotated_45` and
`rotated_-45` are small 45°-rotated text on a large canvas, and Task 3.9 deletes
`internal/rectify` and `internal/orientation` — the deskew machinery that could
make them exact. So the corpus is tiered instead.

- [x] Add a `group` field to each case and to `accuracyCase`
- [x] `group: "upright"` — `simple_1..6`, `rotated_0`, `scanned_document`,
      `german_text` (9 cases) at `min_similarity`/`min_car`/`min_war` = `1.0`
- [x] `group: "rotated"` — `rotated_90/180/270/45/-45` (5 cases) keep their
      similarity bars; Phase 3 decides their fate, not Phase 1
- [x] Run the suite and record the failure output in the commit message

**Accept:** the upright group fails, printing actual vs expected for every wrong
case. That failure is the baseline the rest of the phase is measured against.

### Task 1.2 — Give `german_text.png` a real expectation

- [x] Read the image and hand-key its actual text — it is `Hallo Welt!`
- [x] Replace `expected: ""` with that string
- [x] Delete the `contains_any` / `min_contains` fallback, which is the only
      thing that case ever asserted — and which could never have passed, since
      the image contains no umlaut and no `ß`

**Accept:** no case in the file has an empty `expected`, and grepping for
`contains_any` returns nothing.

### Task 1.3 — Fix the model gate, then attribute the runtime

The original task said to default the test to the mobile models. That premise
was wrong: `accuracy_test.go` asked for _server_ paths only as its `t.Skipf`
gate, while `detector.DefaultConfig()` and `recognizer.DefaultConfig()` both set
`UseServerModel: false`, so the pipeline was already running mobile weights. The
real defect was a test that skipped when server weights were absent despite
never loading them.

- [x] Gate on the models actually loaded, and honour `POGO_ACCURACY_MODELS=server`
      through `Builder.WithServerModels`
- [x] Log per-case elapsed time and region count, so the runtime can be
      attributed rather than guessed at

**Accept:** the suite no longer skips when only mobile weights are present, and
the per-case timing table shows where the wall clock actually goes.

### Task 1.12 — Find out why three fixtures take five and a half minutes

Measured in Task 1.3. `TestOCRAccuracy` is 370 s, and three cases are 89 % of it:

| Case                           | Elapsed | Regions found |
| ------------------------------ | ------- | ------------- |
| `scanned/scanned_document.png` | 2m25s   | 2             |
| `rotated/rotated_45.png`       | 1m42s   | 1             |
| `rotated/rotated_-45.png`      | 1m23s   | 1             |
| all eleven others, combined    | 40s     | 1 each        |

All three are large canvases holding very little text. Spending 145 s to return
two regions from a 1024×768 image is not a slow model, it is a defect. The other
eleven cases average 3.6 s on the same weights.

- [ ] Profile one of the three; find where the time goes (DB post-processing,
      contour extraction and NMS on the graph-paper grid are the first suspects)
- [ ] Fix the pathology, or record why it is inherent
- [ ] Re-measure and put the new table in the commit message

**Accept:** the full accuracy corpus runs in under 60 s, which is what Task 1.3
originally promised. Until then `go test -short` skips the corpus so the unit
suite stays fast — that is a workaround, not the fix.

### Task 1.4 — Read the model's real class count

- [x] Find the output-shape accessor already exposed by `internal/onnx`
- [x] Read the recognizer's output class count at init; the seam is
      `loadCharsetForRecognizer`, `recognizer.go:185-203`
- [x] Surface it on the recognizer so Task 1.5 can assert against it

**Accept:** a unit test asserts 18385 for both bundled recognizer models, read
from the graph rather than hardcoded.

### Task 1.5 — Make a dictionary/model mismatch a startup error

- [x] Fail recognizer construction when
      `charset size + specials != model classes`
- [x] Name both numbers in the error message
- [x] Add the negative test: PP-OCRv5 rec + `ppocr_keys_v1.txt` (142 lines)
- [x] Add the positive test: PP-OCRv5 rec + `ppocrv5_dict.txt`

**Accept:** the negative case returns an error containing both `18385` and
`143`; the positive case returns no error. A mismatch can no longer reach
inference at all.

### Task 1.6 — Append the space token

- [x] Add an explicit append-space step to charset loading in `dictionary.go`,
      off by default
- [x] Turn it on for PP-OCRv5
- [x] Leave `dictionary.go:75`'s empty-line skip alone — the fix is code, not a
      blank line in the dictionary file

**Accept:** charset size becomes 18384 and Task 1.5's assertion passes with the
real dictionary.

### Task 1.7 — Prove the decode index mapping end to end

- [x] Trace `inference.go:229`'s `LookupToken(idx - 1)` against the new size
- [x] Add a test asserting on the space character specifically, not just on
      overall similarity

**Accept:** `rotated/rotated_0.png` recognizes as `Rotated Text`, with the
space — and losing the space again turns a test red rather than shaving a
similarity score.

### Task 1.8 — Collapse the three normalizers into one

- [x] Write one implementation taking scale, mean and std
- [x] Reduce `NormalizeImage`, `NormalizeImageIntoBuffer` and
      `NormalizeImagePooled` (`image_processing.go:154/197/233`) to thin
      wrappers that differ only in how they allocate
- [x] Add a table test covering a known pixel through both a `[0,1]` and a
      `[-1,1]` parameterization

**Accept:** the three entry points produce identical values for identical
parameters, and the arithmetic exists in exactly one place.

### Task 1.9 — Give each consumer its own constants

- [x] Detector passes ImageNet mean/std
- [x] Recognizer passes `0.5/0.5`, yielding `(x/255 - 0.5) / 0.5`
- [x] Update the `internal/rectify` and `internal/orientation` call sites
- [x] Assert the resulting tensor range per consumer

**Accept:** recognizer input reaches negative values; detector input is
ImageNet-centred. Neither is `[0,1]` any more.

### Task 1.10 — Stop hardcoding the channel count

- [x] Replace the literal `3` at `preprocess.go:188`, `preprocess.go:203` and
      `detector.go:143` with a value carried on the config path

**Accept:** `grep -rn "NewImageTensor(.*, 3," internal/` returns nothing.

### Task 1.11 — Declare the CTC layout instead of inferring it

- [ ] Add an `NTC`/`NCT` layout field to the recognizer config
- [ ] Delete `determineClassesFirst` (`inference.go:527-548`)
- [ ] Make `blankIndex` a field, replacing the literals at `inference.go:190`
      and `inference.go:501`
- [ ] Validate the declared layout against the model's actual output shape at
      load

**Accept:** synthetic-tensor tests decode correctly under both layouts, and a
declared layout contradicting the model is a load-time error rather than a
silent fallback that happens to be right.

_Phase 1 exit: all 14 fixtures pass at exact match. `TestProcessImage_Smoke`
reads `hello`, not `hel1o`. Both failing `TestRecognizeBatch` tests are green.
`go test ./...` passes apart from the PDF scenarios Phase 3 deletes, and no
package takes over 60 s._

## Phase 2 — Measure something real

### Task 2.1 — Take the thresholds away from the ground truth

- [ ] Reduce `ocr_accuracy.json` to a manifest of `image` and `expected` only
- [ ] Delete `MinSimilarity`, `MinCAR`, `MinWAR`, `MinAvgConf`, `MinContains`
      from the `accuracyCase` struct (`accuracy_test.go:17-26`)

**Accept:** grepping the manifest for `min_` returns nothing. A failing case can
no longer be fixed by editing its own row.

### Task 2.2 — Define one corpus-level gate

- [ ] Express the gate in code: mean CER, mean WER, exact-match count
- [ ] Verify no per-case escape hatch survives anywhere on the path

**Accept:** degrading any single case fails the corpus gate, and there is no knob
that silences it short of changing the gate itself.

### Task 2.3 — Promote the harness into `pogo eval`

- [ ] Move the CER/WER measurement out of `accuracy_test.go` into a command
- [ ] Print per-case rows plus the aggregate
- [ ] Keep the Go test as a thin caller so `go test` still gates the corpus

**Accept:** `pogo eval ./testdata/corpus` prints a table and exits 0.

### Task 2.4 — Commit a baseline and compare against it

- [ ] Write a baseline file per corpus
- [ ] Have `eval` compare and exit non-zero on regression, naming the metric
- [ ] Prove it by truncating the dictionary and watching it fail

**Accept:** a deliberate regression makes `pogo eval` exit 1 and say which metric
moved and by how much.

### Task 2.5 — Add a corpus that isn't synthetic

- [ ] Collect at least 10 real scans, German included
- [ ] Hand-key ground truth for each
- [ ] Commit them under `testdata/corpus/real/` beside their ground truth

**Accept:** every image is a real capture rather than a render. All 14 cases
today are synthetic, which is why a wrong normalization constant survived.

### Task 2.6 — Baseline the real corpus

- [ ] Run `pogo eval` over it and commit the resulting baseline
- [ ] Record the number in the commit message

**Accept:** `pogo eval ./testdata/corpus/real` exits 0 and its CER is recorded in
the repo, so the next change has something to be compared against.

_Phase 2 exit: one committed CER/WER baseline per corpus. Nothing merges
afterwards that moves CER the wrong way._

## Phase 3 — Cut

Strictly leaf-first — the order below is the dependency order, and taking it out
of order means editing the same code twice. Each task is one commit and ends with
the same three checks, which are therefore not repeated per task:

- [ ] `go build ./...` green
- [ ] `go test ./...` green
- [ ] `pogo eval` CER unchanged from the Phase 2 baseline

### Task 3.1 — Delete the empty and premature

- [ ] `pkg/ocr/`, `internal/version/`, `cmd/simple-model-test/` — all three are
      empty directories
- [ ] `deployment/` and `.goreleaser.yaml` — premature; `--version` prints a
      hardcoded string regardless

**Accept:** no build reference to any of them remains.

### Task 3.2 — Delete the benchmark binary and package

- [ ] `cmd/benchmark` (117 LOC) and `internal/benchmark` (540 LOC)
- [ ] Confirm `internal/benchmark` had exactly one caller before removing it

**Accept:** nothing imports the package and the binary is gone. `pogo eval`
replaces it, measuring accuracy rather than speed.

### Task 3.3 — Delete the batch command and package

- [ ] Re-home `pipeline.CalculateParallelStats` — `internal/batch/config.go` is
      its only external user, and Task 3.7 needs it gone from `parallel.go`
- [ ] `cmd/ocr/cmd/batch.go` (242 LOC) and `internal/batch` (646 LOC)

**Accept:** `pogo --help` no longer lists `batch`. A `for` loop over `Read()`
belongs in `cmd/`.

### Task 3.4 — Delete the server

- [ ] `cmd/ocr/cmd/serve.go` (452 LOC, ~106 flags) and `internal/server` (2,699)
- [ ] The four `server_*.feature` files and `test/integration/cli/support/server_*.go`
- [ ] `docs/openapi.yaml`, which documents only this API

**Accept:** gorilla/websocket and the Prometheus client leave `go.mod`.

### Task 3.5 — Delete PDF

- [ ] `cmd/ocr/cmd/pdf.go` (722 LOC, ~65 flags) and `internal/pdf` (2,834)
- [ ] `internal/pipeline/process_pdf.go`, the second of the two divergent
      implementations
- [ ] `pdf_processing.feature` and `testdata/documents/`

**Accept:** pdfcpu leaves `go.mod`. Nothing is fixed on the way out — the
one-line bug at `pdf.go:687` goes with the file.

### Task 3.6 — Delete barcodes and unblock `go mod tidy`

- [ ] `internal/barcode` (337 LOC) and `internal/pipeline/barcodes*.go` (237)
- [ ] The 12 barcode flags across the surviving commands
- [ ] Confirm the dependency actually leaves the module graph

**Accept:** **`go mod tidy -diff` exits 0.** This is the task that repairs
`just check-tidy`, which fails today because `gozxing_backend.go` imports
`gozxing/pdf417` — a package that exists in no published version of that module.

### Task 3.7 — Delete pipeline dead code

- [ ] `parallel.go` (411 LOC) — its four exported entry points have zero
      non-test callers
- [ ] `profile.go` (39) and `monitor.go` (27) — self-references only
- [ ] `ProcessImages`, keeping `ProcessImagesContext`
- [ ] `AdaptiveWorkerPool` from `resources.go` (~98 LOC), keeping
      `ResourceManager` and `MemoryMonitor` — `pipeline.go` wires both

**Accept:** `resources_test.go` shrinks rather than breaking. This is a partial
file delete, not a whole one; getting it wrong takes live code with it.

### Task 3.8 — Delete detector dead code

- [ ] `batch.go`'s `RunBatchInference` (202 LOC + 333 test)
- [ ] `benchmark.go` (141 LOC)
- [ ] `DefaultAdaptiveNMSThresholds` (`nms.go:25`)
- [ ] Leave `multiscale.go`, `adaptive_threshold.go` and the NMS variants alone

**Accept:** `--det-multiscale` still works and the YAML-configured adaptive NMS
path still has a test. Those paths have no CLI flag but are reachable through
config (`config/loader.go:271-272`, `detector/postprocess_onnx.go:66-96`).

### Task 3.9 — Delete rectify and orientation

- [ ] `internal/rectify` (978 LOC) — UVDoc is off by default and the DocTR path
      is dead, its model file absent
- [ ] `internal/orientation` (902 LOC) — off by default, four overlapping
      escape hatches
- [ ] Their fields in `internal/config`
- [ ] The three orientation call sites in `internal/recognizer`
- [ ] Salvage the homography and warping code into `internal/detect` for crop
      rectification before deleting the rest

**Accept:** the config struct loses every rectify and orientation field, and no
flag referencing them survives in `--help`.

### Task 3.10 — Delete mempool

- [ ] `internal/mempool`: 130 lines of source against 1,007 lines of tests
- [ ] Its 12 call sites across `internal/detector`, `internal/recognizer` and
      `internal/utils`
- [ ] Compare `pogo eval` wall-clock before and after

**Accept:** no material slowdown. If there is one, that is a real finding, and
pooling comes back with the benchmark that justifies it — which is more than it
has today.

### Task 3.11 — Make the godog suite mean something

- [ ] Delete the feature files for every subsystem removed above
- [ ] Write step definitions for what survives — 39 scenarios currently have
      none and are invisible
- [ ] Set `godog.Options{Strict: true}` so undefined steps fail
- [ ] Fix `just test-integration`: it runs `-run "Integration"` while the entry
      point is `TestFeatures`, so today it runs no scenarios at all

**Accept:** zero undefined and zero failing scenarios, and deleting any single
step definition turns the suite red. Today: 179 scenarios, 76 pass, 64 fail,
39 undefined and silent.

### Task 3.12 — Clear the lint backlog

- [ ] Work through the remaining golangci-lint findings in surviving code
- [ ] Expect the count to fall well below 131 on deletions alone; fix the rest

**Accept:** `just lint` exits 0.

_Phase 3 exit: `internal/` source under 6,000 lines. **`just check` exits 0** —
all four steps, for the first time. CER unchanged from Phase 2._

## Phase 4 — The library exists

### Task 4.1 — Define the seams

- [ ] Declare `Detector` and `Recognizer` in `pkg/pogo`
- [ ] Delete the dead test scaffolding at `parallel_test.go:322,330`, which
      declared such interfaces and never used them
- [ ] Write a fake of each

**Accept:** the fakes drive the pipeline in a unit test with no ONNX runtime
loaded at all.

### Task 4.2 — Make the pipeline hold interfaces

- [ ] Replace the concrete structs at `pipeline.go:512-521` with the interfaces
- [ ] Substitute the Task 4.1 fakes in a test

**Accept:** CER unchanged, and the pipeline can be exercised without models.

### Task 4.3 — Implement `Engine`

- [ ] `Open`, `Read`, `Close`
- [ ] `Result`, `Line`, `Quad`
- [ ] Document every exported symbol

**Accept:** `go doc ./pkg/pogo` fits on one screen.

### Task 4.4 — Reduce the CLI to a consumer

- [ ] Cut roughly 272 flag declarations to about 12
- [ ] Cut the config struct from 99 fields to about 15
- [ ] Give every surviving flag a test that exercises it

**Accept:** `pogo image --help` fits in one terminal page, and no flag exists
that nothing tests.

### Task 4.5 — Prove it is importable

- [ ] Add `example_test.go`
- [ ] Write a scratch-module smoke check outside the repo

**Accept:** a 20-line `main.go` in a module **outside** this repo imports pogo
and prints boxes with text and confidence.

_Phase 4 exit: that scratch module compiles and prints correct text._

## Phase 5 — The model is swappable

### Task 5.1 — `ModelSpec` and its loader

- [ ] Define the type: `kind`, `decode`, `input`, `ctc`, `charset`
- [ ] Write the YAML loader
- [ ] Implement `assert_classes` against the model's real output shape

**Accept:** a spec whose `assert_classes` does not match is a load-time error
naming both numbers. This subsumes the hand-rolled check from Task 1.5.

### Task 5.2 — Drive preprocessing and decoding from the spec

- [ ] Preprocessing reads `input:` — scale, mean, std, layout, channels, resize
- [ ] Decoding reads `ctc:` — `blank_index`, `layout`
- [ ] Charset loading reads `charset:` — including `append_space`

**Accept:** changing `mean` or `std` in the YAML changes the produced tensor,
with no recompile.

### Task 5.3 — Retire `internal/models/paths.go`

- [ ] Write sidecar specs for the four bundled models
- [ ] Delete the hardcoded filenames (`:13-29`)
- [ ] Delete the walk-up-for-`go.mod` heuristic (`:53-75`), which breaks for
      installed binaries
- [ ] Delete `ResolveModelPath`'s fallback to a flat path it never stats
      (`:108-128`)

**Accept:** a missing model produces an error naming the path searched, not
`Protobuf parsing failed`; and an installed binary run outside a Go module
resolves its models correctly.

### Task 5.4 — The model store

- [ ] Write a manifest of name → URL + sha256
- [ ] Implement `pogo models pull` into `~/.cache/pogo/models`
- [ ] Verify checksums and refuse to cache a bad download

**Accept:** with an empty cache, `pogo models pull && pogo image x.png` works;
a corrupted download fails its checksum and is not cached.

### Task 5.5 — Get the weights out of the working tree

- [ ] `git rm --cached` the 12 tracked `.onnx` files (229 MB)
- [ ] Fix the ignore pattern to `/models/**/*.onnx` — `/models/*.onnx` never
      matched the nested paths, which is exactly how they got committed
- [ ] Teach CI to pull models before testing
- [ ] No history rewrite: `.git` stays 274 MB, by decision, so no clone breaks

**Accept:** `git ls-files models | grep '\.onnx$'` is empty, and a fresh clone
plus `pogo models pull` passes `pogo eval`.

### Task 5.6 — Prove swappability

- [ ] Pick a structurally different recognizer
- [ ] Write its spec — around 15 lines of YAML
- [ ] Run it end to end

**Accept:** it works with **zero Go changes**. If any Go file had to change, the
spec is not yet the description of the model.

_Phase 5 exit: Task 5.6 passes and no weights remain in the working tree._

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
