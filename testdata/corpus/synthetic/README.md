# Synthetic corpus

Fourteen rendered fixtures with hand-checked ground truth, measured by
`pogo eval ./testdata/corpus` and by `TestOCRAccuracy_SimpleFixtures`.

`manifest.json` carries ground truth and nothing else — no thresholds. What
counts as a pass lives in `internal/eval/gate.go`, per group, so a failing case
cannot be answered by editing its own row.

The images are **copies** of what `cmd/generate-test-data` and
`internal/testutil.GenerateTestImages` write under `testdata/images/`, not
references to them. Two reasons:

- A corpus resolves image paths against its own manifest, so it has to contain
  them to be movable; the loader rejects a path that escapes the corpus.
- `GenerateTestImages` rewrites `testdata/images/**` whenever the testutil suite
  runs. Ground truth that an unrelated test can regenerate is not ground truth —
  PLAN.md records a past regeneration that quietly invalidated the bars.

`baseline.json` is the committed measurement, recorded with
`pogo eval ./testdata/corpus --update-baseline`. It names the weights it was
measured with by variant _and_ by content fingerprint, so numbers from a
different `--models-dir` cannot be compared against it by accident.

Every case here is a render. That is the weakness this corpus cannot fix: see
issue #8 for the real, non-synthetic corpus.
