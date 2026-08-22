package pipeline

import (
	"image"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/eval"
	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestOCRAccuracy_SimpleFixtures measures the fixture corpus and holds it to the
// gates declared in internal/eval.
//
// The corpus manifest carries ground truth only. There is deliberately no
// per-case threshold: a case that reads wrong cannot be answered by editing its
// own row, only by fixing the engine or by moving a gate in eval.Gates, in code,
// in a reviewable diff.
func TestOCRAccuracy_SimpleFixtures(t *testing.T) {
	// The pipeline builder defaults to the mobile variants, so gate on the models
	// that are actually loaded. Set POGO_ACCURACY_MODELS=server to run the server
	// weights instead.
	useServer := os.Getenv("POGO_ACCURACY_MODELS") == "server"
	det := models.GetDetectionModelPath("", useServer)
	rec := models.GetRecognitionModelPath("", useServer)
	dict := models.GetDictionaryPath("", models.DictionaryPPOCRv5)
	for _, p := range []string{det, rec, dict} {
		// The env var only selects between two bundled model constants, so the
		// path is not attacker-controlled.
		//nolint:gosec // G703: path built from bundled model constants, not user input.
		if _, err := os.Stat(p); err != nil {
			t.Skipf("required model missing: %s", p)
		}
	}

	// The corpus is located from the project root rather than the package working
	// directory, so no testdata symlink is required; inside it, image paths are
	// resolved against the corpus itself.
	root, err := testutil.GetProjectRoot()
	require.NoError(t, err)
	corpus := filepath.Join(root, "testdata", "corpus", "synthetic")
	cases, err := eval.LoadManifest(filepath.Join(corpus, eval.ManifestFileName))
	require.NoError(t, err)

	b := NewBuilder().WithModelsDir(models.GetModelsDir(""))
	b.WithImageHeight(48)
	b.WithServerModels(useServer)
	p, err := b.Build()
	if err != nil {
		t.Skipf("pipeline build failed (likely ONNX runtime issue): %v", err)
	}
	defer func() { _ = p.Close() }()

	// The command and the test measure through the same code: this test is a
	// thin caller of eval.Run, and `pogo eval` is another.
	results, err := eval.Run(corpus, cases, recognizeWith(p))
	require.NoError(t, err)

	groups, overall := eval.Summarize(results)
	// Logged on every run, not only on failure: this table is the measurement,
	// and it is what the gates in eval.Gates were set from.
	t.Log("\n" + eval.FormatResults(results, groups, overall))

	violations := eval.CheckAll(groups)
	for _, v := range violations {
		t.Error(v.String())
	}
	assert.Emptyf(t, violations, "corpus gate failed; see the table above for the offending cases")
}

// recognizeWith adapts a pipeline to the eval seam.
func recognizeWith(p *Pipeline) eval.RecognizeFunc {
	return func(img image.Image) (eval.Reading, error) {
		res, err := p.ProcessImage(img)
		if err != nil {
			return eval.Reading{}, err
		}
		txt, err := ToPlainTextImage(res)
		if err != nil {
			return eval.Reading{}, err
		}
		return eval.Reading{Text: txt, Regions: len(res.Regions), AvgConf: avgRecConfidence(res)}, nil
	}
}

// avgRecConfidence averages recognition confidence over the regions that
// produced text. It is reported rather than gated: every fixture carried
// min_avg_conf: 0.0, so the old assertion never once executed.
func avgRecConfidence(res *OCRImageResult) float64 {
	var sum float64
	var n int
	for _, r := range res.Regions {
		if strings.TrimSpace(r.Text) != "" {
			sum += r.RecConfidence
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}
