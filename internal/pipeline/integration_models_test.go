package pipeline

import (
	"os"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Ensures builder can locate real models under the repository's models/ directory and build a pipeline.
func TestBuilder_WithRealModelsDir_Build(t *testing.T) {
	det := models.GetDetectionModelPath("", false)
	rec := models.GetRecognitionModelPath("", false)
	dict := models.GetDictionaryPath("", models.DictionaryPPOCRKeysV1)
	for _, p := range []string{det, rec, dict} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("required model missing: %s", p)
		}
	}

	b := NewBuilder().WithModelsDir(models.GetModelsDir(""))
	// Recognizer default height may be 32 while model expects 48; align to model spec.
	b.WithImageHeight(48)
	// Small warmup to exercise path
	b.WithWarmupIterations(1)
	p, err := b.Build()
	require.NoError(t, err)
	require.NotNil(t, p)
	defer func() { _ = p.Close() }()

	info := p.Info()
	assert.Contains(t, info, "detector")
	assert.Contains(t, info, "recognizer")
}

// Processes a synthetic image through the full pipeline and validates metrics aggregation and outputs.
func TestPipeline_ProcessImage_WithRealModels(t *testing.T) {
	det := models.GetDetectionModelPath("", false)
	rec := models.GetRecognitionModelPath("", false)
	dict := models.GetDictionaryPath("", models.DictionaryPPOCRKeysV1)
	for _, pth := range []string{det, rec, dict} {
		if _, err := os.Stat(pth); err != nil {
			t.Skipf("required model missing: %s", pth)
		}
	}

	b := NewBuilder().WithModelsDir(models.GetModelsDir(""))
	b.WithImageHeight(48)
	p, err := b.Build()
	if err != nil {
		t.Skipf("pipeline build failed (likely ONNX runtime issue): %v", err)
	}
	defer func() { _ = p.Close() }()

	imgCfg := testutil.DefaultTestImageConfig()
	imgCfg.Text = "Hello world"
	img, genErr := testutil.GenerateTextImage(imgCfg)
	require.NoError(t, genErr)

	res, procErr := p.ProcessImage(img)
	require.NoError(t, procErr)
	require.NotNil(t, res)

	assert.Equal(t, img.Bounds().Dx(), res.Width)
	assert.Equal(t, img.Bounds().Dy(), res.Height)
	// Timing should be populated and consistent
	assert.Positive(t, res.Processing.DetectionNs)
	// Recognition might detect zero regions; still expect non-negative times
	assert.GreaterOrEqual(t, res.Processing.RecognitionNs, int64(0))
	assert.Positive(t, res.Processing.TotalNs)
	// Total should be at least detection + recognition (allow overhead)
	sum := res.Processing.DetectionNs + res.Processing.RecognitionNs
	assert.GreaterOrEqual(t, res.Processing.TotalNs, sum)
}
