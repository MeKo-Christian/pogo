package eval

import (
	"errors"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// writeCorpus builds a throwaway corpus of solid-colour PNGs. The images only
// have to decode: the fake recognizer decides what they "read".
func writeCorpus(t *testing.T, cases []Case) string {
	t.Helper()
	dir := t.TempDir()
	for _, c := range cases {
		p := filepath.Join(dir, c.Image)
		require.NoError(t, os.MkdirAll(filepath.Dir(p), 0o750))
		img := image.NewRGBA(image.Rect(0, 0, 2, 2))
		img.Set(0, 0, color.White)
		f, err := os.Create(p) //nolint:gosec // G304: path is inside the test's temp dir.
		require.NoError(t, err)
		require.NoError(t, png.Encode(f, img))
		require.NoError(t, f.Close())
	}
	return dir
}

// The whole eval path runs with no ONNX runtime and no models: the engine
// enters through one function.
func TestRunWithAFakeEngine(t *testing.T) {
	cases := []Case{
		{Group: GroupUpright, Image: imgA, Expected: hello},
		{Group: GroupUpright, Image: "sub/b.png", Expected: world},
	}
	dir := writeCorpus(t, cases)

	readings := []string{hello, "Wor1d"}
	i := 0
	results, err := Run(dir, cases, func(image.Image) (Reading, error) {
		r := Reading{Text: readings[i], Regions: 1, AvgConf: 0.9}
		i++
		return r, nil
	})
	require.NoError(t, err)
	require.Len(t, results, 2)

	assert.True(t, results[0].Exact)
	assert.InDelta(t, 0.0, results[0].CER, 1e-9)
	assert.False(t, results[1].Exact)
	assert.InDelta(t, 0.2, results[1].CER, 1e-9)
	assert.Positive(t, results[0].Elapsed)
}

func TestRunReportsEngineFailure(t *testing.T) {
	cases := []Case{{Group: GroupUpright, Image: imgA, Expected: hello}}
	dir := writeCorpus(t, cases)
	_, err := Run(dir, cases, func(image.Image) (Reading, error) {
		return Reading{}, errors.New("session closed")
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "session closed")
	assert.Contains(t, err.Error(), imgA)
}

func TestRunReportsAnUndecodableImage(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(dir, imgA), []byte("not a png"), 0o600))
	cases := []Case{{Group: GroupUpright, Image: imgA, Expected: hello}}
	_, err := Run(dir, cases, func(image.Image) (Reading, error) { return Reading{}, nil })
	require.Error(t, err)
	assert.Contains(t, err.Error(), "decode")
}

// imgA is the fixture name reused by the corpora these tests build.
const imgA = "a.png"
