package eval

import (
	"fmt"
	"image"
	_ "image/jpeg" // fixtures are PNG or JPEG
	_ "image/png"
	"os"
	"time"
)

// Reading is what an engine returns for one image: the text, how many regions
// produced it, and the mean recognition confidence across them.
type Reading struct {
	Text    string
	Regions int
	AvgConf float64
}

// RecognizeFunc reads one image. It is the only thing eval needs from an OCR
// engine, which is what lets the whole eval path be tested with no models
// loaded at all.
type RecognizeFunc func(img image.Image) (Reading, error)

// Run evaluates every case of a corpus, resolving image paths against the
// corpus directory, and returns one scored result per case in manifest order.
func Run(root string, cases []Case, rec RecognizeFunc) ([]CaseResult, error) {
	results := make([]CaseResult, 0, len(cases))
	for _, c := range cases {
		r, err := runCase(root, c, rec)
		if err != nil {
			return nil, err
		}
		results = append(results, r)
	}
	return results, nil
}

func runCase(root string, c Case, rec RecognizeFunc) (CaseResult, error) {
	started := time.Now()
	p := c.Path(root)
	f, err := os.Open(p) //nolint:gosec // G304: path comes from the corpus manifest, not from a request.
	if err != nil {
		return CaseResult{}, fmt.Errorf("open %s: %w", p, err)
	}
	defer func() { _ = f.Close() }()
	img, _, err := image.Decode(f)
	if err != nil {
		return CaseResult{}, fmt.Errorf("decode %s: %w", p, err)
	}
	reading, err := rec(img)
	if err != nil {
		return CaseResult{}, fmt.Errorf("recognize %s: %w", p, err)
	}
	return Score(c, reading.Text, reading.Regions, reading.AvgConf, time.Since(started)), nil
}
