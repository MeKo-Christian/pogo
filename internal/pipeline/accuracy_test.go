package pipeline

import (
	"encoding/json"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// accuracyCase is one ground-truth entry from testdata/fixtures/ocr_accuracy.json.
//
// Group tiers the corpus. "upright" cases are held to exact match; "rotated"
// cases keep similarity bars because the deskew packages that could make them
// exact (internal/rectify, internal/orientation) are scheduled for deletion in
// PLAN.md Task 3.9. Phase 3 decides whether they survive at all.
type accuracyCase struct {
	Group         string  `json:"group"`
	Image         string  `json:"image"`
	Expected      string  `json:"expected"`
	MinSimilarity float64 `json:"min_similarity"`
	MinCAR        float64 `json:"min_car"`
	MinWAR        float64 `json:"min_war"`
	MinAvgConf    float64 `json:"min_avg_conf"`
}

func levenshtein(a, b string) int {
	ra := []rune(a)
	rb := []rune(b)
	if len(ra) == 0 {
		return len(rb)
	}
	if len(rb) == 0 {
		return len(ra)
	}
	prev := make([]int, len(rb)+1)
	cur := make([]int, len(rb)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ra); i++ {
		cur[0] = i
		for j := 1; j <= len(rb); j++ {
			cost := 0
			if ra[i-1] != rb[j-1] {
				cost = 1
			}
			// min of delete, insert, substitute
			del := prev[j] + 1
			ins := cur[j-1] + 1
			sub := prev[j-1] + cost
			if del < ins {
				if del < sub {
					cur[j] = del
				} else {
					cur[j] = sub
				}
			} else {
				if ins < sub {
					cur[j] = ins
				} else {
					cur[j] = sub
				}
			}
		}
		copy(prev, cur)
	}
	return prev[len(rb)]
}

func similarity(a, b string) float64 {
	a = strings.TrimSpace(strings.ToLower(a))
	b = strings.TrimSpace(strings.ToLower(b))
	if a == "" && b == "" {
		return 1
	}
	d := float64(levenshtein(a, b))
	la := float64(len([]rune(a)))
	lb := float64(len([]rune(b)))
	maxLen := la
	if lb > maxLen {
		maxLen = lb
	}
	if maxLen == 0 {
		return 1
	}
	return 1.0 - d/maxLen
}

func charAccuracyRate(a, b string) float64 { return similarity(a, b) }

func wordAccuracyRate(a, b string) float64 {
	aw := strings.Fields(strings.ToLower(a))
	bw := strings.Fields(strings.ToLower(b))
	if len(bw) == 0 && len(aw) == 0 {
		return 1
	}
	if len(bw) == 0 {
		return 0
	}
	// exact match ratio on expected words
	matched := 0
	// build set from aw
	set := make(map[string]int)
	for _, w := range aw {
		set[w]++
	}
	for _, w := range bw {
		if set[w] > 0 {
			matched++
			set[w]--
		}
	}
	return float64(matched) / float64(len(bw))
}

// TestOCRAccuracy_SimpleFixtures validates text output against ground truth fixtures
// using a similarity threshold to be robust to minor decoding differences.
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

	// Load fixtures. Paths are resolved against the project root rather than
	// the package working directory, so no testdata symlink is required.
	root, err := testutil.GetProjectRoot()
	require.NoError(t, err)
	data, err := os.ReadFile(filepath.Join(testutil.GetFixturesDir(t), "ocr_accuracy.json"))
	require.NoError(t, err)
	var cases []accuracyCase
	require.NoError(t, json.Unmarshal(data, &cases))
	require.NotEmpty(t, cases)

	// Build pipeline
	b := NewBuilder().WithModelsDir(models.GetModelsDir(""))
	b.WithImageHeight(48)
	b.WithServerModels(useServer)
	p, err := b.Build()
	if err != nil {
		t.Skipf("pipeline build failed (likely ONNX runtime issue): %v", err)
	}
	defer func() { _ = p.Close() }()

	for _, c := range cases {
		t.Run(c.Group+"/"+c.Image, func(t *testing.T) {
			started := time.Now()
			var regionCount int
			var recognized string
			defer func() {
				t.Logf("elapsed=%s regions=%d text=%q",
					time.Since(started).Round(time.Millisecond), regionCount, recognized)
			}()
			//nolint:gosec // G304: the path comes from the checked-in fixture file, not from user input.
			f, err := os.Open(filepath.Join(root, c.Image))
			require.NoError(t, err)
			defer func() { _ = f.Close() }()
			img, _, err := image.Decode(f)
			require.NoError(t, err)
			res, err := p.ProcessImage(img)
			require.NoError(t, err)
			regionCount = len(res.Regions)
			txt, err := ToPlainTextImage(res)
			require.NoError(t, err)
			recognized = txt
			sim := similarity(txt, c.Expected)
			assert.GreaterOrEqualf(t, sim, c.MinSimilarity, "similarity=%.3f text=%q expected=%q", sim, txt, c.Expected)
			car := charAccuracyRate(txt, c.Expected)
			assert.GreaterOrEqualf(t, car, c.MinCAR, "CAR=%.3f text=%q expected=%q", car, txt, c.Expected)
			war := wordAccuracyRate(txt, c.Expected)
			assert.GreaterOrEqualf(t, war, c.MinWAR, "WAR=%.3f text=%q expected=%q", war, txt, c.Expected)

			// similarity, CAR and WAR all fold case, so a 1.0 bar on its own
			// would still accept "hello" for "Hello". Cases that ask for a
			// perfect score get a case-sensitive equality check on top,
			// normalizing only whitespace.
			if c.MinSimilarity >= 1.0 {
				assert.Equalf(t, c.Expected, strings.Join(strings.Fields(txt), " "),
					"exact match required, text=%q expected=%q", txt, c.Expected)
			}

			// Minimum average recognition confidence across regions (if present)
			if c.MinAvgConf > 0 {
				var sum float64
				var count int
				for _, r := range res.Regions {
					if strings.TrimSpace(r.Text) != "" {
						sum += r.RecConfidence
						count++
					}
				}
				if count > 0 {
					avg := sum / float64(count)
					assert.GreaterOrEqualf(t, avg, c.MinAvgConf, "avg_rec_conf=%.3f below min %.3f", avg, c.MinAvgConf)
				}
			}
		})
	}
}
