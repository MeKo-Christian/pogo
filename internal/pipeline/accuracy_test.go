package pipeline

import (
    "encoding/json"
    "os"
    "image"
    _ "image/jpeg"
    _ "image/png"
    "strings"
    "testing"

    "github.com/MeKo-Tech/pogo/internal/models"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

type accuracyCase struct {
    Image         string  `json:"image"`
    Expected      string  `json:"expected"`
    MinSimilarity float64 `json:"min_similarity"`
    MinCAR        float64 `json:"min_car"`
    MinWAR        float64 `json:"min_war"`
    MinAvgConf    float64 `json:"min_avg_conf"`
    ContainsAny   []string `json:"contains_any"`
    MinContains   int      `json:"min_contains"`
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
    for j := range prev { prev[j] = j }
    for i := 1; i <= len(ra); i++ {
        cur[0] = i
        for j := 1; j <= len(rb); j++ {
            cost := 0
            if ra[i-1] != rb[j-1] { cost = 1 }
            // min of delete, insert, substitute
            del := prev[j] + 1
            ins := cur[j-1] + 1
            sub := prev[j-1] + cost
            if del < ins { if del < sub { cur[j] = del } else { cur[j] = sub } } else { if ins < sub { cur[j] = ins } else { cur[j] = sub } }
        }
        copy(prev, cur)
    }
    return prev[len(rb)]
}

func similarity(a, b string) float64 {
    a = strings.TrimSpace(strings.ToLower(a))
    b = strings.TrimSpace(strings.ToLower(b))
    if a == "" && b == "" { return 1 }
    d := float64(levenshtein(a, b))
    la := float64(len([]rune(a)))
    lb := float64(len([]rune(b)))
    maxLen := la
    if lb > maxLen { maxLen = lb }
    if maxLen == 0 { return 1 }
    return 1.0 - d/maxLen
}

func charAccuracyRate(a, b string) float64 { return similarity(a, b) }

func wordAccuracyRate(a, b string) float64 {
    aw := strings.Fields(strings.ToLower(a))
    bw := strings.Fields(strings.ToLower(b))
    if len(bw) == 0 && len(aw) == 0 { return 1 }
    if len(bw) == 0 { return 0 }
    // exact match ratio on expected words
    matched := 0
    // build set from aw
    set := make(map[string]int)
    for _, w := range aw { set[w]++ }
    for _, w := range bw {
        if set[w] > 0 { matched++; set[w]-- }
    }
    return float64(matched) / float64(len(bw))
}

// TestOCRAccuracy_SimpleFixtures validates text output against ground truth fixtures
// using a similarity threshold to be robust to minor decoding differences.
func TestOCRAccuracy_SimpleFixtures(t *testing.T) {
    // Ensure models exist; otherwise skip
    det := models.GetDetectionModelPath("", false)
    rec := models.GetRecognitionModelPath("", false)
    dict := models.GetDictionaryPath("", models.DictionaryPPOCRv5)
    for _, p := range []string{det, rec, dict} {
        if _, err := os.Stat(p); err != nil {
            t.Skipf("required model missing: %s", p)
        }
    }

    // Load fixtures
    data, err := os.ReadFile("testdata/fixtures/ocr_accuracy.json")
    require.NoError(t, err)
    var cases []accuracyCase
    require.NoError(t, json.Unmarshal(data, &cases))
    require.NotEmpty(t, cases)

    // Build pipeline
    b := NewBuilder().WithModelsDir(models.GetModelsDir(""))
    b.WithImageHeight(48)
    p, err := b.Build()
    if err != nil {
        t.Skipf("pipeline build failed (likely ONNX runtime issue): %v", err)
    }
    defer func() { _ = p.Close() }()

    for _, c := range cases {
        t.Run(c.Image, func(t *testing.T) {
            f, err := os.Open(c.Image)
            require.NoError(t, err)
            defer func() { _ = f.Close() }()
            img, _, err := image.Decode(f)
            require.NoError(t, err)
            res, err := p.ProcessImage(img)
            require.NoError(t, err)
            txt, err := ToPlainTextImage(res)
            require.NoError(t, err)
            sim := similarity(txt, c.Expected)
            assert.GreaterOrEqualf(t, sim, c.MinSimilarity, "similarity=%.3f text=%q expected=%q", sim, txt, c.Expected)
            car := charAccuracyRate(txt, c.Expected)
            assert.GreaterOrEqualf(t, car, c.MinCAR, "CAR=%.3f text=%q expected=%q", car, txt, c.Expected)
            war := wordAccuracyRate(txt, c.Expected)
            assert.GreaterOrEqualf(t, war, c.MinWAR, "WAR=%.3f text=%q expected=%q", war, txt, c.Expected)

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

            // Contains-any check (e.g., for specific characters like German umlauts)
            if len(c.ContainsAny) > 0 {
                lower := strings.ToLower(txt)
                found := 0
                for _, needle := range c.ContainsAny {
                    if strings.Contains(lower, strings.ToLower(needle)) {
                        found++
                    }
                }
                minNeedles := c.MinContains
                if minNeedles <= 0 {
                    minNeedles = 1
                }
                assert.GreaterOrEqualf(t, found, minNeedles, "expected at least %d occurrences from %v in %q", minNeedles, c.ContainsAny, txt)
            }
        })
    }
}
