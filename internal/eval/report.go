package eval

import (
	"sort"
	"time"
)

// CaseResult is one evaluated case: the ground truth, what the engine read, and
// the metrics between them.
type CaseResult struct {
	Case
	Got     string        `json:"got"`
	CER     float64       `json:"cer"`
	WER     float64       `json:"wer"`
	Exact   bool          `json:"exact"`
	AvgConf float64       `json:"avg_conf"`
	Regions int           `json:"regions"`
	Elapsed time.Duration `json:"elapsed_ns"`
}

// Score fills in the metrics for a reading. It is the only place a CaseResult
// learns whether it passed, so the command and the test cannot measure
// differently.
func Score(c Case, got string, regions int, avgConf float64, elapsed time.Duration) CaseResult {
	return CaseResult{
		Case:    c,
		Got:     got,
		CER:     CER(c.Expected, got),
		WER:     WER(c.Expected, got),
		Exact:   ExactMatch(c.Expected, got),
		AvgConf: avgConf,
		Regions: regions,
		Elapsed: elapsed,
	}
}

// GroupSummary aggregates the cases of one group. The group named
// OverallGroup aggregates the whole corpus.
type GroupSummary struct {
	Group   string  `json:"group"`
	N       int     `json:"n"`
	Exact   int     `json:"exact"`
	MeanCER float64 `json:"mean_cer"`
	MeanWER float64 `json:"mean_wer"`
}

// OverallGroup is the reserved group name for the corpus-wide summary.
const OverallGroup = "overall"

// ExactFraction is the share of cases read exactly right.
func (s GroupSummary) ExactFraction() float64 {
	if s.N == 0 {
		return 1
	}
	return float64(s.Exact) / float64(s.N)
}

// Summarize aggregates results per group, sorted by group name, and returns
// that alongside one corpus-wide summary. Sorting keeps the output
// deterministic across runs.
func Summarize(results []CaseResult) ([]GroupSummary, GroupSummary) {
	byGroup := make(map[string]*GroupSummary)
	overall := GroupSummary{Group: OverallGroup}
	var sumCER, sumWER float64
	for _, r := range results {
		g, ok := byGroup[r.Group]
		if !ok {
			g = &GroupSummary{Group: r.Group}
			byGroup[r.Group] = g
		}
		g.N++
		g.MeanCER += r.CER
		g.MeanWER += r.WER
		if r.Exact {
			g.Exact++
		}
		overall.N++
		sumCER += r.CER
		sumWER += r.WER
		if r.Exact {
			overall.Exact++
		}
	}
	names := make([]string, 0, len(byGroup))
	for name := range byGroup {
		names = append(names, name)
	}
	sort.Strings(names)
	groups := make([]GroupSummary, 0, len(names))
	for _, name := range names {
		g := byGroup[name]
		g.MeanCER /= float64(g.N)
		g.MeanWER /= float64(g.N)
		groups = append(groups, *g)
	}
	if overall.N > 0 {
		overall.MeanCER = sumCER / float64(overall.N)
		overall.MeanWER = sumWER / float64(overall.N)
	}
	return groups, overall
}
