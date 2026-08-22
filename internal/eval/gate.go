package eval

import "fmt"

// Gate is what a group of the corpus must achieve. There is one gate per group
// and no per-case knob anywhere: a case that regresses can only be answered by
// fixing the engine or by changing a gate here, in code, in a reviewable diff.
type Gate struct {
	// MaxMeanCER and MaxMeanWER bound the group's mean error rates.
	MaxMeanCER float64
	MaxMeanWER float64
	// MinExactFraction is the share of cases that must be read exactly right.
	// A fraction rather than a count, so adding a case does not silently
	// weaken the bar.
	MinExactFraction float64
}

// Gates maps group name to its gate.
//
// "upright" is held to exact match: the nine upright fixtures are clean renders
// of short strings, and there is no reason for any of them to be off by a
// character. It does not pass today — the corpus reads "Wor1d", "Samele",
// "Test.", "Haloet" and "3a ScaonedHcument", so 4 of 9 are exact and mean CER
// is 0.1867. PLAN.md's Phase 1 exit line claims otherwise; the measurement says
// it is not met. That failure is the point: the bar states what must be true,
// not what happens to be true, and there is no longer any row to edit to make
// it green.
//
// "rotated" is measured, not aspirational. The five rotated fixtures cannot be
// read without the deskew machinery in internal/rectify and
// internal/orientation, which PLAN.md Task 3.9 deletes; two of them return
// nothing at all. Its numbers are set at what the engine achieves today, with
// no headroom, so the group cannot get worse unnoticed. Phase 3 decides whether
// these cases survive at all.
//
// The rotated mean CER is written as the exact fraction it is. Every rotated
// expectation is twelve runes long, so across five cases the group mean can
// only land on a multiple of 1/60; the measured value is 50/60, and a rounded
// literal such as 0.8333 would sit below it and fail the unrounded comparison
// in Check.
var Gates = map[string]Gate{
	"upright": {MaxMeanCER: 0, MaxMeanWER: 0, MinExactFraction: 1.0},
	"rotated": {MaxMeanCER: 5.0 / 6.0, MaxMeanWER: 1.0, MinExactFraction: 0},
}

// Violation is one broken gate condition, named so the failure says which
// metric moved and by how much.
type Violation struct {
	Group  string
	Metric string
	Want   float64
	Got    float64
}

// unknownGroupMetric marks a violation that is not a missed number but a group
// the gate table does not know. Its Want and Got carry no meaning.
const unknownGroupMetric = "unknown group (no gate defined)"

func (v Violation) String() string {
	if v.Metric == unknownGroupMetric {
		return fmt.Sprintf("group %q: %s", v.Group, v.Metric)
	}
	rel := "at most"
	if v.Metric == "exact fraction" {
		rel = "at least"
	}
	return fmt.Sprintf("group %q: %s = %.4f, want %s %.4f", v.Group, v.Metric, v.Got, rel, v.Want)
}

// Check reports every way the summary misses the gate. An empty slice is a pass.
func (g Gate) Check(s GroupSummary) []Violation {
	var vs []Violation
	if s.MeanCER > g.MaxMeanCER {
		vs = append(vs, Violation{s.Group, "mean CER", g.MaxMeanCER, s.MeanCER})
	}
	if s.MeanWER > g.MaxMeanWER {
		vs = append(vs, Violation{s.Group, "mean WER", g.MaxMeanWER, s.MeanWER})
	}
	if f := s.ExactFraction(); f < g.MinExactFraction {
		vs = append(vs, Violation{s.Group, "exact fraction", g.MinExactFraction, f})
	}
	return vs
}

// CheckAll applies the gate of every group present in the results. A group with
// no gate is a violation rather than a skipped check, so a typo in a manifest
// cannot silence a case.
func CheckAll(groups []GroupSummary) []Violation {
	var vs []Violation
	for _, s := range groups {
		g, ok := Gates[s.Group]
		if !ok {
			vs = append(vs, Violation{s.Group, unknownGroupMetric, 0, 0})
			continue
		}
		vs = append(vs, g.Check(s)...)
	}
	return vs
}
