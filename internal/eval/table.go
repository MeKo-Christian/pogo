package eval

import (
	"fmt"
	"strings"
	"time"
)

// FormatResults renders the per-case rows followed by the per-group and overall
// aggregate. It is what a failing gate prints, so a red run can be diagnosed
// without running the corpus again.
func FormatResults(results []CaseResult, groups []GroupSummary, overall GroupSummary) string {
	var b strings.Builder
	b.WriteString("Per-case results:\n")
	b.WriteString("=================\n")
	for _, r := range results {
		mark := " "
		if !r.Exact {
			mark = "x"
		}
		fmt.Fprintf(&b, "%s %-9s %-44s CER=%.4f WER=%.4f conf=%.3f regions=%d %s\n",
			mark, r.Group, r.Image, r.CER, r.WER, r.AvgConf, r.Regions, r.Elapsed.Round(time.Millisecond))
		// The reading is printed for every case, not only the wrong ones, so a
		// geometry change that shifts a crop can be diffed run against run.
		fmt.Fprintf(&b, "      expected %q\n      got      %q\n", r.Expected, Normalize(r.Got))
	}
	b.WriteString("\nAggregate:\n")
	b.WriteString("==========\n")
	for _, g := range append(append([]GroupSummary{}, groups...), overall) {
		fmt.Fprintf(&b, "  %-9s n=%-3d exact=%d/%d mean CER=%.4f mean WER=%.4f\n",
			g.Group, g.N, g.Exact, g.N, g.MeanCER, g.MeanWER)
	}
	return b.String()
}
