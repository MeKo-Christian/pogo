package eval

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

// BaselineFileName is the file a corpus keeps its committed numbers in.
const BaselineFileName = "baseline.json"

// Baseline is a committed measurement of one corpus: what the engine achieved,
// so that the next change has something to be compared against.
type Baseline struct {
	// Models records which weights produced these numbers ("mobile" or
	// "server"). Comparing across model sets is an error, not a regression.
	Models string `json:"models"`
	// Groups holds one summary per group plus the corpus-wide "overall".
	Groups []GroupSummary `json:"groups"`
}

// NewBaseline builds a baseline from a measured run.
func NewBaseline(models string, groups []GroupSummary, overall GroupSummary) Baseline {
	all := make([]GroupSummary, 0, len(groups)+1)
	all = append(all, groups...)
	all = append(all, overall)
	return Baseline{Models: models, Groups: all}
}

// LoadBaseline reads a baseline file.
func LoadBaseline(path string) (Baseline, error) {
	var b Baseline
	data, err := os.ReadFile(path) //nolint:gosec // G304: path comes from the operator, not from a request.
	if err != nil {
		return b, fmt.Errorf("read baseline %s: %w", path, err)
	}
	if err := json.Unmarshal(data, &b); err != nil {
		return b, fmt.Errorf("parse baseline %s: %w", path, err)
	}
	if b.Models == "" {
		return b, fmt.Errorf("baseline %s does not record which models it was measured with", path)
	}
	return b, nil
}

// Save writes the baseline, indented so its diffs are readable in review.
func (b Baseline) Save(path string) error {
	data, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return fmt.Errorf("encode baseline: %w", err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o600); err != nil {
		return fmt.Errorf("write baseline %s: %w", path, err)
	}
	return nil
}

func (b Baseline) group(name string) (GroupSummary, bool) {
	for _, g := range b.Groups {
		if g.Group == name {
			return g, true
		}
	}
	return GroupSummary{}, false
}

// Change is one metric that moved between the baseline and the current run.
type Change struct {
	Group      string
	Metric     string
	Was, Now   float64
	Regression bool
}

func (c Change) String() string {
	direction := "improved"
	if c.Regression {
		direction = "regressed"
	}
	return fmt.Sprintf("group %q: %s %s %.4f -> %.4f (delta %+.4f)",
		c.Group, c.Metric, direction, c.Was, c.Now, c.Now-c.Was)
}

// Compare reports every metric that moved by more than tolerance, in either
// direction. A group present in the baseline but missing from the run — or the
// reverse — is reported as a regression, because a silently dropped group is
// how a corpus stops measuring anything.
func (b Baseline) Compare(groups []GroupSummary, overall GroupSummary,
	models string, tolerance float64,
) ([]Change, error) {
	if b.Models != models {
		return nil, fmt.Errorf("baseline was measured with %q models but this run used %q; "+
			"re-run with the same models or record a separate baseline", b.Models, models)
	}
	current := append(append([]GroupSummary{}, groups...), overall)
	var changes []Change
	seen := make(map[string]bool, len(current))
	for _, cur := range current {
		seen[cur.Group] = true
		was, ok := b.group(cur.Group)
		if !ok {
			changes = append(changes, Change{Group: cur.Group, Metric: "group", Regression: false})
			continue
		}
		changes = append(changes, compareMetrics(was, cur, tolerance)...)
	}
	for _, was := range b.Groups {
		if !seen[was.Group] {
			changes = append(changes, Change{
				Group: was.Group, Metric: "group missing from this run", Was: float64(was.N), Regression: true,
			})
		}
	}
	sort.SliceStable(changes, func(i, j int) bool { return changes[i].Group < changes[j].Group })
	return changes, nil
}

func compareMetrics(was, now GroupSummary, tolerance float64) []Change {
	var changes []Change
	// Error rates: up is worse.
	for _, m := range []struct {
		name     string
		was, now float64
	}{
		{MetricMeanCER, was.MeanCER, now.MeanCER},
		{MetricMeanWER, was.MeanWER, now.MeanWER},
	} {
		if diff := m.now - m.was; diff > tolerance || diff < -tolerance {
			changes = append(changes, Change{
				Group: now.Group, Metric: m.name, Was: m.was, Now: m.now, Regression: diff > 0,
			})
		}
	}
	// Exact fraction: down is worse.
	wf, nf := was.ExactFraction(), now.ExactFraction()
	if diff := nf - wf; diff > tolerance || diff < -tolerance {
		changes = append(changes, Change{
			Group: now.Group, Metric: MetricExactFraction, Was: wf, Now: nf, Regression: diff < 0,
		})
	}
	return changes
}

// Regressions filters the changes down to the ones that made things worse.
func Regressions(changes []Change) []Change {
	var out []Change
	for _, c := range changes {
		if c.Regression {
			out = append(out, c)
		}
	}
	return out
}
