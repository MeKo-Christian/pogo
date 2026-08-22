package eval

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func baselineFor(cer, wer float64, exact, n int) Baseline {
	groups, overall := summaries(cer, wer, exact, n)
	return NewBaseline("mobile", groups, overall)
}

func summaries(cer, wer float64, exact, n int) ([]GroupSummary, GroupSummary) {
	g := GroupSummary{Group: GroupUpright, N: n, Exact: exact, MeanCER: cer, MeanWER: wer}
	o := g
	o.Group = OverallGroup
	return []GroupSummary{g}, o
}

func TestBaselineRoundTrip(t *testing.T) {
	groups, overall := summaries(0.2, 0.6, 3, 9)
	path := filepath.Join(t.TempDir(), BaselineFileName)
	require.NoError(t, NewBaseline("mobile", groups, overall).Save(path))

	got, err := LoadBaseline(path)
	require.NoError(t, err)
	assert.Equal(t, "mobile", got.Models)
	require.Len(t, got.Groups, 2)
	assert.Equal(t, OverallGroup, got.Groups[1].Group)
}

func TestLoadBaselineRejectsAFileWithoutAModelSet(t *testing.T) {
	path := filepath.Join(t.TempDir(), BaselineFileName)
	require.NoError(t, os.WriteFile(path, []byte(`{"groups":[]}`), 0o600))
	_, err := LoadBaseline(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "which models")
}

func TestCompare(t *testing.T) {
	base := baselineFor(0.2, 0.6, 3, 9)

	tests := []struct {
		name           string
		cer, wer       float64
		exact          int
		wantChanges    int
		wantRegression bool
		wantMetric     string
	}{
		{"unchanged", 0.2, 0.6, 3, 0, false, ""},
		{"within tolerance", 0.202, 0.6, 3, 0, false, ""},
		{"CER regressed", 0.25, 0.6, 3, 2, true, MetricMeanCER},
		{"CER improved", 0.1, 0.6, 3, 2, false, MetricMeanCER},
		{"exact fraction fell", 0.2, 0.6, 2, 2, true, MetricExactFraction},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			groups, overall := summaries(tt.cer, tt.wer, tt.exact, 9)
			changes, err := base.Compare(groups, overall, "mobile", 0.005)
			require.NoError(t, err)
			// One change per group, and "overall" mirrors the single group here.
			assert.Len(t, changes, tt.wantChanges)
			assert.Len(t, Regressions(changes), map[bool]int{true: tt.wantChanges, false: 0}[tt.wantRegression])
			if tt.wantMetric != "" {
				assert.Contains(t, changes[0].String(), tt.wantMetric)
				assert.Contains(t, changes[0].String(), "->")
			}
		})
	}
}

// Numbers from different weights are not comparable, so a mismatch is an error
// rather than a regression that would send someone hunting for a bug.
func TestCompareRejectsADifferentModelSet(t *testing.T) {
	base := baselineFor(0.2, 0.6, 3, 9)
	groups, overall := summaries(0.2, 0.6, 3, 9)
	_, err := base.Compare(groups, overall, "server", 0.005)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "mobile")
	assert.Contains(t, err.Error(), "server")
}

// A group that stops being measured must not read as "nothing moved".
func TestCompareFlagsAMissingGroup(t *testing.T) {
	base := baselineFor(0.2, 0.6, 3, 9)
	base.Groups = append(base.Groups, GroupSummary{Group: GroupRotated, N: 5})
	groups, overall := summaries(0.2, 0.6, 3, 9)
	changes, err := base.Compare(groups, overall, "mobile", 0.005)
	require.NoError(t, err)
	regressions := Regressions(changes)
	require.Len(t, regressions, 1)
	assert.Contains(t, regressions[0].String(), "missing")
}
