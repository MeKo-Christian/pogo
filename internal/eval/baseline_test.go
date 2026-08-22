package eval

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	mobileModels = ModelIdentity{Variant: variantMobile, Fingerprint: "0123456789abcdef"}
	serverModels = ModelIdentity{Variant: "server", Fingerprint: "fedcba9876543210"}
	// Same variant, different weights: --models-dir pointed somewhere else.
	otherMobileModels = ModelIdentity{Variant: variantMobile, Fingerprint: "aaaaaaaaaaaaaaaa"}
)

// baselineFor is the reference measurement the Compare tests move away from.
func baselineFor() Baseline {
	groups, overall := summaries(0.2, 0.6, 3)
	return NewBaseline(mobileModels, groups, overall)
}

// summaries builds one group of nine cases plus its overall mirror.
func summaries(cer, wer float64, exact int) ([]GroupSummary, GroupSummary) {
	const n = 9
	g := GroupSummary{Group: GroupUpright, N: n, Exact: exact, MeanCER: cer, MeanWER: wer}
	o := g
	o.Group = OverallGroup
	return []GroupSummary{g}, o
}

func TestBaselineRoundTrip(t *testing.T) {
	groups, overall := summaries(0.2, 0.6, 3)
	path := filepath.Join(t.TempDir(), BaselineFileName)
	require.NoError(t, NewBaseline(mobileModels, groups, overall).Save(path))

	got, err := LoadBaseline(path)
	require.NoError(t, err)
	assert.Equal(t, mobileModels, got.Models)
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
	base := baselineFor()

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
			groups, overall := summaries(tt.cer, tt.wer, tt.exact)
			changes, err := base.Compare(groups, overall, mobileModels, 0.005)
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
	base := baselineFor()
	groups, overall := summaries(0.2, 0.6, 3)
	_, err := base.Compare(groups, overall, serverModels, 0.005)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "mobile")
	assert.Contains(t, err.Error(), "server")
}

// A group that stops being measured must not read as "nothing moved".
func TestCompareFlagsAMissingGroup(t *testing.T) {
	base := baselineFor()
	base.Groups = append(base.Groups, GroupSummary{Group: GroupRotated, N: 5})
	groups, overall := summaries(0.2, 0.6, 3)
	changes, err := base.Compare(groups, overall, mobileModels, 0.005)
	require.NoError(t, err)
	regressions := Regressions(changes)
	require.Len(t, regressions, 1)
	assert.Contains(t, regressions[0].String(), "missing from this run")
}

// The variant alone is not an identity: --models-dir can point "mobile" at
// entirely different weights, and comparing across them would read as an
// accuracy change.
func TestCompareRejectsTheSameVariantWithDifferentWeights(t *testing.T) {
	base := baselineFor()
	groups, overall := summaries(0.2, 0.6, 3)
	_, err := base.Compare(groups, overall, otherMobileModels, 0.005)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "0123456789abcdef")
	assert.Contains(t, err.Error(), "aaaaaaaaaaaaaaaa")
}

// A group the baseline has never seen is reported, but it is not a regression:
// there is nothing to compare it against yet. It is still gated, and it moves
// the corpus-wide summary, which is compared.
func TestCompareReportsANewGroupWithoutCallingItARegression(t *testing.T) {
	base := baselineFor()
	groups, overall := summaries(0.2, 0.6, 3)
	groups = append(groups, GroupSummary{Group: GroupRotated, N: 5, MeanCER: 0.5, MeanWER: 1})

	changes, err := base.Compare(groups, overall, mobileModels, 0.005)
	require.NoError(t, err)
	require.Len(t, changes, 1)
	assert.Equal(t, GroupRotated, changes[0].Group)
	assert.Contains(t, changes[0].String(), "not in the baseline")
	assert.Empty(t, Regressions(changes))
}
