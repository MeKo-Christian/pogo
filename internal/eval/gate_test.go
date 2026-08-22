package eval

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func results(group string, pairs ...[2]string) []CaseResult {
	out := make([]CaseResult, 0, len(pairs))
	for i, p := range pairs {
		c := Case{Group: group, Image: "img", Expected: p[0]}
		_ = i
		out = append(out, Score(c, p[1], 1, 0.9, time.Millisecond))
	}
	return out
}

func TestSummarize(t *testing.T) {
	rs := append(
		results("upright", [2]string{hello, hello}, [2]string{world, "Werld"}),
		results("rotated", [2]string{rotatedText, ""})...,
	)
	groups, overall := Summarize(rs)
	require.Len(t, groups, 2)
	// Sorted by name, so the output is stable across runs.
	assert.Equal(t, "rotated", groups[0].Group)
	assert.Equal(t, "upright", groups[1].Group)

	up := groups[1]
	assert.Equal(t, 2, up.N)
	assert.Equal(t, 1, up.Exact)
	assert.InDelta(t, 0.1, up.MeanCER, 1e-9) // (0 + 1/5) / 2
	assert.InDelta(t, 0.5, up.ExactFraction(), 1e-9)

	assert.Equal(t, OverallGroup, overall.Group)
	assert.Equal(t, 3, overall.N)
	assert.Equal(t, 1, overall.Exact)
}

func TestGateCheck(t *testing.T) {
	g := Gate{MaxMeanCER: 0.1, MaxMeanWER: 0.2, MinExactFraction: 0.5}
	assert.Empty(t, g.Check(GroupSummary{Group: "g", N: 2, Exact: 1, MeanCER: 0.1, MeanWER: 0.2}))

	vs := g.Check(GroupSummary{Group: "g", N: 2, Exact: 0, MeanCER: 0.3, MeanWER: 0.4})
	require.Len(t, vs, 3)
	assert.Contains(t, vs[0].String(), "mean CER")
	assert.Contains(t, vs[0].String(), "0.3000")
	assert.Contains(t, vs[2].String(), "at least")
}

// A group with no gate must fail rather than pass silently: a typo in a
// manifest's group field would otherwise remove its cases from every check.
func TestCheckAllRejectsUnknownGroup(t *testing.T) {
	vs := CheckAll([]GroupSummary{{Group: "uprihgt", N: 1, Exact: 1}})
	require.Len(t, vs, 1)
	assert.Contains(t, vs[0].String(), "unknown group")
}

// The upright gate is exact match. Degrading a single case must fail it, and no
// data outside this package can soften that.
func TestUprightGateRejectsASingleWrongCharacter(t *testing.T) {
	perfect := results(
		"upright",
		[2]string{hello, hello},
		[2]string{world, world},
	)
	groups, _ := Summarize(perfect)
	assert.Empty(t, CheckAll(groups))

	degraded := results(
		"upright",
		[2]string{hello, hellg},
		[2]string{world, world},
	)
	groups, _ = Summarize(degraded)
	vs := CheckAll(groups)
	assert.NotEmpty(t, vs, "one wrong character must fail the upright gate")
}

func TestFormatResultsShowsExpectedAndGot(t *testing.T) {
	rs := results("upright", [2]string{hello, hellg})
	groups, overall := Summarize(rs)
	out := FormatResults(rs, groups, overall)
	assert.Contains(t, out, `expected "`+hello+`"`)
	assert.Contains(t, out, `got      "`+hellg+`"`)
	assert.Contains(t, out, "overall")
}
