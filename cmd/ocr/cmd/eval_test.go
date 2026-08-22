package cmd

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/eval"
	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// runEvalCmd exercises the command without loading any model: every case here
// must fail before the pipeline is built.
func runEvalCmd(t *testing.T, args ...string) error {
	t.Helper()
	root := GetRootCommand()
	// Command output is discarded: these cases fail before anything is printed.
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetErr(&out)
	// The flags bind to package-level state, so reset it rather than inherit
	// whatever the previous test left behind.
	evalOpts = evalOptions{format: outputFormatText, tolerance: 0.005}
	GetEvalCommand().Flags().Visit(func(f *pflag.Flag) { _ = f.Value.Set(f.DefValue) })
	t.Cleanup(func() {
		root.SetOut(nil)
		root.SetErr(nil)
		root.SetArgs(nil)
	})
	root.SetArgs(append([]string{"eval"}, args...))
	return root.Execute()
}

func TestEvalRejectsAnUnknownFormat(t *testing.T) {
	err := runEvalCmd(t, "testdata", "--format", "yaml")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid format")
}

func TestEvalRejectsADirectoryWithNoCorpus(t *testing.T) {
	dir := t.TempDir()
	err := runEvalCmd(t, dir)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no manifest.json")
}

func TestEvalRejectsAMissingDirectory(t *testing.T) {
	err := runEvalCmd(t, filepath.Join(t.TempDir(), "nowhere"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "corpus directory")
}

// One --baseline file cannot stand for several corpora.
func TestEvalRejectsOneBaselineForManyCorpora(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"a", "b"} {
		sub := filepath.Join(dir, name)
		require.NoError(t, os.MkdirAll(sub, 0o750))
		require.NoError(t, os.WriteFile(filepath.Join(sub, "manifest.json"), []byte("[]"), 0o600))
	}
	err := runEvalCmd(t, dir, "--baseline", filepath.Join(dir, "baseline.json"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "2 corpora")
}

// The bundled corpus must be discoverable where the command looks for it.
func TestEvalFindsTheBundledCorpus(t *testing.T) {
	manifests, err := findCorpora(filepath.Join("..", "..", "..", "testdata", "corpus"))
	require.NoError(t, err)
	require.Len(t, manifests, 1)
	assert.Contains(t, manifests[0], filepath.Join("synthetic", "manifest.json"))
}

// A negative tolerance would report every metric as moved, which is the
// opposite of what the flag is for.
func TestEvalRejectsANegativeTolerance(t *testing.T) {
	err := runEvalCmd(t, "testdata", "--tolerance", "-0.1")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "must not be negative")
}

// A failing run says which of the three things went wrong, because they lead to
// different next steps.
func TestFailureReasonsNameTheActualProblem(t *testing.T) {
	tests := []struct {
		name   string
		report evalReport
		want   string
	}{
		{"gate", evalReport{violations: []eval.Violation{{Group: "upright", Metric: "mean CER"}}}, "1 gate violation(s)"},
		{"regression", evalReport{changes: []eval.Change{{Group: "upright", Regression: true}}}, "1 regression(s)"},
		{"baseline", evalReport{baselineErr: assert.AnError}, "1 baseline(s) could not be compared"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reasons := failureReasons([]evalReport{tt.report})
			require.Len(t, reasons, 1)
			assert.Contains(t, reasons[0], tt.want)
		})
	}
	assert.Empty(t, failureReasons([]evalReport{{}}))
}
