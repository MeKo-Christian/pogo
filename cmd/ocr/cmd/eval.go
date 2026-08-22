package cmd

import (
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/MeKo-Tech/pogo/internal/eval"
	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/pipeline"
	"github.com/spf13/cobra"
)

const defaultCorpusDir = "./testdata/corpus"

type evalOptions struct {
	baseline       string
	updateBaseline bool
	format         string
	serverModels   bool
	tolerance      float64
}

var evalOpts evalOptions

// evalCmd measures OCR accuracy against a ground-truth corpus.
var evalCmd = &cobra.Command{
	Use:   "eval [corpus-dir]",
	Short: "Measure OCR accuracy against a ground-truth corpus",
	Long: `Run every case of a ground-truth corpus and report character and word error
rates, per case and in aggregate.

A corpus is a directory holding a manifest.json of ground truth. Any directory
given is searched for corpora, so a tree of them can be evaluated at once.

The corpus carries ground truth only. What counts as a pass lives in code
(internal/eval), so a failing case cannot be answered by editing its own row.
Results are additionally compared against the corpus baseline; a regression
beyond --tolerance exits non-zero and names the metric that moved.

Examples:
  pogo eval ./testdata/corpus
  pogo eval ./testdata/corpus --format json
  pogo eval ./testdata/corpus --update-baseline`,
	Args:         cobra.MaximumNArgs(1),
	SilenceUsage: true,
	RunE:         runEval,
}

func init() {
	rootCmd.AddCommand(evalCmd)
	f := evalCmd.Flags()
	f.StringVar(&evalOpts.baseline, "baseline", "",
		"baseline file to compare against (default: baseline.json inside each corpus)")
	f.BoolVar(&evalOpts.updateBaseline, "update-baseline", false,
		"write the measured numbers back to the baseline instead of comparing")
	f.StringVar(&evalOpts.format, "format", outputFormatText, "output format: text or json")
	f.BoolVar(&evalOpts.serverModels, "server-models", false, "use the server model variants instead of mobile")
	f.Float64Var(&evalOpts.tolerance, "tolerance", 0.005,
		"absolute metric movement tolerated before a change is reported")
}

// GetEvalCommand returns the eval command for testing purposes.
func GetEvalCommand() *cobra.Command { return evalCmd }

func runEval(cmd *cobra.Command, args []string) error {
	dir := defaultCorpusDir
	if len(args) == 1 {
		dir = args[0]
	}
	manifests, err := resolveCorpora(dir)
	if err != nil {
		return err
	}

	recognize, closePipeline, err := buildEvalRecognizer()
	if err != nil {
		return err
	}
	defer closePipeline()

	var failed bool
	for _, m := range manifests {
		ok, err := evalCorpus(cmd, m, recognize)
		if err != nil {
			return err
		}
		if !ok {
			failed = true
		}
	}
	if failed {
		return errors.New("evaluation failed: see the violations above")
	}
	return nil
}

// evalCorpus runs one corpus and reports whether it passed.
func evalCorpus(cmd *cobra.Command, manifest string, recognize eval.RecognizeFunc) (bool, error) {
	out := cmd.OutOrStdout()
	dir := filepath.Dir(manifest)
	cases, err := eval.LoadManifest(manifest)
	if err != nil {
		return false, err
	}
	results, err := eval.Run(dir, cases, recognize)
	if err != nil {
		return false, err
	}
	groups, overall := eval.Summarize(results)
	violations := eval.CheckAll(groups)

	baselinePath := evalOpts.baseline
	if baselinePath == "" {
		baselinePath = filepath.Join(dir, eval.BaselineFileName)
	}
	var changes []eval.Change
	var baselineErr error
	compared := false
	if evalOpts.updateBaseline {
		// Recording is not comparing: comparing a run against the baseline it
		// just wrote could only ever say "nothing moved".
		baselineErr = eval.NewBaseline(modelSet(), groups, overall).Save(baselinePath)
	} else {
		changes, compared, baselineErr = compareToBaseline(baselinePath, groups, overall)
	}

	if evalOpts.format == outputFormatJSON {
		if err := writeEvalJSON(out, dir, results, groups, overall, violations, changes, baselineErr); err != nil {
			return false, err
		}
	} else {
		writeEvalText(out, evalReport{
			dir: dir, baselinePath: baselinePath, compared: compared,
			results: results, groups: groups, overall: overall,
			violations: violations, changes: changes, baselineErr: baselineErr,
		})
	}
	return baselineErr == nil && len(violations) == 0 && len(eval.Regressions(changes)) == 0, nil
}

// compareToBaseline loads the baseline and compares. A missing baseline is not
// an error: the first run of a new corpus has nothing to compare against.
func compareToBaseline(path string, groups []eval.GroupSummary, overall eval.GroupSummary,
) ([]eval.Change, bool, error) {
	if _, err := os.Stat(path); errors.Is(err, fs.ErrNotExist) {
		return nil, false, nil
	}
	base, err := eval.LoadBaseline(path)
	if err != nil {
		return nil, false, err
	}
	changes, err := base.Compare(groups, overall, modelSet(), evalOpts.tolerance)
	return changes, err == nil, err
}

// resolveCorpora validates the options and finds the corpora to evaluate.
func resolveCorpora(dir string) ([]string, error) {
	if evalOpts.format != outputFormatText && evalOpts.format != outputFormatJSON {
		return nil, fmt.Errorf("invalid format %q (must be %s or %s)",
			evalOpts.format, outputFormatText, outputFormatJSON)
	}
	manifests, err := findCorpora(dir)
	if err != nil {
		return nil, err
	}
	if evalOpts.baseline != "" && len(manifests) > 1 {
		return nil, fmt.Errorf("--baseline names one file but %d corpora were found under %s", len(manifests), dir)
	}
	return manifests, nil
}

// findCorpora returns the manifest files under dir, or dir's own manifest if it
// is a corpus itself.
func findCorpora(dir string) ([]string, error) {
	info, err := os.Stat(dir)
	if err != nil {
		return nil, fmt.Errorf("corpus directory %s: %w", dir, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("corpus path %s is not a directory", dir)
	}
	var manifests []string
	err = filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && d.Name() == eval.ManifestFileName {
			manifests = append(manifests, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("search %s for corpora: %w", dir, err)
	}
	if len(manifests) == 0 {
		return nil, fmt.Errorf("no %s found under %s", eval.ManifestFileName, dir)
	}
	sort.Strings(manifests)
	return manifests, nil
}

// modelSet names the weights a run used, so a baseline cannot be compared
// against numbers produced by a different model.
func modelSet() string {
	if evalOpts.serverModels {
		return "server"
	}
	return "mobile"
}

// buildEvalRecognizer builds the same pipeline the accuracy test builds, so the
// command and the test cannot drift apart.
func buildEvalRecognizer() (eval.RecognizeFunc, func(), error) {
	cfg := GetConfig()
	b := pipeline.NewBuilder().WithModelsDir(models.GetModelsDir(cfg.ModelsDir))
	b.WithImageHeight(48)
	b.WithServerModels(evalOpts.serverModels)
	p, err := b.Build()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to build OCR pipeline: %w", err)
	}
	recognize := func(img image.Image) (eval.Reading, error) {
		res, err := p.ProcessImage(img)
		if err != nil {
			return eval.Reading{}, fmt.Errorf("process image: %w", err)
		}
		txt, err := pipeline.ToPlainTextImage(res)
		if err != nil {
			return eval.Reading{}, fmt.Errorf("render text: %w", err)
		}
		return eval.Reading{Text: txt, Regions: len(res.Regions), AvgConf: avgRecognitionConfidence(res)}, nil
	}
	closer := func() {
		if err := p.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "Error closing pipeline: %v\n", err)
		}
	}
	return recognize, closer, nil
}

func avgRecognitionConfidence(res *pipeline.OCRImageResult) float64 {
	var sum float64
	var n int
	for _, r := range res.Regions {
		if strings.TrimSpace(r.Text) != "" {
			sum += r.RecConfidence
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

// evalReport is what the text renderer needs about one corpus run.
type evalReport struct {
	dir          string
	baselinePath string
	compared     bool
	results      []eval.CaseResult
	groups       []eval.GroupSummary
	overall      eval.GroupSummary
	violations   []eval.Violation
	changes      []eval.Change
	baselineErr  error
}

func writeEvalText(out io.Writer, r evalReport) {
	_, _ = fmt.Fprintf(out, "Corpus: %s (%s models)\n", r.dir, modelSet())
	_, _ = fmt.Fprintln(out, strings.Repeat("=", 40))
	_, _ = fmt.Fprint(out, eval.FormatResults(r.results, r.groups, r.overall))

	_, _ = fmt.Fprintf(out, "\nBaseline: %s\n", r.baselinePath)
	writeBaselineSection(out, r)

	_, _ = fmt.Fprintln(out, "\nGate:")
	if len(r.violations) == 0 {
		_, _ = fmt.Fprintln(out, "  passed")
		return
	}
	for _, v := range r.violations {
		_, _ = fmt.Fprintf(out, "  FAIL %s\n", v)
	}
}

func writeBaselineSection(out io.Writer, r evalReport) {
	switch {
	case r.baselineErr != nil:
		_, _ = fmt.Fprintf(out, "  ERROR: %v\n", r.baselineErr)
	case evalOpts.updateBaseline:
		_, _ = fmt.Fprintln(out, "  recorded (nothing compared: a run cannot be measured against itself)")
	case !r.compared:
		_, _ = fmt.Fprintln(out, "  none recorded yet; re-run with --update-baseline to record this run")
	case len(r.changes) == 0:
		_, _ = fmt.Fprintln(out, "  no metric moved beyond the tolerance")
	default:
		for _, c := range r.changes {
			_, _ = fmt.Fprintf(out, "  %s\n", c)
		}
		if len(eval.Regressions(r.changes)) == 0 {
			_, _ = fmt.Fprintln(out, "  all movement is an improvement; re-run with --update-baseline to record it")
		}
	}
}

type evalJSON struct {
	Corpus      string              `json:"corpus"`
	Models      string              `json:"models"`
	Cases       []eval.CaseResult   `json:"cases"`
	Groups      []eval.GroupSummary `json:"groups"`
	Overall     eval.GroupSummary   `json:"overall"`
	Violations  []string            `json:"gate_violations"`
	Changes     []string            `json:"baseline_changes"`
	BaselineErr string              `json:"baseline_error,omitempty"`
	Passed      bool                `json:"passed"`
}

func writeEvalJSON(out io.Writer, dir string, results []eval.CaseResult,
	groups []eval.GroupSummary, overall eval.GroupSummary,
	violations []eval.Violation, changes []eval.Change, baselineErr error,
) error {
	doc := evalJSON{
		Corpus:  dir,
		Models:  modelSet(),
		Cases:   results,
		Groups:  groups,
		Overall: overall,
		Passed:  baselineErr == nil && len(violations) == 0 && len(eval.Regressions(changes)) == 0,
	}
	for _, v := range violations {
		doc.Violations = append(doc.Violations, v.String())
	}
	for _, c := range changes {
		doc.Changes = append(doc.Changes, c.String())
	}
	if baselineErr != nil {
		doc.BaselineErr = baselineErr.Error()
	}
	data, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return fmt.Errorf("encode eval result: %w", err)
	}
	_, err = fmt.Fprintln(out, string(data))
	return err
}
