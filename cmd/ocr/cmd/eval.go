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

	recognize, identity, closePipeline, err := buildEvalRecognizer()
	if err != nil {
		return err
	}
	defer closePipeline()

	reports := make([]evalReport, 0, len(manifests))
	for _, m := range manifests {
		r, err := evalCorpus(m, recognize, identity)
		if err != nil {
			return err
		}
		reports = append(reports, r)
	}

	if err := writeEvalReports(cmd.OutOrStdout(), reports); err != nil {
		return err
	}
	if reasons := failureReasons(reports); len(reasons) > 0 {
		return fmt.Errorf("evaluation failed: %s", strings.Join(reasons, ", "))
	}
	return nil
}

// failureReasons names why a run failed, in the operator's terms. A gate
// violation, a regression against the baseline and an unusable baseline are
// three different problems and lead to three different next steps.
func failureReasons(reports []evalReport) []string {
	var gates, regressions, baselines int
	for _, r := range reports {
		gates += len(r.violations)
		regressions += len(eval.Regressions(r.changes))
		if r.baselineErr != nil {
			baselines++
		}
	}
	var reasons []string
	if gates > 0 {
		reasons = append(reasons, fmt.Sprintf("%d gate violation(s)", gates))
	}
	if regressions > 0 {
		reasons = append(reasons, fmt.Sprintf("%d regression(s) against the baseline", regressions))
	}
	if baselines > 0 {
		reasons = append(reasons, fmt.Sprintf("%d baseline(s) could not be compared", baselines))
	}
	return reasons
}

// evalCorpus measures one corpus. Rendering happens afterwards, once, so that a
// tree of corpora produces a single document rather than one per corpus.
func evalCorpus(manifest string, recognize eval.RecognizeFunc, identity eval.ModelIdentity) (evalReport, error) {
	dir := filepath.Dir(manifest)
	cases, err := eval.LoadManifest(manifest)
	if err != nil {
		return evalReport{}, err
	}
	results, err := eval.Run(dir, cases, recognize)
	if err != nil {
		return evalReport{}, err
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
		baselineErr = eval.NewBaseline(identity, groups, overall).Save(baselinePath)
	} else {
		changes, compared, baselineErr = compareToBaseline(baselinePath, groups, overall, identity)
	}

	return evalReport{
		dir: dir, models: identity, baselinePath: baselinePath, compared: compared,
		results: results, groups: groups, overall: overall,
		violations: violations, changes: changes, baselineErr: baselineErr,
	}, nil
}

// compareToBaseline loads the baseline and compares. A missing baseline is not
// an error: the first run of a new corpus has nothing to compare against.
func compareToBaseline(path string, groups []eval.GroupSummary, overall eval.GroupSummary,
	identity eval.ModelIdentity,
) ([]eval.Change, bool, error) {
	if _, err := os.Stat(path); errors.Is(err, fs.ErrNotExist) {
		return nil, false, nil
	}
	base, err := eval.LoadBaseline(path)
	if err != nil {
		return nil, false, err
	}
	changes, err := base.Compare(groups, overall, identity, evalOpts.tolerance)
	return changes, err == nil, err
}

// resolveCorpora validates the options and finds the corpora to evaluate.
func resolveCorpora(dir string) ([]string, error) {
	if evalOpts.format != outputFormatText && evalOpts.format != outputFormatJSON {
		return nil, fmt.Errorf("invalid format %q (must be %s or %s)",
			evalOpts.format, outputFormatText, outputFormatJSON)
	}
	// A negative tolerance would report every metric as moved, inverting what
	// the flag is for.
	if evalOpts.tolerance < 0 {
		return nil, fmt.Errorf("invalid tolerance %.4f (must not be negative)", evalOpts.tolerance)
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

// modelVariant names the bundled variant a run asked for. It is a label, not an
// identity: eval.Fingerprint supplies the identity a baseline is keyed on.
func modelVariant() string {
	if evalOpts.serverModels {
		return "server"
	}
	return "mobile"
}

// modelIdentity fingerprints the artifacts the pipeline actually resolved, so a
// baseline recorded with one --models-dir cannot be silently compared against a
// run that used another.
func modelIdentity(cfg pipeline.Config) (eval.ModelIdentity, error) {
	paths := []string{cfg.Detector.ModelPath, cfg.Recognizer.ModelPath}
	if cfg.Recognizer.DictPath != "" {
		paths = append(paths, cfg.Recognizer.DictPath)
	}
	paths = append(paths, cfg.Recognizer.DictPaths...)
	return eval.Fingerprint(modelVariant(), paths)
}

// buildEvalRecognizer builds the same pipeline the accuracy test builds, so the
// command and the test cannot drift apart.
func buildEvalRecognizer() (eval.RecognizeFunc, eval.ModelIdentity, func(), error) {
	cfg := GetConfig()
	b := pipeline.NewBuilder().WithModelsDir(models.GetModelsDir(cfg.ModelsDir))
	b.WithImageHeight(48)
	b.WithServerModels(evalOpts.serverModels)
	identity, err := modelIdentity(b.Config())
	if err != nil {
		return nil, eval.ModelIdentity{}, nil, err
	}
	p, err := b.Build()
	if err != nil {
		return nil, eval.ModelIdentity{}, nil, fmt.Errorf("failed to build OCR pipeline: %w", err)
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
	return recognize, identity, closer, nil
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

// evalReport is one corpus's measurement, kept until every corpus has been
// measured so that the whole tree renders as one document.
type evalReport struct {
	dir          string
	models       eval.ModelIdentity
	baselinePath string
	compared     bool
	results      []eval.CaseResult
	groups       []eval.GroupSummary
	overall      eval.GroupSummary
	violations   []eval.Violation
	changes      []eval.Change
	baselineErr  error
}

func (r evalReport) passed() bool {
	return r.baselineErr == nil && len(r.violations) == 0 && len(eval.Regressions(r.changes)) == 0
}

// writeEvalReports renders every corpus of the run. Under --format json this is
// a single JSON document covering the whole tree, not one object per corpus:
// a directory of corpora is explicitly supported, and a `{...}{...}` stream
// would not be parseable.
func writeEvalReports(out io.Writer, reports []evalReport) error {
	if evalOpts.format == outputFormatJSON {
		return writeEvalJSON(out, reports)
	}
	for i, r := range reports {
		if i > 0 {
			_, _ = fmt.Fprintln(out)
		}
		writeEvalText(out, r)
	}
	return nil
}

func writeEvalText(out io.Writer, r evalReport) {
	_, _ = fmt.Fprintf(out, "Corpus: %s (%s models)\n", r.dir, r.models)
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

// evalRunJSON is the whole run: one document, however many corpora it covers.
type evalRunJSON struct {
	Corpora []evalCorpusJSON `json:"corpora"`
	Passed  bool             `json:"passed"`
}

type evalCorpusJSON struct {
	Corpus      string              `json:"corpus"`
	Models      eval.ModelIdentity  `json:"models"`
	Baseline    string              `json:"baseline"`
	Cases       []eval.CaseResult   `json:"cases"`
	Groups      []eval.GroupSummary `json:"groups"`
	Overall     eval.GroupSummary   `json:"overall"`
	Violations  []string            `json:"gate_violations"`
	Changes     []string            `json:"baseline_changes"`
	BaselineErr string              `json:"baseline_error,omitempty"`
	Passed      bool                `json:"passed"`
}

func writeEvalJSON(out io.Writer, reports []evalReport) error {
	doc := evalRunJSON{Corpora: make([]evalCorpusJSON, 0, len(reports)), Passed: true}
	for _, r := range reports {
		c := evalCorpusJSON{
			Corpus:   r.dir,
			Models:   r.models,
			Baseline: r.baselinePath,
			Cases:    r.results,
			Groups:   r.groups,
			Overall:  r.overall,
			Passed:   r.passed(),
		}
		for _, v := range r.violations {
			c.Violations = append(c.Violations, v.String())
		}
		for _, ch := range r.changes {
			c.Changes = append(c.Changes, ch.String())
		}
		if r.baselineErr != nil {
			c.BaselineErr = r.baselineErr.Error()
		}
		doc.Passed = doc.Passed && c.Passed
		doc.Corpora = append(doc.Corpora, c)
	}
	data, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return fmt.Errorf("encode eval result: %w", err)
	}
	_, err = fmt.Fprintln(out, string(data))
	return err
}
