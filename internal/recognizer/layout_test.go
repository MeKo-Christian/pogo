package recognizer

import (
	"os"
	"strings"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	onnxrt "github.com/yalue/onnxruntime_go"
)

func TestCTCLayout_ClassesFirst(t *testing.T) {
	assert.False(t, LayoutNTC.classesFirst())
	assert.True(t, LayoutNCT.classesFirst())
	assert.False(t, CTCLayout("").classesFirst())
}

func TestCTCLayout_Validate(t *testing.T) {
	require.NoError(t, LayoutNTC.validate())
	require.NoError(t, LayoutNCT.validate())
	require.NoError(t, CTCLayout("").validate())

	err := CTCLayout("tnc").validate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "tnc")
}

func TestConfig_CTCLayoutDefaultsToNTC(t *testing.T) {
	assert.Equal(t, LayoutNTC, Config{}.ctcLayout())
	assert.Equal(t, LayoutNTC, DefaultConfig().ctcLayout())
	assert.Equal(t, LayoutNCT, Config{CTCLayout: LayoutNCT}.ctcLayout())
}

func TestNormalizeOutputDims(t *testing.T) {
	assert.Equal(t, []int64{1, 10, 100}, normalizeOutputDims([]int64{1, 10, 100}))
	assert.Equal(t, []int64{1, 10, 100}, normalizeOutputDims([]int64{1, 10, 100, 1}))
	assert.Equal(t, []int64{1, 10, 100}, normalizeOutputDims([]int64{1, 10, 100, 1, 1}))
	// A trailing non-unit dimension is left alone.
	assert.Equal(t, []int64{1, 10, 100, 2}, normalizeOutputDims([]int64{1, 10, 100, 2}))
	// Never strips below rank 3.
	assert.Equal(t, []int64{1, 1, 1}, normalizeOutputDims([]int64{1, 1, 1}))
}

// TestOutputClassCount_ReadsTheLayoutsDimension proves the class count is read
// from whichever axis the declared layout says carries the classes, rather than
// always from the last one.
func TestOutputClassCount_ReadsTheLayoutsDimension(t *testing.T) {
	tests := []struct {
		name   string
		dims   []int64
		layout CTCLayout
		want   int
	}{
		{"ntc static classes", []int64{-1, -1, 18385}, LayoutNTC, 18385},
		{"nct static classes", []int64{-1, 18385, -1}, LayoutNCT, 18385},
		{"ntc but classes dim dynamic", []int64{-1, 18385, -1}, LayoutNTC, 0},
		{"nct but classes dim dynamic", []int64{-1, -1, 18385}, LayoutNCT, 0},
		{"trailing unit dim ignored", []int64{1, 10, 100, 1}, LayoutNTC, 100},
		{"empty", nil, LayoutNTC, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, outputClassCount(tt.dims, tt.layout))
		})
	}
}

func TestValidateCTCLayout(t *testing.T) {
	// 4 tokens + 1 blank = 5 classes.
	charset := &Charset{Tokens: []string{"a", "b", "c", "d"}}

	tests := []struct {
		name      string
		dims      []int64
		layout    CTCLayout
		wantError bool
	}{
		{"ntc agrees with the model", []int64{1, 20, 5}, LayoutNTC, false},
		{"nct agrees with the model", []int64{1, 5, 20}, LayoutNCT, false},
		{"ntc declared but classes are in dim 1", []int64{1, 5, 20}, LayoutNTC, true},
		{"nct declared but classes are in dim 2", []int64{1, 20, 5}, LayoutNCT, true},
		{"both dynamic, declaration stands", []int64{-1, -1, -1}, LayoutNTC, false},
		{"class dim dynamic, time dim innocent", []int64{-1, 20, -1}, LayoutNTC, false},
		{"rank too low to check", []int64{1, 5}, LayoutNTC, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := onnxrt.InputOutputInfo{Dimensions: tt.dims}
			err := validateCTCLayout(charset, info, Config{CTCLayout: tt.layout})
			if !tt.wantError {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			// The message must name the layout and both dimension indices.
			assert.Contains(t, err.Error(), string(tt.layout))
			assert.Contains(t, err.Error(), "5")
		})
	}
}

func TestValidateCTCLayout_NilCharset(t *testing.T) {
	info := onnxrt.InputOutputInfo{Dimensions: []int64{1, 20, 5}}
	require.NoError(t, validateCTCLayout(nil, info, Config{CTCLayout: LayoutNTC}))
}

func TestNewRecognizer_RejectsUnknownLayout(t *testing.T) {
	cfg := DefaultConfig()
	cfg.CTCLayout = "tnc"
	_, err := NewRecognizer(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "tnc")
}

func TestNewRecognizer_RejectsNegativeBlankIndex(t *testing.T) {
	cfg := DefaultConfig()
	cfg.BlankIndex = -1
	_, err := NewRecognizer(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "blank index")
}

// TestNewRecognizer_RejectsContradictoryLayout declares NCT against a bundled
// model whose class dimension is demonstrably last, and expects a load-time
// error rather than a silent fallback that happens to decode.
func TestNewRecognizer_RejectsContradictoryLayout(t *testing.T) {
	cfg := DefaultConfig()
	if _, err := os.Stat(cfg.ModelPath); err != nil {
		t.Skipf("recognition model missing: %s", cfg.ModelPath)
	}
	if _, err := os.Stat(cfg.DictPath); err != nil {
		t.Skipf("dictionary missing: %s", cfg.DictPath)
	}
	cfg.CTCLayout = LayoutNCT

	_, err := NewRecognizer(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), string(LayoutNCT))
	assert.Contains(t, err.Error(), "18385")
}

// TestNewRecognizer_AcceptsDeclaredNTC is the positive counterpart.
func TestNewRecognizer_AcceptsDeclaredNTC(t *testing.T) {
	cfg := DefaultConfig()
	for _, p := range []string{cfg.ModelPath, cfg.DictPath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("required file missing: %s", p)
		}
	}
	cfg.CTCLayout = LayoutNTC

	r, err := NewRecognizer(cfg)
	if err != nil && strings.Contains(err.Error(), "onnxruntime") {
		t.Skipf("ONNX runtime unavailable: %v", err)
	}
	require.NoError(t, err)
	defer func() { _ = r.Close() }()
	assert.Equal(t, 18385, r.OutputClasses())
}

// TestDecodeUnderBothLayouts feeds the same logical sequence to the CTC decoder
// in both memory layouts and requires the same decoded indices out. This is what
// the deleted determineClassesFirst heuristic was guessing at.
func TestDecodeUnderBothLayouts(t *testing.T) {
	const (
		timesteps = 4
		classes   = 5
	)
	// Argmax path over time: blank, 1, 1, 3 -> collapses to [1, 3].
	argmax := []int{0, 1, 1, 3}

	ntc := make([]float32, timesteps*classes)
	nct := make([]float32, classes*timesteps)
	for tIdx, cls := range argmax {
		for c := range classes {
			v := float32(0.1)
			if c == cls {
				v = 0.9
			}
			ntc[tIdx*classes+c] = v
			nct[c*timesteps+tIdx] = v
		}
	}

	gotNTC := DecodeCTCGreedy(ntc, []int64{1, timesteps, classes}, 0, LayoutNTC.classesFirst())
	gotNCT := DecodeCTCGreedy(nct, []int64{1, classes, timesteps}, 0, LayoutNCT.classesFirst())

	require.Len(t, gotNTC, 1)
	require.Len(t, gotNCT, 1)
	// DecodeCTCGreedy returns the raw per-timestep argmax path; collapsing is a
	// separate step.
	assert.Equal(t, argmax, gotNTC[0].Indices)
	assert.Equal(t, gotNTC[0].Indices, gotNCT[0].Indices,
		"the same sequence must decode identically under both layouts")

	collapsed, _ := CTCCollapse(gotNTC[0].Indices, gotNTC[0].Probs, 0)
	assert.Equal(t, []int{1, 3}, collapsed, "blank and repeats collapse away")
}

// TestBundledModelIsNTC records what the bundled models actually declare, so a
// model swap that changes the layout fails here instead of silently decoding
// nonsense.
func TestBundledModelIsNTC(t *testing.T) {
	for _, tc := range []struct {
		name   string
		server bool
	}{{"mobile", false}, {"server", true}} {
		t.Run(tc.name, func(t *testing.T) {
			path := models.GetRecognitionModelPath("", tc.server)
			if _, err := os.Stat(path); err != nil {
				t.Skipf("model missing: %s", path)
			}
			_, outputs, err := onnxrt.GetInputOutputInfo(path)
			if err != nil {
				t.Skipf("cannot read model info: %v", err)
			}
			require.NotEmpty(t, outputs)
			dims := normalizeOutputDims(outputs[0].Dimensions)
			require.Len(t, dims, 3)
			assert.Equal(t, int64(18385), dims[2], "classes belong in the last dimension (NTC)")
		})
	}
}
