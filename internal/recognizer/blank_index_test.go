package recognizer

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	onnxrt "github.com/yalue/onnxruntime_go"
)

// newTestCharset builds a charset with the lookup maps LookupToken needs.
func newTestCharset(tokens ...string) *Charset {
	c := &Charset{
		Tokens:       tokens,
		IndexToToken: make(map[int]string, len(tokens)),
		TokenToIndex: make(map[string]int, len(tokens)),
	}
	for i, tok := range tokens {
		c.IndexToToken[i] = tok
		c.TokenToIndex[tok] = i
	}
	return c
}

// TestCharsetTokenIndex_ShiftsAroundTheBlank pins down the mapping the review
// asked for: classes before the blank keep their index, the blank itself has no
// token, and classes after it are shifted down by one.
func TestCharsetTokenIndex_ShiftsAroundTheBlank(t *testing.T) {
	tests := []struct {
		name  string
		class int
		blank int
		want  int
	}{
		{"blank at 0 drops class 0", 0, 0, -1},
		{"blank at 0 shifts class 1", 1, 0, 0},
		{"blank at 0 shifts class 5", 5, 0, 4},
		{"nonzero blank keeps classes below it", 0, 2, 0},
		{"nonzero blank keeps class right below it", 1, 2, 1},
		{"nonzero blank drops itself", 2, 2, -1},
		{"nonzero blank shifts classes above it", 3, 2, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, charsetTokenIndex(tt.class, tt.blank))
		})
	}
}

// TestConvertIndicesToRunes_NonzeroBlank decodes class sequences with a blank at
// 0 and at 2 and expects both to yield the tokens the layout implies.
func TestConvertIndicesToRunes_NonzeroBlank(t *testing.T) {
	// Charset holds four tokens; with a blank there are five classes.
	charset := newTestCharset("a", "b", "c", "d")

	// blank=0: classes 1..4 map to tokens 0..3.
	assert.Equal(t, "abcd", convertIndicesToRunes([]int{1, 2, 3, 4}, 0, charset, nil))
	// blank=0: the blank itself contributes nothing.
	assert.Equal(t, "ab", convertIndicesToRunes([]int{1, 0, 2}, 0, charset, nil))

	// blank=2: classes 0 and 1 map to tokens 0 and 1, classes 3 and 4 to 2 and 3.
	assert.Equal(t, "abcd", convertIndicesToRunes([]int{0, 1, 3, 4}, 2, charset, nil))
	// blank=2: class 2 is the blank and contributes nothing.
	assert.Equal(t, "ac", convertIndicesToRunes([]int{0, 2, 3}, 2, charset, nil))
}

func TestValidateBlankIndexAgainstModel(t *testing.T) {
	tests := []struct {
		name      string
		dims      []int64
		layout    CTCLayout
		blank     int
		wantError bool
	}{
		{"blank inside the class range", []int64{1, 20, 5}, LayoutNTC, 4, false},
		{"blank equal to the class count", []int64{1, 20, 5}, LayoutNTC, 5, true},
		{"blank beyond the class count", []int64{1, 20, 5}, LayoutNTC, 99, true},
		{"nct reads the right dimension", []int64{1, 5, 20}, LayoutNCT, 7, true},
		{"dynamic class dim defers the check", []int64{-1, -1, -1}, LayoutNTC, 99, false},
		{"rank too low defers the check", []int64{1, 5}, LayoutNTC, 99, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := onnxrt.InputOutputInfo{Dimensions: tt.dims}
			err := validateBlankIndexAgainstModel(info, Config{CTCLayout: tt.layout, BlankIndex: tt.blank})
			if !tt.wantError {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), "blank index")
		})
	}
}

// TestOutputClassCount_RankBelowThreeIsUnknown guards the fix for the review
// finding that a rank-2 output must not report its last dimension as the class
// count, because the declared layout names no class axis there.
func TestOutputClassCount_RankBelowThreeIsUnknown(t *testing.T) {
	assert.Equal(t, 0, outputClassCount([]int64{20, 5}, LayoutNTC))
	assert.Equal(t, 0, outputClassCount([]int64{20, 5}, LayoutNCT))
	assert.Equal(t, 0, outputClassCount([]int64{5}, LayoutNTC))
}
