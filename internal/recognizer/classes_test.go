package recognizer

import (
	"image"
	_ "image/png" // register PNG decoder for the model-backed decode test
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/models"
	onnxmock "github.com/MeKo-Tech/pogo/internal/onnx/mock"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	onnxrt "github.com/yalue/onnxruntime_go"
)

// ppocrv5Classes is the number of output classes declared by the bundled
// PP-OCRv5 recognition models: 1 CTC blank + 18383 dictionary tokens + 1 space.
const ppocrv5Classes = 18385

// Fixture names shared by the validation tests below.
const (
	recOutputName = "fetch_name_0"
	stubDictPath  = "dict.txt"
)

// Task 1.4 -------------------------------------------------------------------

func TestOutputClasses_BundledModels(t *testing.T) {
	dictPath := models.GetDictionaryPath("", models.DictionaryPPOCRv5)
	if !fileExists(dictPath) {
		t.Skip("Dictionary not available, skipping test")
	}

	tests := []struct {
		name   string
		server bool
	}{
		{name: "mobile", server: false},
		{name: "server", server: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			modelPath := models.GetRecognitionModelPath("", tt.server)
			if !fileExists(modelPath) {
				t.Skip("Recognition model not available, skipping test")
			}

			cfg := DefaultConfig()
			cfg.ModelPath = modelPath
			cfg.DictPath = dictPath
			cfg.UseServerModel = tt.server

			r, err := NewRecognizer(cfg)
			require.NoError(t, err)
			defer func() { require.NoError(t, r.Close()) }()

			assert.Equal(t, ppocrv5Classes, r.OutputClasses())
		})
	}
}

func TestOutputClassCount(t *testing.T) {
	tests := []struct {
		name   string
		dims   []int64
		layout CTCLayout
		want   int
	}{
		{name: "classes last, static", dims: []int64{-1, -1, 18385}, layout: LayoutNTC, want: 18385},
		{name: "classes last, both static", dims: []int64{1, 40, 18385}, layout: LayoutNTC, want: 18385},
		{name: "classes first, static", dims: []int64{-1, 18385, -1}, layout: LayoutNCT, want: 18385},
		{name: "classes first, both static", dims: []int64{1, 18385, 40}, layout: LayoutNCT, want: 18385},
		// The layout decides which axis is read; it is not inferred from the sizes.
		{name: "classes first read as classes last", dims: []int64{1, 18385, 40}, layout: LayoutNTC, want: 40},
		{name: "trailing unit dims are stripped", dims: []int64{1, 40, 18385, 1}, layout: LayoutNTC, want: 18385},
		{name: "dynamic class dim", dims: []int64{-1, -1, -1}, layout: LayoutNTC, want: 0},
		{name: "zero class dim", dims: []int64{1, 1, 0}, layout: LayoutNTC, want: 0},
		{name: "empty", dims: nil, layout: LayoutNTC, want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, outputClassCount(tt.dims, tt.layout))
		})
	}
}

// A classes-first [N, C, T] output must not be rejected: the model declares
// LayoutNCT, and validation has to read the class count from the axis that
// declaration names.
func TestValidateCharsetAgainstModel_ClassesFirstLayout(t *testing.T) {
	cs := newCharset([]string{"a", "b", "c"}, CharsetOptions{})
	info := onnxrt.InputOutputInfo{Name: recOutputName, Dimensions: onnxrt.NewShape(-1, 4, 25)}
	require.NoError(t, validateCharsetAgainstModel(cs, info,
		Config{DictPath: stubDictPath, CTCLayout: LayoutNCT}))
}

// The same output under the default layout is a mismatch, and validateCTCLayout
// says so: the classes sit in the axis the declaration calls "time".
func TestValidateCTCLayout_ContradictedByModel(t *testing.T) {
	cs := newCharset([]string{"a", "b", "c"}, CharsetOptions{})
	info := onnxrt.InputOutputInfo{Name: recOutputName, Dimensions: onnxrt.NewShape(-1, 4, 25)}
	err := validateCTCLayout(cs, info, Config{DictPath: stubDictPath, CTCLayout: LayoutNTC})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "contradicts the model")
}

func TestValidateCharsetAgainstModel_MismatchHintsAtSpaceToken(t *testing.T) {
	cs := newCharset([]string{"a", "b", "c"}, CharsetOptions{AppendSpace: true})
	info := onnxrt.InputOutputInfo{Name: recOutputName, Dimensions: onnxrt.NewShape(-1, -1, 4)}
	err := validateCharsetAgainstModel(cs, info, Config{DictPath: stubDictPath, AppendSpaceToken: true})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "append_space_token=false")
}

// Task 1.6 -------------------------------------------------------------------

func TestLoadCharsetWithOptions_AppendSpace(t *testing.T) {
	dictPath := models.GetDictionaryPath("", models.DictionaryPPOCRv5)
	if !fileExists(dictPath) {
		t.Skip("Dictionary not available, skipping test")
	}

	plain, err := LoadCharset(dictPath)
	require.NoError(t, err)
	assert.False(t, plain.Contains(SpaceToken), "default loader must not append the space token")

	withSpace, err := LoadCharsetWithOptions(dictPath, CharsetOptions{AppendSpace: true})
	require.NoError(t, err)
	assert.Equal(t, plain.Size()+1, withSpace.Size())
	assert.True(t, withSpace.Contains(SpaceToken))
	// The space must be the last token, i.e. the highest class index.
	assert.Equal(t, plain.Size(), withSpace.LookupIndex(SpaceToken))
	// blank + tokens == model class count
	assert.Equal(t, ppocrv5Classes, withSpace.Size()+1)
}

func TestLoadCharsetsWithOptions_AppendSpaceOnce(t *testing.T) {
	dictPath := models.GetDictionaryPath("", models.DictionaryPPOCRv5)
	if !fileExists(dictPath) {
		t.Skip("Dictionary not available, skipping test")
	}

	// Passing the same file twice must not duplicate tokens, and the space
	// token must be appended exactly once.
	cs, err := LoadCharsetsWithOptions([]string{dictPath, dictPath}, CharsetOptions{AppendSpace: true})
	require.NoError(t, err)

	plain, err := LoadCharsets([]string{dictPath})
	require.NoError(t, err)

	assert.Equal(t, plain.Size()+1, cs.Size())

	spaces := 0
	for _, tok := range cs.Tokens {
		if tok == SpaceToken {
			spaces++
		}
	}
	assert.Equal(t, 1, spaces)
}

func TestApplyCharsetOptions_AlreadyContainsSpace(t *testing.T) {
	tokens := []string{"a", SpaceToken, "b"}
	got := applyCharsetOptions(tokens, CharsetOptions{AppendSpace: true})
	assert.Equal(t, tokens, got, "space must not be appended twice")
}

// Side benefit of task 1.6: Filter no longer strips spaces.
func TestCharsetFilter_KeepsSpaceWhenAppended(t *testing.T) {
	tokens := []string{"H", "i"}

	without := newCharset(tokens, CharsetOptions{})
	assert.Equal(t, "Hi", without.Filter("H i"), "without the space token, Filter strips spaces")

	with := newCharset(tokens, CharsetOptions{AppendSpace: true})
	assert.Equal(t, "H i", with.Filter("H i"), "with the space token, Filter must keep spaces")
}

// Task 1.5 -------------------------------------------------------------------

func TestNewRecognizer_DictionaryModelMismatch(t *testing.T) {
	modelPath := models.GetRecognitionModelPath("", false)
	dictPath := models.GetDictionaryPath("", models.DictionaryPPOCRKeysV1)
	if !fileExists(modelPath) {
		t.Skip("Recognition model not available, skipping test")
	}
	if !fileExists(dictPath) {
		t.Skip("Dictionary not available, skipping test")
	}

	cfg := DefaultConfig()
	cfg.ModelPath = modelPath
	cfg.DictPath = dictPath

	r, err := NewRecognizer(cfg)
	require.Error(t, err)
	require.Nil(t, r)
	assert.Contains(t, err.Error(), "18385")
	assert.Contains(t, err.Error(), "143")
	assert.Contains(t, err.Error(), dictPath)
}

func TestNewRecognizer_DictionaryModelMatch(t *testing.T) {
	modelPath := models.GetRecognitionModelPath("", false)
	dictPath := models.GetDictionaryPath("", models.DictionaryPPOCRv5)
	if !fileExists(modelPath) || !fileExists(dictPath) {
		t.Skip("Recognition model or dictionary not available, skipping test")
	}

	cfg := DefaultConfig()
	cfg.ModelPath = modelPath
	cfg.DictPath = dictPath

	r, err := NewRecognizer(cfg)
	require.NoError(t, err)
	defer func() { require.NoError(t, r.Close()) }()

	assert.Equal(t, r.OutputClasses(), r.GetCharset().Size()+1)
}

func TestValidateCharsetAgainstModel_DynamicClassDimIsSkipped(t *testing.T) {
	cs := newCharset([]string{"a", "b"}, CharsetOptions{})
	info := onnxrt.InputOutputInfo{Name: recOutputName, Dimensions: onnxrt.NewShape(-1, -1, -1)}
	require.NoError(t, validateCharsetAgainstModel(cs, info, Config{DictPath: stubDictPath}))
}

// Task 1.7 -------------------------------------------------------------------

// TestConvertIndicesToRunes_DecodesSpaceClass is a pure unit test: it builds
// synthetic logits whose greedy argmax path visits the space class and asserts
// the decoded text contains a space. It is red when AppendSpace is off.
func TestConvertIndicesToRunes_DecodesSpaceClass(t *testing.T) {
	tokens := []string{"H", "i"}

	withSpace := newCharset(tokens, CharsetOptions{AppendSpace: true})
	classes := withSpace.Size() + 1 // blank + H + i + space
	require.Equal(t, 4, classes)

	// Greedy path: blank, H, blank, space, space, blank, i
	// Class indices: 0 = blank, 1 = "H", 2 = "i", 3 = " ".
	path := []int{0, 1, 0, 3, 3, 0, 2}
	logits := onnxmock.NewGreedyPathLogits(path, classes, false, 1.0, 0.0)

	decoded := DecodeCTCGreedy(logits.Data, logits.Shape, 0, false)
	require.Len(t, decoded, 1)
	require.Equal(t, []int{1, 3, 2}, decoded[0].Collapsed)

	got := convertIndicesToRunes(decoded[0].Collapsed, 0, withSpace, nil)
	assert.Equal(t, "H i", got)
	assert.Contains(t, got, " ")

	// Control: without the appended space token the same path loses the space.
	withoutSpace := newCharset(tokens, CharsetOptions{})
	assert.Equal(t, "Hi", convertIndicesToRunes(decoded[0].Collapsed, 0, withoutSpace, nil))
}

// TestRecognizeRegion_DecodesSpace_Integration is model-backed: rotated_0.png
// contains "Rotated Text" and must decode with the space intact.
func TestRecognizeRegion_DecodesSpace_Integration(t *testing.T) {
	modelPath := models.GetRecognitionModelPath("", false)
	dictPath := models.GetDictionaryPath("", models.DictionaryPPOCRv5)
	if !fileExists(modelPath) || !fileExists(dictPath) {
		t.Skip("Recognition model or dictionary not available, skipping test")
	}

	imgPath := filepath.Join(testutil.GetTestDataDir(t), "images", "rotated", "rotated_0.png")
	if !fileExists(imgPath) {
		t.Skip("Test image not available, skipping test")
	}

	f, err := os.Open(imgPath) //nolint:gosec // G304: test fixture path
	require.NoError(t, err)
	defer func() { require.NoError(t, f.Close()) }()

	img, _, err := image.Decode(f)
	require.NoError(t, err)

	cfg := DefaultConfig()
	cfg.ModelPath = modelPath
	cfg.DictPath = dictPath

	r, err := NewRecognizer(cfg)
	require.NoError(t, err)
	defer func() { require.NoError(t, r.Close()) }()

	// The fixture is a mostly empty 640x480 canvas with one small text line on it.
	// Feeding the whole canvas to the recognizer squashes the line into a few
	// pixels, so crop to the inked area with a small margin, as a detector would.
	x0, y0, x1, y1 := inkBounds(t, img, 8)
	region := detector.DetectedRegion{
		Polygon: []utils.Point{
			{X: x0, Y: y0},
			{X: x1, Y: y0},
			{X: x1, Y: y1},
			{X: x0, Y: y1},
		},
		Box:        utils.NewBox(x0, y0, x1, y1),
		Confidence: 1.0,
	}

	res, err := r.RecognizeRegion(img, region)
	require.NoError(t, err)
	require.NotNil(t, res)

	assert.Equal(t, "Rotated Text", strings.TrimSpace(res.Text))
}

// inkBounds returns the bounding box of the dark (inked) pixels of img, grown by
// margin pixels and clamped to the image bounds.
func inkBounds(t *testing.T, img image.Image, margin float64) (float64, float64, float64, float64) {
	t.Helper()

	b := img.Bounds()
	minX, minY := b.Max.X, b.Max.Y
	maxX, maxY := b.Min.X, b.Min.Y
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bl, _ := img.At(x, y).RGBA()
			// Rec. 601 luma on 16-bit channels; below 50% counts as ink.
			luma := (299*int(r) + 587*int(g) + 114*int(bl)) / 1000
			if luma >= 0x8000 {
				continue
			}
			minX = min(minX, x)
			minY = min(minY, y)
			maxX = max(maxX, x)
			maxY = max(maxY, y)
		}
	}
	require.LessOrEqual(t, minX, maxX, "image contains no dark pixels")

	x0 := math.Max(float64(b.Min.X), float64(minX)-margin)
	y0 := math.Max(float64(b.Min.Y), float64(minY)-margin)
	x1 := math.Min(float64(b.Max.X), float64(maxX+1)+margin)
	y1 := math.Min(float64(b.Max.Y), float64(maxY+1)+margin)
	return x0, y0, x1, y1
}
