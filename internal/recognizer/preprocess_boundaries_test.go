package recognizer

import (
	"image/color"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResizeForRecognition_MaxWidthAndPadMultiple(t *testing.T) {
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "W"
	cfg.Size = testutil.SmallSize
	cfg.Background = color.White
	cfg.Foreground = color.Black
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	targetH := 32
	maxW := 64
	padMult := 16
	resized, outW, outH, err := ResizeForRecognition(img, targetH, maxW, padMult)
	require.NoError(t, err)
	require.NotNil(t, resized)

	// Height fixed to targetH
	assert.Equal(t, targetH, outH)
	// Width is clamped to <= maxW and padded to multiple of padMult
	assert.LessOrEqual(t, outW, maxW)
	assert.Equal(t, 0, outW%padMult)
}

func TestNormalizeForRecognition_ValueRange(t *testing.T) {
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "Normalize"
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	// Use defaults from recognizer config
	resized, outW, outH, err := ResizeForRecognition(img, 32, 0, 8)
	require.NoError(t, err)
	ten, err := NormalizeForRecognition(resized)
	require.NoError(t, err)
	require.Equal(t, int64(1), ten.Shape[0])
	require.Equal(t, int64(3), ten.Shape[1])
	require.Equal(t, int64(outH), ten.Shape[2])
	require.Equal(t, int64(outW), ten.Shape[3])
	for _, v := range ten.Data {
		assert.GreaterOrEqual(t, v, float32(0))
		assert.LessOrEqual(t, v, float32(1))
	}
}
