package recognizer

import (
	"image/color"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNormalizeForRecognitionWithPool_BufferAndTensor(t *testing.T) {
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "Pool"
	cfg.Size = testutil.SmallSize
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)
	// Ensure a solid background; generating default already has white background.
	_ = color.White

	resized, outW, outH, err := ResizeForRecognition(img, 32, 0, 8)
	require.NoError(t, err)
	ten, buf, err := NormalizeForRecognitionWithPool(resized)
	require.NoError(t, err)
	require.NotNil(t, buf)
	// Tensor shape should match out dims
	require.Equal(t, int64(1), ten.Shape[0])
	require.Equal(t, int64(3), ten.Shape[1])
	require.Equal(t, int64(outH), ten.Shape[2])
	require.Equal(t, int64(outW), ten.Shape[3])
	// Data range in [0,1]
	for _, v := range ten.Data {
		assert.GreaterOrEqual(t, v, float32(0))
		assert.LessOrEqual(t, v, float32(1))
	}
}
