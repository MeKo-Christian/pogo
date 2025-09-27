package detector

import (
	"testing"

	"github.com/MeKo-Tech/pogo/internal/onnx/mock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPostProcessDB_UniformMap(t *testing.T) {
	// All below threshold -> no regions
	uniLow := mock.NewUniformMap(16, 10, 0.2)
	regs := PostProcessDB(uniLow.Data, uniLow.Width, uniLow.Height, 0.3, 0.3)
	assert.Empty(t, regs)

	// All above threshold -> one region spanning full area with high confidence
	uniHigh := mock.NewUniformMap(12, 7, 0.95)
	regs2 := PostProcessDB(uniHigh.Data, uniHigh.Width, uniHigh.Height, 0.5, 0.7)
	require.Len(t, regs2, 1)
	r := regs2[0]
	assert.InDelta(t, 0.95, r.Confidence, 1e-3)
	assert.InDelta(t, 0.0, r.Box.MinX, 0.0001)
	assert.InDelta(t, 0.0, r.Box.MinY, 0.0001)
	assert.InDelta(t, float64(uniHigh.Width), r.Box.MaxX, 0.0001)
	assert.InDelta(t, float64(uniHigh.Height), r.Box.MaxY, 0.0001)
}

func TestPostProcessDB_CenteredBlob(t *testing.T) {
	// Blob in center -> one compact region
	m := mock.NewCenteredBlobMap(41, 31, 1.0, 5.0)
	regs := PostProcessDB(m.Data, m.Width, m.Height, 0.5, 0.5)
	require.NotEmpty(t, regs)
	// Expect at least one region and that it doesn't cover entire image
	got := regs[0]
	assert.Greater(t, got.Box.Width(), float64(1))
	assert.Greater(t, got.Box.Height(), float64(1))
	assert.Less(t, got.Box.Width(), float64(m.Width))
	assert.Less(t, got.Box.Height(), float64(m.Height))
}

func TestPostProcessDB_TextStripes(t *testing.T) {
	// Horizontal high stripes should yield multiple components
	m := mock.NewTextStripeMap(64, 32, 3, 1, 0.9, 0.05)
	regs := PostProcessDB(m.Data, m.Width, m.Height, 0.5, 0.6)
	// Expect roughly Height/period components
	period := 3 + 1
	// Due to bounding boxes merging adjacent pixels within rows, each bright stripe becomes one region
	expected := m.Height / period
	// Allow +/-1 tolerance depending on map size
	assert.GreaterOrEqual(t, len(regs), expected-1)
	assert.LessOrEqual(t, len(regs), expected+1)
}
