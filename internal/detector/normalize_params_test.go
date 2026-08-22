package detector

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDefaultNormalizeParams_ImageNet(t *testing.T) {
	p := DefaultNormalizeParams()
	assert.Equal(t, 3, p.Channels)
	assert.InDelta(t, 1.0/255.0, p.Scale, 1e-9)
	assert.Equal(t, [3]float32{0.485, 0.456, 0.406}, p.Mean)
	assert.Equal(t, [3]float32{0.229, 0.224, 0.225}, p.Std)
	assert.Equal(t, p, DefaultConfig().Normalize)
}

func TestConfig_NormalizeParams_FallsBackForBareLiteral(t *testing.T) {
	// A Config built as a bare literal carries a zero-valued NormalizeParams,
	// which would be unusable (std = 0); it must fall back to the defaults.
	assert.Equal(t, DefaultNormalizeParams(), Config{}.normalizeParams())

	partial := Config{}
	partial.Normalize.Channels = 3
	assert.Equal(t, DefaultNormalizeParams(), partial.normalizeParams())

	custom := DefaultConfig()
	custom.Normalize.Mean = [3]float32{0, 0, 0}
	custom.Normalize.Std = [3]float32{1, 1, 1}
	assert.Equal(t, custom.Normalize, custom.normalizeParams())
}
