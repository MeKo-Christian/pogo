package pipeline

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProcessPDF_ErrorPropagation(t *testing.T) {
	p := &Pipeline{}
	// Not initialized -> should fail early
	_, err := p.ProcessPDF("/no/such/file.pdf", "1-2")
	require.Error(t, err)

	// Build a pipeline-like struct with nil components is enough for early check.
	// Once properly initialized in integration, error will originate from extract step.
}

func TestPipeline_Info_MapShape(t *testing.T) {
	p := &Pipeline{cfg: DefaultConfig()}
	info := p.Info()
	// Keys present even without initialized models
	assert.Contains(t, info, "models_dir")
	assert.Contains(t, info, "enable_orientation")
	assert.Contains(t, info, "orientation")
	assert.Contains(t, info, "textline_orientation")
	// No detector/recognizer maps when nil
	_, hasDet := info["detector"]
	_, hasRec := info["recognizer"]
	assert.False(t, hasDet)
	assert.False(t, hasRec)
}
