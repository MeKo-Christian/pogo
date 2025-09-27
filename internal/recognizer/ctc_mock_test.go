package recognizer

import (
	"testing"

	onnxmock "github.com/MeKo-Tech/pogo/internal/onnx/mock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDecodeCTCGreedy_WithMockLogits_TxC(t *testing.T) {
	// Greedy path indices include repeats and blanks(0)
	idx := []int{0, 2, 2, 0, 3, 3, 4}
	logits := onnxmock.NewGreedyPathLogits(idx, 6, false, 1.0, 0.0)
	dec := DecodeCTCGreedy(logits.Data, logits.Shape, 0, false)
	require.Len(t, dec, 1)
	d := dec[0]
	assert.Equal(t, idx, d.Indices)
	// Collapsed must remove repeats and blanks
	assert.Equal(t, []int{2, 3, 4}, d.Collapsed)
}

func TestDecodeCTCGreedy_WithMockLogits_CxT(t *testing.T) {
	// Same indices but classes-first layout
	idx := []int{1, 1, 0, 5}
	logits := onnxmock.NewGreedyPathLogits(idx, 8, true, 0.9, 0.01)
	dec := DecodeCTCGreedy(logits.Data, logits.Shape, 0, true)
	require.Len(t, dec, 1)
	d := dec[0]
	assert.Equal(t, idx, d.Indices)
	assert.Equal(t, []int{1, 5}, d.Collapsed)
}
