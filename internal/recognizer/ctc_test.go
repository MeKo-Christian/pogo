package recognizer

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCTCCollapse(t *testing.T) {
	// indices with repeats and blanks(0): 1,1,0,2,2,2,3,0,3 -> collapse -> 1,2,3,3 -> remove blanks -> 1,2,3,3
	idx := []int{1, 1, 0, 2, 2, 2, 3, 0, 3}
	pr := []float64{.8, .7, .1, .9, .85, .8, .6, .1, .5}
	outIdx, outPr := CTCCollapse(idx, pr, 0)
	assert.Equal(t, []int{1, 2, 3, 3}, outIdx)
	assert.Equal(t, []float64{.8, .9, .6, .5}, outPr)
}

func TestDecodeCTCGreedy_TxC(t *testing.T) {
	// Single batch, T=4, C=4 (blank=0)
	// logits shaped [N,T,C] = [1,4,4]
	shape := []int64{1, 4, 4}
	// timestep 0: class 1 max, p=.9
	// timestep 1: class 1 max, p=.8 (repeat)
	// timestep 2: class 0 max (blank)
	// timestep 3: class 2 max, p=.7
	logits := []float32{
		0.1, 0.9, 0.0, 0.0,
		0.2, 0.8, 0.0, 0.0,
		0.9, 0.05, 0.03, 0.02,
		0.1, 0.2, 0.7, 0.0,
	}
	dec := DecodeCTCGreedy(logits, shape, 0, false)
	if assert.Len(t, dec, 1) {
		d := dec[0]
		assert.Equal(t, []int{1, 1, 0, 2}, d.Indices)
		assert.InDelta(t, 0.9, d.Probs[0], 1e-6)
		assert.InDelta(t, 0.8, d.Probs[1], 1e-6)
		assert.InDelta(t, 0.9, d.Probs[2], 1e-6) // blank prob
		assert.InDelta(t, 0.7, d.Probs[3], 1e-6)
		assert.Equal(t, []int{1, 2}, d.Collapsed)
		assert.InDelta(t, 0.9, d.CollapsedProb[0], 1e-6)
		assert.InDelta(t, 0.7, d.CollapsedProb[1], 1e-6)
		conf := SequenceConfidence(d.CollapsedProb)
		assert.InDelta(t, (0.9+0.7)/2, conf, 1e-6)
	}
}

func TestDecodeCTCGreedy_CxT(t *testing.T) {
	// Same data but [N,C,T] = [1,4,4]
	shape := []int64{1, 4, 4}
	logits := []float32{
		// class 0 over T
		0.1, 0.2, 0.9, 0.1,
		// class 1 over T
		0.9, 0.8, 0.05, 0.2,
		// class 2 over T
		0.0, 0.0, 0.03, 0.7,
		// class 3 over T
		0.0, 0.0, 0.02, 0.0,
	}
	dec := DecodeCTCGreedy(logits, shape, 0, true)
	if assert.Len(t, dec, 1) {
		d := dec[0]
		assert.Equal(t, []int{1, 1, 0, 2}, d.Indices)
		assert.Equal(t, []int{1, 2}, d.Collapsed)
	}
}
