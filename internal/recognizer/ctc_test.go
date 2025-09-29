package recognizer

import (
	"math"
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

func TestDecodeCTCBeamSearch_Basic(t *testing.T) {
	// Simple test case: single timestep with clear winner
	shape := []int64{1, 1, 4}               // [N=1, T=1, C=4]
	logits := []float32{0.1, 0.9, 0.0, 0.0} // class 1 has highest prob

	dec := DecodeCTCBeamSearch(logits, shape, 0, 5, false)
	if assert.Len(t, dec, 1) {
		d := dec[0]
		assert.Equal(t, []int{1}, d.Sequence) // Should collapse to just class 1
		assert.True(t, d.Probability > -1.0)  // Should have reasonable log prob
		assert.InDelta(t, 0.9, d.CharProbs[0], 1e-6)
	}
}

func TestDecodeCTCBeamSearch_MultipleTimesteps(t *testing.T) {
	// Test with multiple timesteps and beam search
	shape := []int64{1, 3, 4} // [N=1, T=3, C=4]
	// Timestep 0: class 1 (0.8)
	// Timestep 1: class 1 (0.7) - should merge
	// Timestep 2: class 2 (0.6)
	logits := []float32{
		0.1, 0.8, 0.05, 0.05, // t=0
		0.2, 0.7, 0.05, 0.05, // t=1
		0.1, 0.1, 0.6, 0.2, // t=2
	}

	dec := DecodeCTCBeamSearch(logits, shape, 0, 3, false)
	if assert.Len(t, dec, 1) {
		d := dec[0]
		assert.Equal(t, []int{1, 2}, d.Sequence) // Should be 1,2 (merged consecutive 1s)
		assert.Len(t, d.CharProbs, 2)
		assert.True(t, d.CharProbs[0] > 0.7)         // First char prob
		assert.InDelta(t, 0.6, d.CharProbs[1], 1e-6) // Second char prob
	}
}

func TestDecodeCTCBeamSearch_WithBlanks(t *testing.T) {
	// Test handling of blank characters
	shape := []int64{1, 4, 4}
	// Sequence: 1, 1, blank(0), 2
	logits := []float32{
		0.1, 0.8, 0.05, 0.05, // t=0: class 1
		0.2, 0.7, 0.05, 0.05, // t=1: class 1 (repeat)
		0.9, 0.05, 0.03, 0.02, // t=2: blank
		0.1, 0.2, 0.6, 0.1, // t=3: class 2
	}

	dec := DecodeCTCBeamSearch(logits, shape, 0, 5, false)
	if assert.Len(t, dec, 1) {
		d := dec[0]
		assert.Equal(t, []int{1, 2}, d.Sequence) // Should collapse repeats and remove blank
		assert.Len(t, d.CharProbs, 2)
	}
}

func TestDecodeCTCBeamSearch_BeamWidth(t *testing.T) {
	// Test that beam width affects results appropriately
	shape := []int64{1, 2, 4}
	logits := []float32{
		0.4, 0.3, 0.2, 0.1, // t=0: class 0 (blank) has highest prob
		0.1, 0.2, 0.3, 0.4, // t=1: class 3 has highest prob
	}

	// With beam width 1, should be greedy-like
	dec1 := DecodeCTCBeamSearch(logits, shape, 0, 1, false)
	if assert.Len(t, dec1, 1) {
		d1 := dec1[0]
		// Since class 0 at t=0 is blank, sequence doesn't extend
		// At t=1, class 3 extends to [3]
		assert.Equal(t, []int{3}, d1.Sequence)
	}

	// With larger beam width, might find better path
	dec5 := DecodeCTCBeamSearch(logits, shape, 0, 5, false)
	if assert.Len(t, dec5, 1) {
		d5 := dec5[0]
		// Should still find the same path
		assert.Equal(t, []int{3}, d5.Sequence)
	}
}

func TestDecodeCTCBeamSearch_ClassesFirst(t *testing.T) {
	// Test with classes-first layout [N,C,T]
	shape := []int64{1, 4, 3} // [N=1, C=4, T=3]
	logits := []float32{
		// class 0 over timesteps: 0.1, 0.2, 0.9
		0.1, 0.2, 0.9,
		// class 1 over timesteps: 0.8, 0.7, 0.05
		0.8, 0.7, 0.05,
		// class 2 over timesteps: 0.05, 0.05, 0.6
		0.05, 0.05, 0.6,
		// class 3 over timesteps: 0.05, 0.05, 0.02
		0.05, 0.05, 0.02,
	}

	dec := DecodeCTCBeamSearch(logits, shape, 0, 3, true)
	if assert.Len(t, dec, 1) {
		d := dec[0]
		// t=0: class 1 wins, extends to [1]
		// t=1: class 1 wins again, same as last, no extension
		// t=2: class 0 (blank) wins, no extension
		assert.Equal(t, []int{1}, d.Sequence)
	}
}

func TestBeamCandidate_ExtendCandidate(t *testing.T) {
	candidate := BeamCandidate{
		Sequence:    []int{1},
		Probability: -0.1, // log(0.9)
		TimeStep:    0,
		LastChar:    1,
		CharProbs:   []float64{0.9},
	}

	// Test extending with same character (should NOT extend sequence in CTC beam search)
	extended := extendCandidateCTC(candidate, 1, -0.2, 0) // log(0.8)
	if assert.NotNil(t, extended) {
		assert.Equal(t, []int{1}, extended.Sequence)        // Same sequence (no extension)
		assert.InDelta(t, -0.3, extended.Probability, 1e-6) // -0.1 + -0.2
		assert.Equal(t, 1, extended.LastChar)               // Same last char
		assert.Equal(t, []float64{0.9}, extended.CharProbs) // Same char probs
	}

	// Test extending with different character
	extended2 := extendCandidateCTC(candidate, 2, math.Log(0.6), 0) // log(0.6)
	if assert.NotNil(t, extended2) {
		assert.Equal(t, []int{1, 2}, extended2.Sequence) // Extended sequence
		assert.InDelta(t, -0.1+math.Log(0.6), extended2.Probability, 1e-6)
		assert.Equal(t, 2, extended2.LastChar)
		assert.Len(t, extended2.CharProbs, 2)
		assert.InDelta(t, 0.9, extended2.CharProbs[0], 1e-6)
		assert.InDelta(t, 0.6, extended2.CharProbs[1], 1e-6)
	}

	// Test extending with blank
	extended3 := extendCandidateCTC(candidate, 0, -0.1, 0) // log(0.9), blank=0
	if assert.NotNil(t, extended3) {
		assert.Equal(t, []int{1}, extended3.Sequence)        // Same sequence
		assert.InDelta(t, -0.2, extended3.Probability, 1e-6) // -0.1 + -0.1
		assert.Equal(t, 1, extended3.LastChar)               // Same last char
		assert.Equal(t, []float64{0.9}, extended3.CharProbs) // Same char probs
	}
}
