package recognizer

import (
	"math"
)

// DecodedSequence holds CTC-decoded indices and per-character probabilities.
type DecodedSequence struct {
	Indices       []int
	Probs         []float64
	Collapsed     []int
	CollapsedProb []float64
}

// argmax returns index of max value and the value.
func argmax(v []float32) (int, float32) {
	if len(v) == 0 {
		return -1, 0
	}
	idx := 0
	maxVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > maxVal {
			maxVal = v[i]
			idx = i
		}
	}
	return idx, maxVal
}

// softmaxProbOfIndex computes the softmax probability of v[idx] among v.
// If values already look like probabilities (sum≈1 and in [0,1]), returns v[idx].
func softmaxProbOfIndex(v []float32, idx int) float64 {
	if len(v) == 0 || idx < 0 || idx >= len(v) {
		return 0
	}
	// Quick check for probability-like outputs
	var sum float64
	minV, maxV := v[0], v[0]
	for _, x := range v {
		sum += float64(x)
		if x < minV {
			minV = x
		}
		if x > maxV {
			maxV = x
		}
	}
	if sum > 0.99 && sum < 1.01 && minV >= 0 && maxV <= 1 {
		return float64(v[idx])
	}
	// Compute probability for arg via stable softmax
	// p_i = exp(x_i - m) / sum_j exp(x_j - m)
	m := float32(0)
	for i, x := range v {
		if i == 0 || x > m {
			m = x
		}
	}
	var denom float64
	for _, x := range v {
		denom += math.Exp(float64(x - m))
	}
	num := math.Exp(float64(v[idx] - m))
	if denom == 0 {
		return 0
	}
	return num / denom
}

// CTCCollapse removes repeated consecutive indices and blanks, returning collapsed sequence and probs.
func CTCCollapse(indices []int, probs []float64, blank int) ([]int, []float64) {
	outIdx := make([]int, 0, len(indices))
	outProb := make([]float64, 0, len(probs))
	prev := -1
	for i, idx := range indices {
		if idx == blank { // drop blanks
			prev = idx
			continue
		}
		if idx == prev { // collapse repeats
			continue
		}
		outIdx = append(outIdx, idx)
		if i < len(probs) {
			outProb = append(outProb, probs[i])
		} else {
			outProb = append(outProb, 0)
		}
		prev = idx
	}
	return outIdx, outProb
}

// DecodeCTCGreedy decodes logits with CTC greedy decoding.
// data layout can be [N, T, C] or [N, C, T]; specify classesFirst=true for [N,C,T].
func DecodeCTCGreedy(logits []float32, shape []int64, blank int, classesFirst bool) []DecodedSequence {
	// Normalize shape: expect rank 3 [N, T, C] or [N, C, T]
	if len(shape) < 3 {
		return nil
	}
	// Collapse trailing dims of size 1
	dims := make([]int64, 0, len(shape))
	dims = append(dims, shape...)
	for len(dims) > 3 && dims[len(dims)-1] == 1 {
		dims = dims[:len(dims)-1]
	}
	n := int(dims[0])
	if n <= 0 {
		return nil
	}
	var tDim, cDim int
	if classesFirst {
		cDim = int(dims[1])
		tDim = int(dims[2])
	} else {
		tDim = int(dims[1])
		cDim = int(dims[2])
	}
	if tDim <= 0 || cDim <= 0 {
		return nil
	}

	out := make([]DecodedSequence, n)
	// Strides
	perBatch := tDim * cDim
	for b := range n {
		start := b * perBatch
		indices := make([]int, tDim)
		probs := make([]float64, tDim)
		for t := range tDim {
			// slice of classes for this timestep
			var clsSlice []float32
			if classesFirst {
				// [C, T]: for fixed t, classes are offset t + k*T
				clsSlice = make([]float32, cDim)
				for k := range cDim {
					clsSlice[k] = logits[start+k*tDim+t]
				}
			} else {
				// [T, C]: contiguous C at this step
				off := start + t*cDim
				clsSlice = logits[off : off+cDim]
			}
			idx, _ := argmax(clsSlice)
			indices[t] = idx
			probs[t] = softmaxProbOfIndex(clsSlice, idx)
		}
		collIdx, collProb := CTCCollapse(indices, probs, blank)
		out[b] = DecodedSequence{Indices: indices, Probs: probs, Collapsed: collIdx, CollapsedProb: collProb}
	}
	return out
}

// SequenceConfidence returns the average of per-character probabilities; 0 if empty.
func SequenceConfidence(charProbs []float64) float64 {
	if len(charProbs) == 0 {
		return 0
	}
	var s float64
	for _, p := range charProbs {
		s += p
	}
	return s / float64(len(charProbs))
}
