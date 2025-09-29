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

// BeamCandidate represents a single candidate in beam search.
type BeamCandidate struct {
	Sequence    []int     // Current sequence of character indices
	Probability float64   // Log probability of this sequence
	TimeStep    int       // Current time step
	LastChar    int       // Last character added (for CTC merge rules)
	CharProbs   []float64 // Per-character probabilities
}

// BeamSearchResult holds the result of beam search decoding.
type BeamSearchResult struct {
	Sequence    []int     // Final sequence (collapsed)
	Probability float64   // Log probability
	CharProbs   []float64 // Per-character probabilities
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

// isProbabilityDistribution checks if values look like probabilities (sum≈1 and in [0,1]).
func isProbabilityDistribution(v []float32) bool {
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
	return sum > 0.99 && sum < 1.01 && minV >= 0 && maxV <= 1
}

// computeSoftmaxProb computes the softmax probability of v[idx] using stable softmax.
func computeSoftmaxProb(v []float32, idx int) float64 {
	// Find max for numerical stability
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
	if denom == 0 {
		return 0
	}
	num := math.Exp(float64(v[idx] - m))
	return num / denom
}

// softmaxProbOfIndex computes the softmax probability of v[idx] among v.
// If values already look like probabilities (sum≈1 and in [0,1]), returns v[idx].
func softmaxProbOfIndex(v []float32, idx int) float64 {
	if len(v) == 0 || idx < 0 || idx >= len(v) {
		return 0
	}
	// Quick check for probability-like outputs
	if isProbabilityDistribution(v) {
		return float64(v[idx])
	}
	// Compute probability for arg via stable softmax
	return computeSoftmaxProb(v, idx)
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

// normalizeShape normalizes the shape by collapsing trailing dimensions of size 1.
func normalizeShape(shape []int64) []int64 {
	if len(shape) < 3 {
		return nil
	}
	dims := make([]int64, 0, len(shape))
	dims = append(dims, shape...)
	for len(dims) > 3 && dims[len(dims)-1] == 1 {
		dims = dims[:len(dims)-1]
	}
	return dims
}

// extractDimensions extracts tDim and cDim from normalized dimensions.
func extractDimensions(dims []int64, classesFirst bool) (int, int) {
	if len(dims) != 3 {
		return 0, 0
	}
	if classesFirst {
		return int(dims[2]), int(dims[1]) // tDim, cDim
	}
	return int(dims[1]), int(dims[2]) // tDim, cDim
}

// extractClassSlice extracts the class slice for a given timestep.
func extractClassSlice(logits []float32, start, t, tDim, cDim int, classesFirst bool) []float32 {
	if classesFirst {
		// [C, T]: for fixed t, classes are offset t + k*T
		clsSlice := make([]float32, cDim)
		for k := range cDim {
			clsSlice[k] = logits[start+k*tDim+t]
		}
		return clsSlice
	}
	// [T, C]: contiguous C at this step
	off := start + t*cDim
	return logits[off : off+cDim]
}

// DecodeCTCGreedy decodes logits with CTC greedy decoding.
// data layout can be [N, T, C] or [N, C, T]; specify classesFirst=true for [N,C,T].
func DecodeCTCGreedy(logits []float32, shape []int64, blank int, classesFirst bool) []DecodedSequence {
	dims := normalizeShape(shape)
	if dims == nil {
		return nil
	}
	n := int(dims[0])
	if n <= 0 {
		return nil
	}
	tDim, cDim := extractDimensions(dims, classesFirst)
	if tDim <= 0 || cDim <= 0 {
		return nil
	}

	out := make([]DecodedSequence, n)
	perBatch := tDim * cDim
	for b := range n {
		start := b * perBatch
		indices := make([]int, tDim)
		probs := make([]float64, tDim)
		for t := range tDim {
			clsSlice := extractClassSlice(logits, start, t, tDim, cDim, classesFirst)
			idx, _ := argmax(clsSlice)
			indices[t] = idx
			probs[t] = softmaxProbOfIndex(clsSlice, idx)
		}
		collIdx, collProb := CTCCollapse(indices, probs, blank)
		out[b] = DecodedSequence{Indices: indices, Probs: probs, Collapsed: collIdx, CollapsedProb: collProb}
	}
	return out
}

// DecodeCTCBeamSearch performs beam search CTC decoding.
// beamWidth controls how many candidates to keep at each step.
// Returns the best candidate found.
func DecodeCTCBeamSearch(logits []float32, shape []int64, blank int, beamWidth int, classesFirst bool) []BeamSearchResult {
	dims := normalizeShape(shape)
	if dims == nil || beamWidth <= 0 {
		return nil
	}
	n := int(dims[0])
	if n <= 0 {
		return nil
	}
	tDim, cDim := extractDimensions(dims, classesFirst)
	if tDim <= 0 || cDim <= 0 {
		return nil
	}

	out := make([]BeamSearchResult, n)
	perBatch := tDim * cDim
	for b := range n {
		start := b * perBatch
		candidates := beamSearchSingle(logits, start, tDim, cDim, blank, beamWidth, classesFirst)
		if len(candidates) > 0 {
			// Take the best candidate
			best := candidates[0]
			out[b] = BeamSearchResult{
				Sequence:    best.Sequence,
				Probability: best.Probability,
				CharProbs:   best.CharProbs,
			}
		}
	}
	return out
}

// beamSearchSingle performs beam search for a single sequence.
func beamSearchSingle(logits []float32, start, tDim, cDim, blank, beamWidth int, classesFirst bool) []BeamCandidate {
	// Initialize beam with empty sequence
	initial := BeamCandidate{
		Sequence:    []int{},
		Probability: 0.0, // log(1) = 0
		TimeStep:    -1,
		LastChar:    -1,
		CharProbs:   []float64{},
	}
	beam := []BeamCandidate{initial}

	// Process each timestep
	for t := 0; t < tDim; t++ {
		clsSlice := extractClassSlice(logits, start, t, tDim, cDim, classesFirst)
		beam = beamSearchStep(beam, clsSlice, blank, beamWidth)
		if len(beam) == 0 {
			break
		}
	}

	// Sort final beam by probability (highest first)
	sortBeamByProbability(beam)

	return beam
}

// beamSearchStep extends all candidates in the beam for one timestep.
// beamWidth controls how many candidates to keep at each step.
// Includes early pruning optimization.
func beamSearchStep(beam []BeamCandidate, clsProbs []float32, blank int, beamWidth int) []BeamCandidate {
	if len(beam) == 0 {
		return nil
	}

	// Estimate capacity - each candidate can produce up to 2 extensions (blank + non-blank)
	newBeam := make([]BeamCandidate, 0, len(beam)*2)

	// Track the best probability seen so far for early pruning
	var bestProb float64 = -1e10
	if len(beam) > 0 {
		bestProb = beam[0].Probability
	}

	for _, candidate := range beam {
		// Early pruning: skip candidates that are already much worse than the best
		// This is a heuristic to reduce computation for large beam widths
		if candidate.Probability < bestProb-10.0 { // More than 10 log-prob units worse
			continue
		}

		for charIdx, prob := range clsProbs {
			if prob <= 1e-7 { // Skip very low probability characters
				continue
			}

			logProb := math.Log(float64(prob))
			newCandidate := extendCandidateCTC(candidate, charIdx, logProb, blank)
			if newCandidate != nil {
				newBeam = append(newBeam, *newCandidate)

				// Update best probability for pruning
				if newCandidate.Probability > bestProb {
					bestProb = newCandidate.Probability
				}
			}
		}
	}

	// Sort by probability and keep top beamWidth
	sortBeamByProbability(newBeam)
	if len(newBeam) > beamWidth {
		newBeam = newBeam[:beamWidth]
	}

	return newBeam
}

// extendCandidateCTC extends a candidate with a new character according to CTC beam search rules.
// In CTC beam search, we maintain the prefix of the final collapsed sequence.
func extendCandidateCTC(candidate BeamCandidate, charIdx int, logProb float64, blank int) *BeamCandidate {
	// If this is a blank, we don't extend the sequence, just update probability
	if charIdx == blank {
		return &BeamCandidate{
			Sequence:    candidate.Sequence, // Same sequence
			Probability: candidate.Probability + logProb,
			TimeStep:    candidate.TimeStep + 1,
			LastChar:    candidate.LastChar,  // Same last char
			CharProbs:   candidate.CharProbs, // Same char probs
		}
	}

	// If this character is the same as the last one, we don't extend (CTC collapse rule)
	if charIdx == candidate.LastChar {
		return &BeamCandidate{
			Sequence:    candidate.Sequence, // Same sequence
			Probability: candidate.Probability + logProb,
			TimeStep:    candidate.TimeStep + 1,
			LastChar:    candidate.LastChar,  // Same last char
			CharProbs:   candidate.CharProbs, // Same char probs
		}
	}

	// Different character - extend the sequence
	newSeq := make([]int, len(candidate.Sequence)+1)
	copy(newSeq, candidate.Sequence)
	newSeq[len(candidate.Sequence)] = charIdx

	newCharProbs := make([]float64, len(candidate.CharProbs)+1)
	copy(newCharProbs, candidate.CharProbs)
	newCharProbs[len(candidate.CharProbs)] = math.Exp(logProb)

	return &BeamCandidate{
		Sequence:    newSeq,
		Probability: candidate.Probability + logProb,
		TimeStep:    candidate.TimeStep + 1,
		LastChar:    charIdx,
		CharProbs:   newCharProbs,
	}
}

// sortBeamByProbability sorts beam candidates by probability (highest first).
func sortBeamByProbability(beam []BeamCandidate) {
	// Simple insertion sort for small beam sizes
	for i := 1; i < len(beam); i++ {
		key := beam[i]
		j := i - 1
		for j >= 0 && beam[j].Probability < key.Probability {
			beam[j+1] = beam[j]
			j--
		}
		beam[j+1] = key
	}
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
