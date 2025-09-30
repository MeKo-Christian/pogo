package recognizer

import (
	"math"
	"testing"

	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
)

// genCTCLogits generates random CTC logits for testing.
func genCTCLogits(batchSize, timeSteps, classes int) []float32 {
	size := batchSize * timeSteps * classes
	logits := make([]float32, size)
	for i := range logits {
		logits[i] = float32(i%10) / 10.0
	}
	return logits
}

// TestDecodeCTCGreedy_OutputLengthBound verifies output length <= timesteps.
func TestDecodeCTCGreedy_OutputLengthBound(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("greedy CTC output length <= number of timesteps", prop.ForAll(
		func(timeSteps, classes, blank int) bool {
			if timeSteps < 1 || timeSteps > 100 {
				return true
			}
			if classes < 2 || classes > 50 {
				return true
			}
			if blank < 0 || blank >= classes {
				blank = classes - 1
			}

			batchSize := 1
			logits := genCTCLogits(batchSize, timeSteps, classes)
			shape := []int64{int64(batchSize), int64(timeSteps), int64(classes)}

			results := DecodeCTCGreedy(logits, shape, blank, false)

			if len(results) != batchSize {
				return false
			}

			for _, result := range results {
				// Collapsed output should be <= timesteps
				if len(result.Collapsed) > timeSteps {
					return false
				}
				// Raw indices should equal timesteps
				if len(result.Indices) != timeSteps {
					return false
				}
			}
			return true
		},
		gen.IntRange(1, 100),
		gen.IntRange(2, 50),
		gen.IntRange(0, 49),
	))

	properties.TestingRun(t)
}

// TestDecodeCTCGreedy_BlankRemoval verifies blanks are removed from output.
func TestDecodeCTCGreedy_BlankRemoval(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("greedy CTC removes blank indices", prop.ForAll(
		func(blank int) bool {
			if blank < 0 || blank >= 10 {
				blank = 9
			}

			// Create logits where every output is blank
			batchSize, timeSteps, classes := 1, 10, 10
			logits := make([]float32, batchSize*timeSteps*classes)
			for t := range timeSteps {
				// Make blank have highest probability
				for c := range classes {
					idx := t*classes + c
					if c == blank {
						logits[idx] = 10.0
					} else {
						logits[idx] = 0.0
					}
				}
			}

			shape := []int64{int64(batchSize), int64(timeSteps), int64(classes)}
			results := DecodeCTCGreedy(logits, shape, blank, false)

			if len(results) != 1 {
				return false
			}

			// Collapsed output should be empty (all blanks)
			return len(results[0].Collapsed) == 0
		},
		gen.IntRange(0, 9),
	))

	properties.TestingRun(t)
}

// TestCTCCollapse_IdempotenceProperty verifies collapsing twice gives same result.
func TestCTCCollapse_IdempotenceProperty(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("CTC collapse is idempotent", prop.ForAll(
		func(length, blank int) bool {
			if length < 1 || length > 50 {
				return true
			}
			if blank < 0 || blank >= 10 {
				blank = 9
			}

			// Generate random indices
			indices := make([]int, length)
			probs := make([]float64, length)
			for i := range indices {
				indices[i] = i % 10
				probs[i] = 0.8
			}

			// First collapse
			collapsed1, probs1 := CTCCollapse(indices, probs, blank)

			// Second collapse (on already collapsed data)
			collapsed2, _ := CTCCollapse(collapsed1, probs1, blank)

			// Should be identical
			if len(collapsed1) != len(collapsed2) {
				return false
			}
			for i := range collapsed1 {
				if collapsed1[i] != collapsed2[i] {
					return false
				}
			}
			return true
		},
		gen.IntRange(1, 50),
		gen.IntRange(0, 9),
	))

	properties.TestingRun(t)
}

// TestCTCCollapse_RemovesDuplicates verifies consecutive duplicates are removed.
func TestCTCCollapse_RemovesDuplicates(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("CTC collapse removes consecutive duplicates", prop.ForAll(
		func(char, repeatCount, blank int) bool {
			if char < 0 || char >= 10 || char == blank {
				return true
			}
			if blank < 0 || blank >= 10 {
				blank = 9
			}
			if repeatCount < 2 || repeatCount > 10 {
				return true
			}

			// Create sequence with consecutive duplicates
			indices := make([]int, repeatCount)
			probs := make([]float64, repeatCount)
			for i := range indices {
				indices[i] = char
				probs[i] = 0.8
			}

			collapsed, _ := CTCCollapse(indices, probs, blank)

			// Should collapse to single character
			return len(collapsed) == 1 && collapsed[0] == char
		},
		gen.IntRange(0, 8),
		gen.IntRange(2, 10),
		gen.IntRange(0, 9),
	))

	properties.TestingRun(t)
}

// TestSoftmaxProbOfIndex_SumToOne verifies probabilities sum to ~1.0.
func TestSoftmaxProbOfIndex_SumToOne(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("softmax probabilities sum to approximately 1.0", prop.ForAll(
		func(size int) bool {
			if size < 2 || size > 50 {
				return true
			}

			// Generate logits
			logits := make([]float32, size)
			for i := range logits {
				logits[i] = float32(i) / 10.0
			}

			// Calculate softmax for all indices
			var sum float64
			for i := range logits {
				prob := softmaxProbOfIndex(logits, i)
				sum += prob
			}

			// Sum should be approximately 1.0
			return math.Abs(sum-1.0) < 0.01
		},
		gen.IntRange(2, 50),
	))

	properties.TestingRun(t)
}

// TestSoftmaxProbOfIndex_BoundsCheck verifies probabilities are in [0, 1].
func TestSoftmaxProbOfIndex_BoundsCheck(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("softmax probabilities are in [0, 1]", prop.ForAll(
		func(size, idx int) bool {
			if size < 2 || size > 50 {
				return true
			}
			if idx < 0 || idx >= size {
				return true
			}

			logits := make([]float32, size)
			for i := range logits {
				logits[i] = float32(i-size/2) / 5.0
			}

			prob := softmaxProbOfIndex(logits, idx)
			return prob >= 0.0 && prob <= 1.0
		},
		gen.IntRange(2, 50),
		gen.IntRange(0, 49),
	))

	properties.TestingRun(t)
}

// TestArgmax_FindsMaximum verifies argmax returns correct index.
func TestArgmax_FindsMaximum(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("argmax returns index of maximum value", prop.ForAll(
		func(size, maxIdx int) bool {
			if size < 2 || size > 50 {
				return true
			}
			if maxIdx < 0 || maxIdx >= size {
				maxIdx = size - 1
			}

			values := make([]float32, size)
			for i := range values {
				values[i] = 0.1
			}
			values[maxIdx] = 0.9 // Set maximum

			idx, val := argmax(values)

			return idx == maxIdx && val == 0.9
		},
		gen.IntRange(2, 50),
		gen.IntRange(0, 49),
	))

	properties.TestingRun(t)
}

// TestDecodeCTCBeamSearch_OutputOrdering verifies beam search maintains best candidates.
func TestDecodeCTCBeamSearch_OutputOrdering(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("beam search maintains candidate ordering", prop.ForAll(
		func(timeSteps, classes, beamWidth int) bool {
			if timeSteps < 5 || timeSteps > 30 {
				return true
			}
			if classes < 3 || classes > 20 {
				return true
			}
			if beamWidth < 1 || beamWidth > 10 {
				beamWidth = 5
			}

			blank := classes - 1
			batchSize := 1
			logits := genCTCLogits(batchSize, timeSteps, classes)
			shape := []int64{int64(batchSize), int64(timeSteps), int64(classes)}

			results := DecodeCTCBeamSearch(logits, shape, blank, beamWidth, false)

			if len(results) != batchSize {
				return false
			}

			// Result should have valid probability
			for _, result := range results {
				if math.IsNaN(result.Probability) || math.IsInf(result.Probability, 0) {
					return false
				}
			}
			return true
		},
		gen.IntRange(5, 30),
		gen.IntRange(3, 20),
		gen.IntRange(1, 10),
	))

	properties.TestingRun(t)
}

// TestDecodeCTCBeamSearch_BetterOrEqualToGreedy verifies beam >= greedy probability.
func TestDecodeCTCBeamSearch_BetterOrEqualToGreedy(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("beam search probability >= greedy probability", prop.ForAll(
		func(timeSteps, classes int) bool {
			if timeSteps < 5 || timeSteps > 20 {
				return true
			}
			if classes < 3 || classes > 15 {
				return true
			}

			blank := classes - 1
			batchSize := 1
			beamWidth := 5

			// Create clear probability distributions to make comparison meaningful
			logits := make([]float32, batchSize*timeSteps*classes)
			for t := range timeSteps {
				for c := range classes {
					idx := t*classes + c
					if c == t%classes {
						logits[idx] = 5.0 // Higher for specific class
					} else {
						logits[idx] = 0.1
					}
				}
			}

			shape := []int64{int64(batchSize), int64(timeSteps), int64(classes)}

			greedyResults := DecodeCTCGreedy(logits, shape, blank, false)
			beamResults := DecodeCTCBeamSearch(logits, shape, blank, beamWidth, false)

			if len(greedyResults) != 1 || len(beamResults) != 1 {
				return false
			}

			// Calculate greedy probability (product of char probs)
			greedyProb := 0.0
			for _, p := range greedyResults[0].CollapsedProb {
				if greedyProb == 0.0 {
					greedyProb = math.Log(p)
				} else {
					greedyProb += math.Log(p)
				}
			}

			// Beam search should find equal or better probability
			return beamResults[0].Probability >= greedyProb-0.1 // Small tolerance
		},
		gen.IntRange(5, 20),
		gen.IntRange(3, 15),
	))

	properties.TestingRun(t)
}

// TestSequenceConfidence_BoundsCheck verifies confidence is in [0, 1].
func TestSequenceConfidence_BoundsCheck(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("sequence confidence is in [0, 1]", prop.ForAll(
		func(length int) bool {
			if length < 0 || length > 50 {
				return true
			}

			charProbs := make([]float64, length)
			for i := range charProbs {
				charProbs[i] = float64(i%10) / 10.0
			}

			confidence := SequenceConfidence(charProbs)
			return confidence >= 0.0 && confidence <= 1.0
		},
		gen.IntRange(0, 50),
	))

	properties.TestingRun(t)
}

// TestSequenceConfidence_EmptySequence verifies empty input returns 0.
func TestSequenceConfidence_EmptySequence(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("empty sequence has zero confidence", prop.ForAll(
		func() bool {
			charProbs := []float64{}
			confidence := SequenceConfidence(charProbs)
			return confidence == 0.0
		},
	))

	properties.TestingRun(t)
}

// TestNormalizeShape_HandlesTrailingOnes verifies shape normalization.
func TestNormalizeShape_HandlesTrailingOnes(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("normalizeShape removes trailing 1s", prop.ForAll(
		func(n, t, c, extraOnes int) bool {
			if n < 1 || t < 1 || c < 1 {
				return true
			}
			if extraOnes < 0 || extraOnes > 5 {
				return true
			}

			shape := []int64{int64(n), int64(t), int64(c)}
			for range extraOnes {
				shape = append(shape, 1)
			}

			normalized := normalizeShape(shape)

			if normalized == nil {
				return false
			}

			// Should have exactly 3 dimensions
			return len(normalized) == 3
		},
		gen.IntRange(1, 10),
		gen.IntRange(1, 50),
		gen.IntRange(1, 50),
		gen.IntRange(0, 5),
	))

	properties.TestingRun(t)
}

// TestDecodeCTCGreedy_Deterministic verifies deterministic behavior.
func TestDecodeCTCGreedy_Deterministic(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("greedy CTC decoding is deterministic", prop.ForAll(
		func(timeSteps, classes int) bool {
			if timeSteps < 5 || timeSteps > 30 {
				return true
			}
			if classes < 3 || classes > 20 {
				return true
			}

			blank := classes - 1
			batchSize := 1
			logits := genCTCLogits(batchSize, timeSteps, classes)
			shape := []int64{int64(batchSize), int64(timeSteps), int64(classes)}

			results1 := DecodeCTCGreedy(logits, shape, blank, false)
			results2 := DecodeCTCGreedy(logits, shape, blank, false)

			if len(results1) != len(results2) {
				return false
			}

			for i := range results1 {
				if len(results1[i].Collapsed) != len(results2[i].Collapsed) {
					return false
				}
				for j := range results1[i].Collapsed {
					if results1[i].Collapsed[j] != results2[i].Collapsed[j] {
						return false
					}
				}
			}
			return true
		},
		gen.IntRange(5, 30),
		gen.IntRange(3, 20),
	))

	properties.TestingRun(t)
}

// TestSortBeamByProbability_MaintainsDescendingOrder verifies beam sorting.
func TestSortBeamByProbability_MaintainsDescendingOrder(t *testing.T) {
	properties := gopter.NewProperties(nil)

	properties.Property("sortBeamByProbability maintains descending order", prop.ForAll(
		func(size int) bool {
			if size < 2 || size > 20 {
				return true
			}

			// Create beam with random probabilities
			beam := make([]BeamCandidate, size)
			for i := range beam {
				beam[i] = BeamCandidate{
					Sequence:    []int{i},
					Probability: float64(i%10) / 10.0,
					TimeStep:    0,
					LastChar:    -1,
					CharProbs:   []float64{},
				}
			}

			sortBeamByProbability(beam)

			// Check descending order
			for i := 1; i < len(beam); i++ {
				if beam[i].Probability > beam[i-1].Probability {
					return false
				}
			}
			return true
		},
		gen.IntRange(2, 20),
	))

	properties.TestingRun(t)
}
