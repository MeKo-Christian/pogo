// Package eval measures OCR accuracy against a ground-truth corpus.
//
// The metrics live here rather than in a test file so that both `go test` and
// the `pogo eval` command measure the same way. Thresholds do not live here at
// all: see gate.go, and note that no per-case threshold exists anywhere.
package eval

import "strings"

// Normalize collapses runs of whitespace to single spaces and trims the ends.
//
// It deliberately does not fold case. The metrics this package computes are
// case-sensitive, because an OCR engine that reads "hello" for "Hello" is
// wrong, and a case-folding metric cannot say so.
func Normalize(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// Levenshtein returns the edit distance between two sequences, counting one
// insertion, deletion or substitution as one edit. It is generic so the same
// implementation serves both the character and the word metric.
func Levenshtein[T comparable](a, b []T) int {
	if len(a) == 0 {
		return len(b)
	}
	if len(b) == 0 {
		return len(a)
	}
	prev := make([]int, len(b)+1)
	cur := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 0
			if a[i-1] != b[j-1] {
				cost = 1
			}
			cur[j] = min(prev[j]+1, min(cur[j-1]+1, prev[j-1]+cost))
		}
		copy(prev, cur)
	}
	return prev[len(b)]
}

// errorRate is the shared body of CER and WER: edit distance over the length of
// the ground truth, capped at 1 so that a wildly over-long reading cannot drag
// a corpus mean past "everything wrong".
func errorRate(distance, expectedLen int) float64 {
	if expectedLen == 0 {
		if distance == 0 {
			return 0
		}
		return 1
	}
	r := float64(distance) / float64(expectedLen)
	if r > 1 {
		return 1
	}
	return r
}

// CER is the character error rate: rune edit distance divided by the length of
// the expected text. 0 is a perfect reading; 1 is the worst reported value.
func CER(expected, got string) float64 {
	e := []rune(Normalize(expected))
	g := []rune(Normalize(got))
	return errorRate(Levenshtein(e, g), len(e))
}

// WER is the word error rate: edit distance over the word *sequence*, divided
// by the expected word count.
//
// This is positional, unlike the bag-of-words ratio it replaces, so reading
// "Text Rotated" for "Rotated Text" is now an error rather than a perfect score.
func WER(expected, got string) float64 {
	e := strings.Fields(expected)
	g := strings.Fields(got)
	return errorRate(Levenshtein(e, g), len(e))
}

// ExactMatch reports whether the reading equals the ground truth once
// whitespace is normalized. Case and punctuation must match.
func ExactMatch(expected, got string) bool {
	return Normalize(expected) == Normalize(got)
}
