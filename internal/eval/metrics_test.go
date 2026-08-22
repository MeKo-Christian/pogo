package eval

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNormalize(t *testing.T) {
	assert.Equal(t, "Hallo Welt!", Normalize("  Hallo   \n Welt!\t"))
	assert.Empty(t, Normalize("   "))
	// Case is preserved: a reading that gets the case wrong is wrong.
	assert.Equal(t, hello, Normalize(hello))
}

func TestLevenshtein(t *testing.T) {
	tests := []struct {
		name string
		a, b string
		want int
	}{
		{"both empty", "", "", 0},
		{"insert into empty", "", abc, 3},
		{"delete to empty", abc, "", 3},
		{"equal", abc, abc, 0},
		{"substitution", hello, hellg, 1},
		{"insertion", hello, "Helllo", 1},
		{"deletion", hello, "Helo", 1},
		{"multibyte", "Grüße", "Grusse", 3}, // ü->u, ß->ss
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, Levenshtein([]rune(tt.a), []rune(tt.b)))
			// Edit distance is symmetric.
			assert.Equal(t, tt.want, Levenshtein([]rune(tt.b), []rune(tt.a)))
		})
	}
}

func TestCER(t *testing.T) {
	tests := []struct {
		name          string
		expected, got string
		want          float64
	}{
		{"perfect", hello, hello, 0},
		{"one substitution of five", hello, hellg, 0.2},
		{"dropped space", rotatedText, "RotatedText", 1.0 / 12.0},
		{"whitespace only differs", rotatedText, " Rotated  Text ", 0},
		{"case differs", hello, "hello", 0.2},
		{"empty reading", hello, "", 1},
		{"both empty", "", "", 0},
		{"expected empty, got text", "", "noise", 1},
		{"capped at one", "ab", "xxxxxxxxxx", 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.InDelta(t, tt.want, CER(tt.expected, tt.got), 1e-9)
		})
	}
}

func TestWER(t *testing.T) {
	tests := []struct {
		name          string
		expected, got string
		want          float64
	}{
		{"perfect", rotatedText, rotatedText, 0},
		{"one word wrong of two", rotatedText, "Rotated Toxt", 0.5},
		{"dropped space merges words", rotatedText, "RotatedText", 1},
		// The bag-of-words rate this replaces scored word order 1.0.
		{"word order matters", rotatedText, "Text Rotated", 1},
		{"empty reading", rotatedText, "", 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.InDelta(t, tt.want, WER(tt.expected, tt.got), 1e-9)
		})
	}
}

func TestExactMatch(t *testing.T) {
	assert.True(t, ExactMatch("Hallo Welt!", " Hallo   Welt! "))
	assert.False(t, ExactMatch(hello, "hello"))
	assert.False(t, ExactMatch(rotatedText, "RotatedText"))
}

func TestErrorRatesAreBounded(t *testing.T) {
	for _, got := range []string{"", "x", "a very long and entirely wrong reading indeed"} {
		assert.GreaterOrEqual(t, CER(hello, got), 0.0)
		assert.LessOrEqual(t, CER(hello, got), 1.0)
		assert.False(t, math.IsNaN(WER(hello, got)))
	}
}

// Shared fixtures for the table tests in this package.
const (
	abc         = "abc"
	hello       = "Hello"
	hellg       = "Hellg"
	world       = "World"
	rotatedText = "Rotated Text"
)
