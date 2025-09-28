package recognizer

import (
	"regexp"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// CleanOptions controls text post-processing behavior.
type CleanOptions struct {
	NormalizeForm      string            // "NFC" (default), "NFKC", "NFD", "NFKD", "" to disable
	CollapseWhitespace bool              // collapse runs of whitespace to a single space
	Trim               bool              // trim leading/trailing whitespace
	RemoveControlChars bool              // remove non-printable control characters
	RemoveZeroWidth    bool              // remove zero-width spaces/joiners
	ReplaceMap         map[string]string // string replacements applied after normalization
	Language           string            // optional language tag (e.g., "en", "de"); affects default ReplaceMap
}

// DefaultCleanOptions returns sensible defaults for OCR text.
func DefaultCleanOptions() CleanOptions {
	return CleanOptions{
		NormalizeForm:      "NFC",
		CollapseWhitespace: true,
		Trim:               true,
		RemoveControlChars: true,
		RemoveZeroWidth:    true,
		ReplaceMap:         nil,
		Language:           "",
	}
}

// PostProcessText applies normalization and cleaning to OCR text.
func PostProcessText(s string, opts CleanOptions) string {
	if s == "" {
		return s
	}

	s = applyNormalization(s, opts)
	s = applyZeroWidthRemoval(s, opts)
	s = applyControlCharRemoval(s, opts)
	s = applyReplacements(s, opts)
	s = applyWhitespaceCollapse(s, opts)
	s = applyTrim(s, opts)

	return s
}

func applyNormalization(s string, opts CleanOptions) string {
	switch strings.ToUpper(opts.NormalizeForm) {
	case "NFC", "":
		return norm.NFC.String(s)
	case "NFKC":
		return norm.NFKC.String(s)
	case "NFD":
		return norm.NFD.String(s)
	case "NFKD":
		return norm.NFKD.String(s)
	}
	return s
}

func applyZeroWidthRemoval(s string, opts CleanOptions) string {
	if opts.RemoveZeroWidth {
		return removeZeroWidth(s)
	}
	return s
}

func applyControlCharRemoval(s string, opts CleanOptions) string {
	if opts.RemoveControlChars {
		return removeControlChars(s)
	}
	return s
}

func applyReplacements(s string, opts CleanOptions) string {
	if len(opts.ReplaceMap) > 0 {
		return applyReplaceMap(s, opts.ReplaceMap)
	}
	if opts.Language != "" {
		replaceMap := DefaultReplaceMapForLanguage(opts.Language)
		return applyReplaceMap(s, replaceMap)
	}
	return s
}

func applyReplaceMap(s string, replaceMap map[string]string) string {
	// Replace longer keys first to avoid partial overlaps
	keys := sortedKeysByLength(replaceMap)
	for _, k := range keys {
		s = strings.ReplaceAll(s, k, replaceMap[k])
	}
	return s
}

func sortedKeysByLength(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	// Sort by length descending
	for i := range len(keys) - 1 {
		for j := i + 1; j < len(keys); j++ {
			if len(keys[j]) > len(keys[i]) {
				keys[i], keys[j] = keys[j], keys[i]
			}
		}
	}
	return keys
}

func applyWhitespaceCollapse(s string, opts CleanOptions) string {
	if opts.CollapseWhitespace {
		return collapseWhitespace(s)
	}
	return s
}

func applyTrim(s string, opts CleanOptions) string {
	if opts.Trim {
		return strings.TrimSpace(s)
	}
	return s
}

func removeControlChars(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if r == '\n' || r == '\r' || r == '\t' {
			b.WriteRune(r)
			continue
		}
		if unicode.IsControl(r) {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// DefaultReplaceMapForLanguage returns common replacements for a language.
// These are light-touch rules aimed at cleaning OCR artifacts and typographical quotes.
func DefaultReplaceMapForLanguage(lang string) map[string]string {
	lang = strings.ToLower(lang)
	m := map[string]string{
		"\u2018": "'",  // ‘
		"\u2019": "'",  // ’
		"\u201C": "\"", // “
		"\u201D": "\"", // ”
		"\u2013": "-",  // – en dash
		"\u2014": "-",  // — em dash
		"\u00A0": " ",  // non-breaking space
		"\u2009": " ",  // thin space
	}
	switch lang {
	case "de":
		// German quotes „“… to ""
		m["\u201E"] = "\"" // „
		m["\u201C"] = "\"" // “
	case "fr":
		// French guillemets « » with spaces to quotes
		m["\u00AB"] = "\"" // «
		m["\u00BB"] = "\"" // »
	}
	return m
}

var wsRe = regexp.MustCompile(`\s+`)

func collapseWhitespace(s string) string { return wsRe.ReplaceAllString(s, " ") }

// removeZeroWidth removes common zero-width characters used in OCR noise.
func removeZeroWidth(s string) string {
	if s == "" {
		return s
	}
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch r {
		case '\u200B', // ZERO WIDTH SPACE
			'\u200C', // ZERO WIDTH NON-JOINER
			'\u200D', // ZERO WIDTH JOINER
			'\uFEFF': // ZERO WIDTH NO-BREAK SPACE (BOM)
			// skip
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// ValidateText performs simple validation checks and returns true if the text looks valid.
// It checks for a minimum ratio of letters/digits and a max control-char ratio.
func ValidateText(s string) bool {
	if s == "" {
		return true
	}
	letters, digits, controls, total := countCharacters(s)
	if total == 0 {
		return true
	}
	return validateRatios(controls, letters+digits, total)
}

// countCharacters counts different types of characters in the string.
func countCharacters(s string) (int, int, int, int) {
	var letters, digits, controls, total int
	for _, r := range s {
		total++
		switch {
		case unicode.IsLetter(r):
			letters++
		case unicode.IsDigit(r):
			digits++
		case unicode.IsControl(r):
			if r != '\n' && r != '\r' && r != '\t' {
				controls++
			}
		}
	}
	return letters, digits, controls, total
}

// validateRatios checks if the character ratios meet validation criteria.
func validateRatios(controls, textChars, total int) bool {
	controlRatio := float64(controls) / float64(total)
	textRatio := float64(textChars) / float64(total)
	return controlRatio < 0.05 && textRatio > 0.3
}
