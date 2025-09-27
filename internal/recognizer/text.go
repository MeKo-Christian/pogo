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
	// Unicode normalization
	switch strings.ToUpper(opts.NormalizeForm) {
	case "NFC", "":
		s = norm.NFC.String(s)
	case "NFKC":
		s = norm.NFKC.String(s)
	case "NFD":
		s = norm.NFD.String(s)
	case "NFKD":
		s = norm.NFKD.String(s)
	}

	// Remove zero-width characters
	if opts.RemoveZeroWidth {
		s = removeZeroWidth(s)
	}

	// Remove control chars (except tab, newline, carriage return)
	if opts.RemoveControlChars {
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
		s = b.String()
	}

	// Apply language-specific replacements if requested
	if opts.Language != "" && opts.ReplaceMap == nil {
		opts.ReplaceMap = DefaultReplaceMapForLanguage(opts.Language)
	}

	// Apply replacements
	if len(opts.ReplaceMap) > 0 {
		// Replace longer keys first to avoid partial overlaps
		keys := make([]string, 0, len(opts.ReplaceMap))
		for k := range opts.ReplaceMap {
			keys = append(keys, k)
		}
		// simple length-desc sort
		for i := range len(keys) - 1 {
			maxIdx := i
			for j := i + 1; j < len(keys); j++ {
				if len(keys[j]) > len(keys[maxIdx]) {
					maxIdx = j
				}
			}
			keys[i], keys[maxIdx] = keys[maxIdx], keys[i]
		}
		for _, k := range keys {
			s = strings.ReplaceAll(s, k, opts.ReplaceMap[k])
		}
	}

	// Collapse whitespace
	if opts.CollapseWhitespace {
		s = collapseWhitespace(s)
	}

	if opts.Trim {
		s = strings.TrimSpace(s)
	}
	return s
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
	if total == 0 {
		return true
	}
	controlRatio := float64(controls) / float64(total)
	textRatio := float64(letters+digits) / float64(total)
	return controlRatio < 0.05 && textRatio > 0.3
}
