package recognizer

import (
	"unicode"
)

// DetectLanguage performs a lightweight heuristic language detection on text.
// Returns a BCP-47-like lowercase code (e.g., "en", "de", "fr", "es") or "" if unknown.
// This is intended for post-processing hints, not for model selection.
func DetectLanguage(s string) string {
	if s == "" {
		return ""
	}
	var letters, ascii, extended, german, french, spanish int
	// Count features
	for _, r := range s {
		if unicode.IsLetter(r) {
			letters++
		}
		switch {
		case r <= 0x007F:
			if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') {
				ascii++
			}
		case r >= 0x00C0 && r <= 0x017F: // Latin-1 supplements + extensions
			extended++
			// Diacritics hints
			switch r {
			case 'ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß':
				german++
			case 'è', 'ê', 'à', 'ù', 'ç', 'È', 'À', 'Ç':
				french++
			case 'á', 'í', 'ó', 'ú', 'ñ', 'Á', 'Í', 'Ó', 'Ú', 'Ñ':
				spanish++
			}
		default:
			// Non-Latin scripts -> unknown here
		}
	}
	if letters == 0 {
		return ""
	}
	// Simple rules
	if german > french && german > spanish {
		return "de"
	}
	if french > german && french > spanish {
		return "fr"
	}
	if spanish > german && spanish > french {
		return "es"
	}
	// Predominantly ASCII letters -> assume English
	if ascii > 0 && ascii*100/letters > 80 {
		return "en"
	}
	// If extended latin appears and not strongly indicating others, leave blank
	_ = extended
	return ""
}
