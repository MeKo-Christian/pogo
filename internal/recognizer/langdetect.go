package recognizer

import (
	"unicode"
)

type languageStats struct {
	letters, ascii, extended, german, french, spanish int
}

// DetectLanguage performs a lightweight heuristic language detection on text.
// Returns a BCP-47-like lowercase code (e.g., "en", "de", "fr", "es") or "" if unknown.
// This is intended for post-processing hints, not for model selection.
func DetectLanguage(s string) string {
	if s == "" {
		return ""
	}

	stats := analyzeText(s)
	return determineLanguage(stats)
}

func analyzeText(s string) languageStats {
	var stats languageStats

	for _, r := range s {
		if unicode.IsLetter(r) {
			stats.letters++
		}

		switch {
		case r <= 0x007F:
			if isASCIILetter(r) {
				stats.ascii++
			}
		case r >= 0x00C0 && r <= 0x017F: // Latin-1 supplements + extensions
			stats.extended++
			updateDiacriticCounts(r, &stats)
		default:
			// Non-Latin scripts -> unknown here
		}
	}

	return stats
}

func isASCIILetter(r rune) bool {
	return (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z')
}

func updateDiacriticCounts(r rune, stats *languageStats) {
	switch r {
	case 'ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß':
		stats.german++
	case 'è', 'ê', 'à', 'ù', 'ç', 'È', 'À', 'Ç':
		stats.french++
	case 'á', 'í', 'ó', 'ú', 'ñ', 'Á', 'Í', 'Ó', 'Ú', 'Ñ':
		stats.spanish++
	}
}

func determineLanguage(stats languageStats) string {
	if stats.letters == 0 {
		return ""
	}

	// Check for specific language indicators
	if stats.german > stats.french && stats.german > stats.spanish {
		return "de"
	}
	if stats.french > stats.german && stats.french > stats.spanish {
		return "fr"
	}
	if stats.spanish > stats.german && stats.spanish > stats.french {
		return "es"
	}

	// Predominantly ASCII letters -> assume English
	if stats.ascii > 0 && stats.ascii*100/stats.letters > 80 {
		return "en"
	}

	return ""
}
