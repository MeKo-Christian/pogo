package recognizer

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDetectLanguage_Basic(t *testing.T) {
	assert.Equal(t, "en", DetectLanguage("This is a simple English sentence."))
	assert.Equal(t, "de", DetectLanguage("Grüße aus München, wie läuft's?"))
	assert.Equal(t, "fr", DetectLanguage("C'était une journée très spéciale."))
	assert.Equal(t, "es", DetectLanguage("¿Dónde está la biblioteca?"))
}
