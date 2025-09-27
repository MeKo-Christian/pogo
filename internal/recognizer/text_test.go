package recognizer

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPostProcessText_Defaults(t *testing.T) {
	in := "\uFEFF  Hello\tWorld\u200B!  "
	out := PostProcessText(in, DefaultCleanOptions())
	assert.Equal(t, "Hello World!", out)
}

func TestPostProcessText_LanguageReplacements(t *testing.T) {
	in := "“Quoted” and – dash — and nbsp\u00A0here"
	opts := DefaultCleanOptions()
	opts.Language = "en"
	out := PostProcessText(in, opts)
	assert.Equal(t, `"Quoted" and - dash - and nbsp here`, out)
}

func TestPostProcessText_WhitespaceCollapse(t *testing.T) {
	in := "Line\n\nwith\t\tmany   spaces"
	opts := DefaultCleanOptions()
	out := PostProcessText(in, opts)
	// Tabs/newlines collapsed to single spaces
	assert.Equal(t, "Line with many spaces", out)
}

func TestValidateText(t *testing.T) {
	assert.True(t, ValidateText("Hello 123"))
	assert.True(t, ValidateText(""))
	assert.False(t, ValidateText("\x00\x01\x02"))
}
