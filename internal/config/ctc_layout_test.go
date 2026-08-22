package config

import (
	"testing"

	"github.com/MeKo-Tech/pogo/internal/recognizer"
	"github.com/stretchr/testify/assert"
)

// TestToRecognizerConfig_CarriesCTCLayout proves the declared CTC layout and
// blank index reach recognizer.Config through the normal YAML/JSON/CLI path,
// rather than being reachable only from Go code that builds the low-level
// config by hand.
func TestToRecognizerConfig_CarriesCTCLayout(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Pipeline.Recognizer.CTCLayout = "nct"
	cfg.Pipeline.Recognizer.BlankIndex = 3

	recCfg := cfg.toRecognizerConfig()

	assert.Equal(t, recognizer.LayoutNCT, recCfg.CTCLayout)
	assert.Equal(t, 3, recCfg.BlankIndex)
}

// TestToRecognizerConfig_EmptyLayoutKeepsTheDefault makes sure an unset layout
// does not clobber the recognizer default.
func TestToRecognizerConfig_EmptyLayoutKeepsTheDefault(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Pipeline.Recognizer.CTCLayout = ""

	recCfg := cfg.toRecognizerConfig()

	assert.Equal(t, recognizer.DefaultConfig().CTCLayout, recCfg.CTCLayout)
}

// TestDefaultRecognizerConfig_ExposesCTCLayout checks the application default
// mirrors the recognizer default, so a freshly written config file shows the
// layout instead of an empty string.
func TestDefaultRecognizerConfig_ExposesCTCLayout(t *testing.T) {
	def := defaultRecognizerConfig()

	assert.Equal(t, string(recognizer.DefaultConfig().CTCLayout), def.CTCLayout)
	assert.Equal(t, recognizer.DefaultConfig().BlankIndex, def.BlankIndex)
}
