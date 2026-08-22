package recognizer

import "fmt"

// CTCLayout declares the axis order of a recognition model's output tensor.
//
// It replaces the old determineClassesFirst heuristic, which compared the
// dictionary size against the tensor dimensions and silently defaulted to NTC
// whenever neither matched. With PP-OCRv5 that comparison never matched — the
// dictionary implied 18384 classes while the model declares 18385 — so the
// heuristic fell through to its default and was right only by accident.
type CTCLayout string

const (
	// LayoutNTC is [N, T, C]: batch, timesteps, classes. PaddleOCR's default.
	LayoutNTC CTCLayout = "ntc"
	// LayoutNCT is [N, C, T]: batch, classes, timesteps.
	LayoutNCT CTCLayout = "nct"
)

// classesFirst reports whether the class dimension precedes the time dimension,
// which is what the CTC decoders take as their indexing flag.
func (l CTCLayout) classesFirst() bool { return l == LayoutNCT }

// validate rejects anything that is not one of the two known layouts. The empty
// string is accepted and means "unset", which callers resolve to LayoutNTC.
func (l CTCLayout) validate() error {
	switch l {
	case "", LayoutNTC, LayoutNCT:
		return nil
	default:
		return fmt.Errorf("unknown ctc layout %q (want %q or %q)", string(l), LayoutNTC, LayoutNCT)
	}
}

// ctcLayout resolves the configured layout, defaulting to LayoutNTC when unset
// so that zero-valued Configs keep working.
func (c Config) ctcLayout() CTCLayout {
	if c.CTCLayout == "" {
		return LayoutNTC
	}
	return c.CTCLayout
}
