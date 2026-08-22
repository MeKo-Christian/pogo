package eval

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// Case is one ground-truth entry of a corpus manifest.
//
// It carries what is true about the image and nothing about what counts as a
// pass. Thresholds used to live beside the ground truth, which meant a failing
// case could be fixed by editing its own row; the gate now lives in Go (gate.go)
// and applies to the corpus as a whole.
type Case struct {
	// Group tiers the corpus. Every group must have a Gate.
	Group string `json:"group"`
	// Image is the path to the fixture, relative to the project root.
	Image string `json:"image"`
	// Expected is the hand-keyed ground truth.
	Expected string `json:"expected"`
}

// LoadManifest reads a corpus manifest and validates every row against the
// project root the images are resolved from.
func LoadManifest(path, root string) ([]Case, error) {
	data, err := os.ReadFile(path) //nolint:gosec // G304: corpus path comes from the operator, not from a request.
	if err != nil {
		return nil, fmt.Errorf("read manifest %s: %w", path, err)
	}
	var cases []Case
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("parse manifest %s: %w", path, err)
	}
	if len(cases) == 0 {
		return nil, fmt.Errorf("manifest %s has no cases", path)
	}
	for i, c := range cases {
		if err := c.validate(root); err != nil {
			return nil, fmt.Errorf("manifest %s case %d: %w", path, i, err)
		}
	}
	return cases, nil
}

func (c Case) validate(root string) error {
	if c.Image == "" {
		return errors.New("empty image path")
	}
	if Normalize(c.Expected) == "" {
		return fmt.Errorf("%s: empty expected text; every case needs hand-keyed ground truth", c.Image)
	}
	if _, ok := Gates[c.Group]; !ok {
		return fmt.Errorf("%s: unknown group %q; add a gate for it or fix the typo", c.Image, c.Group)
	}
	p := filepath.Join(root, c.Image)
	if _, err := os.Stat(p); err != nil {
		return fmt.Errorf("%s: image not found at %s: %w", c.Image, p, err)
	}
	return nil
}
