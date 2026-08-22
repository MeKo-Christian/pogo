package eval

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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
	// Image is the path to the fixture, relative to the manifest that names it.
	Image string `json:"image"`
	// Expected is the hand-keyed ground truth.
	Expected string `json:"expected"`
}

// ManifestFileName is the file that marks a directory as a corpus.
const ManifestFileName = "manifest.json"

// LoadManifest reads a corpus manifest and validates every row.
//
// Image paths are resolved against the manifest's own directory, so a corpus is
// a self-contained directory that can be moved or shipped without rewriting it.
func LoadManifest(path string) ([]Case, error) {
	root := filepath.Dir(path)
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

// Path resolves the case's image against the corpus directory root.
func (c Case) Path(root string) string { return filepath.Join(root, c.Image) }

func (c Case) validate(root string) error {
	if c.Image == "" {
		return errors.New("empty image path")
	}
	// A corpus must contain its own images. Escaping the corpus directory would
	// make it depend on the tree that happens to surround it, which is exactly
	// what "the manifest resolves its own paths" is meant to avoid.
	if filepath.IsAbs(c.Image) {
		return fmt.Errorf("%s: image path must be relative to the corpus, not absolute", c.Image)
	}
	rel, err := filepath.Rel(root, c.Path(root))
	if err != nil || strings.HasPrefix(rel, "..") {
		return fmt.Errorf("%s: image path escapes the corpus directory %s", c.Image, root)
	}
	if Normalize(c.Expected) == "" {
		return fmt.Errorf("%s: empty expected text; every case needs hand-keyed ground truth", c.Image)
	}
	if _, ok := Gates[c.Group]; !ok {
		return fmt.Errorf("%s: unknown group %q; add a gate for it or fix the typo", c.Image, c.Group)
	}
	p := c.Path(root)
	if _, err := os.Stat(p); err != nil {
		return fmt.Errorf("%s: image not found at %s: %w", c.Image, p, err)
	}
	return nil
}
