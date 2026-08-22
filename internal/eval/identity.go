package eval

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"sort"
)

// ModelIdentity says which weights produced a measurement.
//
// The variant is for humans. The fingerprint is what the comparison actually
// trusts: a name like "mobile" says nothing once --models-dir or
// GO_OAR_OCR_MODELS_DIR points somewhere else, so two experiments could
// overwrite or be compared against each other's numbers and the difference
// would be read as an accuracy change.
type ModelIdentity struct {
	Variant     string `json:"variant"`
	Fingerprint string `json:"fingerprint"`
}

// Fingerprint hashes the artifacts a measurement depends on — the detector and
// recognizer graphs and every dictionary — into one identity. Files are hashed
// by content, so a rebuilt or swapped model is a different identity even at the
// same path.
func Fingerprint(variant string, paths []string) (ModelIdentity, error) {
	sorted := append([]string{}, paths...)
	sort.Strings(sorted)
	sum := sha256.New()
	for _, p := range sorted {
		h, err := hashFile(p)
		if err != nil {
			return ModelIdentity{}, err
		}
		// The path is folded in as well, so moving a model to a different
		// filename is visible even when the bytes are identical.
		// Writing to a hash never fails.
		_, _ = fmt.Fprintf(sum, "%s:%s\n", p, h)
	}
	return ModelIdentity{Variant: variant, Fingerprint: hex.EncodeToString(sum.Sum(nil))[:16]}, nil
}

func hashFile(path string) (string, error) {
	f, err := os.Open(path) //nolint:gosec // G304: paths come from the resolved pipeline config, not from a request.
	if err != nil {
		return "", fmt.Errorf("fingerprint %s: %w", path, err)
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fmt.Errorf("fingerprint %s: %w", path, err)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// String renders the identity for an error message.
func (m ModelIdentity) String() string {
	return fmt.Sprintf("%s (%s)", m.Variant, m.Fingerprint)
}

// Matches reports whether two measurements are comparable at all.
func (m ModelIdentity) Matches(other ModelIdentity) bool {
	return m.Variant == other.Variant && m.Fingerprint == other.Fingerprint
}
