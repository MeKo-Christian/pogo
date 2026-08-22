package eval

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func writeManifest(t *testing.T, root, body string) string {
	t.Helper()
	p := filepath.Join(root, "manifest.json")
	require.NoError(t, os.WriteFile(p, []byte(body), 0o600))
	return p
}

func TestLoadManifest(t *testing.T) {
	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, "a.png"), []byte("not really a png"), 0o600))

	p := writeManifest(t, root, `[{"group":"upright","image":"a.png","expected":"Hello"}]`)
	cases, err := LoadManifest(p, root)
	require.NoError(t, err)
	require.Len(t, cases, 1)
	assert.Equal(t, "Hello", cases[0].Expected)
}

func TestLoadManifestRejectsBadRows(t *testing.T) {
	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, "a.png"), []byte("x"), 0o600))

	tests := []struct {
		name, body, wantErr string
	}{
		{"empty expected", `[{"group":"upright","image":"a.png","expected":""}]`, "hand-keyed ground truth"},
		{"blank expected", `[{"group":"upright","image":"a.png","expected":"   "}]`, "hand-keyed ground truth"},
		{"unknown group", `[{"group":"sideways","image":"a.png","expected":"Hi"}]`, "unknown group"},
		{"missing image", `[{"group":"upright","image":"nope.png","expected":"Hi"}]`, "image not found"},
		{"no cases", `[]`, "no cases"},
		{"not json", `{`, "parse manifest"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := LoadManifest(writeManifest(t, root, tt.body), root)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}

// The shipped manifest must load and must carry no threshold of its own.
func TestShippedManifestHasNoThresholds(t *testing.T) {
	root, err := filepath.Abs("../..")
	require.NoError(t, err)
	p := filepath.Join(root, "testdata", "fixtures", "ocr_accuracy.json")
	cases, err := LoadManifest(p, root)
	require.NoError(t, err)
	assert.Len(t, cases, 14)

	raw, err := os.ReadFile(p) //nolint:gosec // G304: fixed path inside the repo.
	require.NoError(t, err)
	assert.NotContains(t, string(raw), "min_", "the manifest must carry ground truth only")
}
