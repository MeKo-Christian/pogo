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
	cases, err := LoadManifest(p)
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
			_, err := LoadManifest(writeManifest(t, root, tt.body))
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}

// The shipped manifest must load and must carry no threshold of its own.
func TestShippedManifestHasNoThresholds(t *testing.T) {
	root, err := filepath.Abs("../..")
	require.NoError(t, err)
	p := filepath.Join(root, "testdata", "corpus", "synthetic", ManifestFileName)
	cases, err := LoadManifest(p)
	require.NoError(t, err)
	assert.Len(t, cases, 14)

	raw, err := os.ReadFile(p) //nolint:gosec // G304: fixed path inside the repo.
	require.NoError(t, err)
	assert.NotContains(t, string(raw), "min_", "the manifest must carry ground truth only")
}

// A corpus must contain its images. Escaping the corpus directory is what makes
// one non-portable, so it is rejected at load rather than discovered later as a
// missing file.
func TestLoadManifestRejectsPathsOutsideTheCorpus(t *testing.T) {
	root := t.TempDir()
	outside := filepath.Join(t.TempDir(), "a.png")
	require.NoError(t, os.WriteFile(outside, []byte("x"), 0o600))

	p := writeManifest(t, root, `[{"group":"upright","image":"../a.png","expected":"Hi"}]`)
	_, err := LoadManifest(p)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "escapes the corpus directory")

	p = writeManifest(t, root, `[{"group":"upright","image":"`+outside+`","expected":"Hi"}]`)
	_, err = LoadManifest(p)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "must be relative")
}

// The bundled corpus must satisfy that contract: every image lives inside it.
func TestShippedCorpusIsSelfContained(t *testing.T) {
	root, err := filepath.Abs("../..")
	require.NoError(t, err)
	corpus := filepath.Join(root, "testdata", "corpus", "synthetic")
	cases, err := LoadManifest(filepath.Join(corpus, ManifestFileName))
	require.NoError(t, err)
	for _, c := range cases {
		rel, err := filepath.Rel(corpus, c.Path(corpus))
		require.NoError(t, err)
		assert.NotContains(t, rel, "..", "case %s reaches outside the corpus", c.Image)
	}
}
