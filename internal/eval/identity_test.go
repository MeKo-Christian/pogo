package eval

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func writeModels(t *testing.T, dir string, contents map[string]string) []string {
	t.Helper()
	paths := make([]string, 0, len(contents))
	for name, body := range contents {
		p := filepath.Join(dir, name)
		require.NoError(t, os.WriteFile(p, []byte(body), 0o600))
		paths = append(paths, p)
	}
	return paths
}

const (
	variantMobile = "mobile"
	someHash      = "aaaa"
)

var bundle = map[string]string{
	"det.onnx": "detector weights",
	"rec.onnx": "recognizer weights",
	"dict.txt": "a\nb\nc\n",
}

// A baseline is committed and has to match on any checkout, so the directory the
// weights happen to live in must not enter the identity.
func TestFingerprintIgnoresTheDirectory(t *testing.T) {
	a, err := Fingerprint(variantMobile, writeModels(t, t.TempDir(), bundle))
	require.NoError(t, err)
	b, err := Fingerprint(variantMobile, writeModels(t, t.TempDir(), bundle))
	require.NoError(t, err)
	assert.Equal(t, a, b, "the same weights in another checkout must be the same identity")
	assert.True(t, a.Matches(b))
}

// Different bytes are a different measurement, wherever they live.
func TestFingerprintFollowsContent(t *testing.T) {
	base, err := Fingerprint(variantMobile, writeModels(t, t.TempDir(), bundle))
	require.NoError(t, err)

	changed := map[string]string{}
	for k, v := range bundle {
		changed[k] = v
	}
	changed["dict.txt"] = "a\nb\nc\nd\n"
	other, err := Fingerprint(variantMobile, writeModels(t, t.TempDir(), changed))
	require.NoError(t, err)

	assert.NotEqual(t, base.Fingerprint, other.Fingerprint)
	assert.False(t, base.Matches(other))
}

// The file name is part of the identity, so swapping which dictionary is loaded
// is visible even when some other file has the same bytes.
func TestFingerprintFollowsFileNames(t *testing.T) {
	a, err := Fingerprint(variantMobile, writeModels(t, t.TempDir(), map[string]string{"dict.txt": "x"}))
	require.NoError(t, err)
	b, err := Fingerprint(variantMobile, writeModels(t, t.TempDir(), map[string]string{"other.txt": "x"}))
	require.NoError(t, err)
	assert.NotEqual(t, a.Fingerprint, b.Fingerprint)
}

// The order the pipeline config happens to list artifacts in is not information.
func TestFingerprintIsOrderIndependent(t *testing.T) {
	dir := t.TempDir()
	paths := writeModels(t, dir, bundle)
	a, err := Fingerprint(variantMobile, paths)
	require.NoError(t, err)

	reversed := make([]string, 0, len(paths))
	for i := len(paths) - 1; i >= 0; i-- {
		reversed = append(reversed, paths[i])
	}
	b, err := Fingerprint(variantMobile, reversed)
	require.NoError(t, err)
	assert.Equal(t, a, b)
}

func TestFingerprintReportsAMissingArtifact(t *testing.T) {
	_, err := Fingerprint(variantMobile, []string{filepath.Join(t.TempDir(), "absent.onnx")})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "fingerprint")
}

// The variant is a label; it never makes two different weight sets comparable.
func TestMatchesRequiresBoth(t *testing.T) {
	a := ModelIdentity{Variant: variantMobile, Fingerprint: someHash}
	assert.False(t, a.Matches(ModelIdentity{Variant: variantMobile, Fingerprint: "bbbb"}))
	assert.False(t, a.Matches(ModelIdentity{Variant: "server", Fingerprint: someHash}))
	assert.True(t, a.Matches(ModelIdentity{Variant: variantMobile, Fingerprint: someHash}))
}
