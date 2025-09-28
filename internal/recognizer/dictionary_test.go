package recognizer

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLoadCharset_EmptyPath(t *testing.T) {
	cs, err := LoadCharset("")
	require.Error(t, err)
	require.Nil(t, cs)
}

func TestLoadCharset_FileNotFound(t *testing.T) {
	cs, err := LoadCharset("no/such/file.txt")
	require.Error(t, err)
	require.Nil(t, cs)
}

func TestLoadCharset_Valid(t *testing.T) {
	dir := t.TempDir()
	dictPath := filepath.Join(dir, "dict.txt")

	content := "\xEF\xBB\xBFa\nß\n你\nこんにちは\nspace\n#comment\n\n" //nolint:gosmopolitan // includes BOM and Unicode, plus an ignored empty line
	require.NoError(t, os.WriteFile(dictPath, []byte(content), 0o644))

	cs, err := LoadCharset(dictPath)
	require.NoError(t, err)
	require.NotNil(t, cs)

	// Expect tokens: a, ß, 你, こんにちは, space, #comment (we don't treat '#' specially; user should pre-clean if needed)
	require.Equal(t, 6, cs.Size())
	require.Equal(t, 0, cs.LookupIndex("a"))
	require.Equal(t, 1, cs.LookupIndex("ß"))
	require.Equal(t, 2, cs.LookupIndex("你")) //nolint:gosmopolitan
	require.Equal(t, 3, cs.LookupIndex("こんにちは"))
	require.Equal(t, 4, cs.LookupIndex("space"))
	require.Equal(t, 5, cs.LookupIndex("#comment"))

	require.Equal(t, "ß", cs.LookupToken(1))
	require.Equal(t, "こんにちは", cs.LookupToken(3))
}

func TestLoadCharsets_Merge(t *testing.T) {
	dir := t.TempDir()
	p1 := filepath.Join(dir, "d1.txt")
	p2 := filepath.Join(dir, "d2.txt")
	// Overlapping token 'b'
	require.NoError(t, os.WriteFile(p1, []byte("a\nb\nç\n"), 0o644))
	require.NoError(t, os.WriteFile(p2, []byte("b\nc\n€\n"), 0o644))

	cs, err := LoadCharsets([]string{p1, p2})
	require.NoError(t, err)
	require.NotNil(t, cs)
	require.Equal(t, 5, cs.Size())
	require.Equal(t, 0, cs.LookupIndex("a"))
	require.Equal(t, 1, cs.LookupIndex("b"))
	require.Equal(t, 2, cs.LookupIndex("ç"))
	require.Equal(t, 3, cs.LookupIndex("c"))
	require.Equal(t, 4, cs.LookupIndex("€"))
}
