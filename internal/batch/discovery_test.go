package batch

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDiscoverImageFiles_EmptyArgs(t *testing.T) {
	files, err := discoverImageFiles([]string{}, false, []string{"*.png"}, []string{})
	require.NoError(t, err)
	assert.Empty(t, files)
}

func TestDiscoverImageFiles_SingleFile(t *testing.T) {
	tempDir := testutil.CreateTempDir(t)

	// Create test files
	pngFile := filepath.Join(tempDir, "test.png")
	txtFile := filepath.Join(tempDir, "test.txt")
	jpgFile := filepath.Join(tempDir, "test.jpg")

	require.NoError(t, os.WriteFile(pngFile, []byte("fake png"), 0o600))
	require.NoError(t, os.WriteFile(txtFile, []byte("text file"), 0o600))
	require.NoError(t, os.WriteFile(jpgFile, []byte("fake jpg"), 0o600))

	files, err := discoverImageFiles([]string{pngFile, jpgFile}, false, []string{"*.png", "*.jpg"}, []string{})
	require.NoError(t, err)
	assert.Len(t, files, 2)
	assert.Contains(t, files, pngFile)
	assert.Contains(t, files, jpgFile)
}

func TestDiscoverImageFiles_Directory(t *testing.T) {
	tempDir := testutil.CreateTempDir(t)

	// Create test files in directory
	pngFile := filepath.Join(tempDir, "image.png")
	jpgFile := filepath.Join(tempDir, "photo.jpg")
	txtFile := filepath.Join(tempDir, "notes.txt")

	require.NoError(t, os.WriteFile(pngFile, []byte("fake png"), 0o600))
	require.NoError(t, os.WriteFile(jpgFile, []byte("fake jpg"), 0o600))
	require.NoError(t, os.WriteFile(txtFile, []byte("text"), 0o600))

	files, err := discoverImageFiles([]string{tempDir}, false, []string{"*.png", "*.jpg"}, []string{})
	require.NoError(t, err)
	assert.Len(t, files, 2)
	assert.Contains(t, files, pngFile)
	assert.Contains(t, files, jpgFile)
}

func TestDiscoverImageFiles_Recursive(t *testing.T) {
	tempDir := testutil.CreateTempDir(t)

	// Create nested directory structure
	subDir := filepath.Join(tempDir, "subdir")
	require.NoError(t, os.MkdirAll(subDir, 0o750))

	// Create test files
	rootPng := filepath.Join(tempDir, "root.png")
	subPng := filepath.Join(subDir, "sub.png")
	subTxt := filepath.Join(subDir, "sub.txt")

	require.NoError(t, os.WriteFile(rootPng, []byte("root png"), 0o600))
	require.NoError(t, os.WriteFile(subPng, []byte("sub png"), 0o600))
	require.NoError(t, os.WriteFile(subTxt, []byte("sub txt"), 0o600))

	files, err := discoverImageFiles([]string{tempDir}, true, []string{"*.png"}, []string{})
	require.NoError(t, err)
	assert.Len(t, files, 2)
	assert.Contains(t, files, rootPng)
	assert.Contains(t, files, subPng)
}

func TestDiscoverImageFiles_NonRecursive(t *testing.T) {
	tempDir := testutil.CreateTempDir(t)

	// Create nested directory structure
	subDir := filepath.Join(tempDir, "subdir")
	require.NoError(t, os.MkdirAll(subDir, 0o750))

	// Create test files
	rootPng := filepath.Join(tempDir, "root.png")
	subPng := filepath.Join(subDir, "sub.png")

	require.NoError(t, os.WriteFile(rootPng, []byte("root png"), 0o600))
	require.NoError(t, os.WriteFile(subPng, []byte("sub png"), 0o600))

	files, err := discoverImageFiles([]string{tempDir}, false, []string{"*.png"}, []string{})
	require.NoError(t, err)
	assert.Len(t, files, 1)
	assert.Contains(t, files, rootPng)
	assert.NotContains(t, files, subPng)
}

func TestDiscoverImageFiles_IncludeExcludePatterns(t *testing.T) {
	tempDir := testutil.CreateTempDir(t)

	// Create test files
	test1Png := filepath.Join(tempDir, "test1.png")
	test2Png := filepath.Join(tempDir, "test2.png")
	excludePng := filepath.Join(tempDir, "exclude.png")

	require.NoError(t, os.WriteFile(test1Png, []byte("test1"), 0o600))
	require.NoError(t, os.WriteFile(test2Png, []byte("test2"), 0o600))
	require.NoError(t, os.WriteFile(excludePng, []byte("exclude"), 0o600))

	files, err := discoverImageFiles([]string{tempDir}, false, []string{"*.png"}, []string{"*exclude*"})
	require.NoError(t, err)
	assert.Len(t, files, 2)
	assert.Contains(t, files, test1Png)
	assert.Contains(t, files, test2Png)
	assert.NotContains(t, files, excludePng)
}

func TestDiscoverImageFiles_NonExistentDirectory(t *testing.T) {
	files, err := discoverImageFiles([]string{"/nonexistent/directory"}, false, []string{"*.png"}, []string{})
	require.Error(t, err)
	assert.Nil(t, files)
	assert.Contains(t, err.Error(), "cannot access")
}

func TestDiscoverInDirectory_EmptyDirectory(t *testing.T) {
	tempDir := testutil.CreateTempDir(t)

	files, err := discoverInDirectory(tempDir, false, []string{"*.png"}, []string{})
	require.NoError(t, err)
	assert.Empty(t, files)
}

func TestDiscoverInDirectory_WithFiles(t *testing.T) {
	tempDir := testutil.CreateTempDir(t)

	// Create test files
	pngFile := filepath.Join(tempDir, "test.png")
	jpgFile := filepath.Join(tempDir, "test.jpg")
	txtFile := filepath.Join(tempDir, "test.txt")

	require.NoError(t, os.WriteFile(pngFile, []byte("png"), 0o600))
	require.NoError(t, os.WriteFile(jpgFile, []byte("jpg"), 0o600))
	require.NoError(t, os.WriteFile(txtFile, []byte("txt"), 0o600))

	files, err := discoverInDirectory(tempDir, false, []string{"*.png", "*.jpg"}, []string{})
	require.NoError(t, err)
	assert.Len(t, files, 2)
	assert.Contains(t, files, pngFile)
	assert.Contains(t, files, jpgFile)
}

func TestMatchesPatterns_EmptyPatterns(t *testing.T) {
	assert.False(t, matchesAnyPattern("test.png", []string{}))
}

func TestMatchesPatterns_SinglePattern(t *testing.T) {
	testCases := []struct {
		filename string
		pattern  string
		expected bool
	}{
		{"test.png", "*.png", true},
		{"test.jpg", "*.png", false},
		{"photo.png", "*.png", true},
		{"test.PNG", "*.png", false}, // Case sensitive
		{"test.png", "test.*", true},
		{"other.png", "test.*", false},
	}

	for _, tc := range testCases {
		result := matchesAnyPattern(tc.filename, []string{tc.pattern})
		assert.Equal(t, tc.expected, result, "filename=%s, pattern=%s", tc.filename, tc.pattern)
	}
}

func TestMatchesPatterns_MultiplePatterns(t *testing.T) {
	patterns := []string{"*.png", "*.jpg", "special.*"}

	testCases := []struct {
		filename string
		expected bool
	}{
		{"test.png", true},
		{"photo.jpg", true},
		{"special.gif", true},
		{"document.pdf", false},
		{"image.bmp", false},
	}

	for _, tc := range testCases {
		result := matchesAnyPattern(tc.filename, patterns)
		assert.Equal(t, tc.expected, result, "filename=%s", tc.filename)
	}
}

func TestMatchesPatterns_WithExclude(t *testing.T) {
	// Test that matchesAnyPattern only checks patterns, not exclusion
	// (exclusion is handled at a higher level in shouldIncludeFile)
	assert.True(t, matchesAnyPattern("exclude.png", []string{"*.png"}))
	assert.False(t, matchesAnyPattern("exclude.png", []string{"*.jpg"}))
}
