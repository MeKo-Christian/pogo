package testutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetProjectRoot(t *testing.T) {
	root, err := GetProjectRoot()
	require.NoError(t, err)
	assert.NotEmpty(t, root)
	assert.True(t, FileExists(root+"/go.mod"))
}

func TestGetTestDataDir(t *testing.T) {
	testDataDir := GetTestDataDir(t)
	assert.NotEmpty(t, testDataDir)
	assert.Contains(t, testDataDir, "testdata")
}

func TestGetTestImageDir(t *testing.T) {
	simpleDir := GetTestImageDir(t, "simple")
	assert.Contains(t, simpleDir, "testdata/images/simple")

	multilineDir := GetTestImageDir(t, "multiline")
	assert.Contains(t, multilineDir, "testdata/images/multiline")
}

func TestEnsureDir(t *testing.T) {
	tempDir := CreateTempDir(t)
	testDir := tempDir + "/test/nested/dir"

	err := EnsureDir(testDir)
	require.NoError(t, err)
	assert.True(t, DirExists(testDir))
}

func TestFileExists(t *testing.T) {
	// Test with non-existent file
	assert.False(t, FileExists("/non/existent/file"))

	// Test with existing file (go.mod in project root)
	root, err := GetProjectRoot()
	require.NoError(t, err)
	assert.True(t, FileExists(root+"/go.mod"))
}
