package testutil

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
)

// GetProjectRoot returns the project root directory by finding go.mod.
func GetProjectRoot() (string, error) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return "", errors.New("failed to get caller information")
	}
	dir := filepath.Dir(filename)

	// Walk up the directory tree to find go.mod
	for {
		goModPath := filepath.Join(dir, "go.mod")
		if _, err := os.Stat(goModPath); err == nil {
			return dir, nil
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	return "", fmt.Errorf("could not find go.mod file starting from %s", filepath.Dir(filename))
}

// GetTestDataDir returns the path to the testdata directory.
func GetTestDataDir(t *testing.T) string {
	t.Helper()

	root, err := GetProjectRoot()
	require.NoError(t, err, "Failed to find project root")

	return filepath.Join(root, "testdata")
}

// GetTestImageDir returns the path to a specific test image directory.
func GetTestImageDir(t *testing.T, category string) string {
	t.Helper()

	testDataDir := GetTestDataDir(t)
	return filepath.Join(testDataDir, "images", category)
}

// GetSyntheticDir returns the path to the synthetic test data directory.
func GetSyntheticDir(t *testing.T) string {
	t.Helper()

	testDataDir := GetTestDataDir(t)
	return filepath.Join(testDataDir, "synthetic")
}

// GetFixturesDir returns the path to the test fixtures directory.
func GetFixturesDir(t *testing.T) string {
	t.Helper()

	testDataDir := GetTestDataDir(t)
	return filepath.Join(testDataDir, "fixtures")
}

// CreateTempDir creates a temporary directory for testing.
func CreateTempDir(t *testing.T) string {
	t.Helper()

	tempDir := t.TempDir()
	return tempDir
}

// EnsureDir creates a directory if it doesn't exist.
func EnsureDir(path string) error {
	return os.MkdirAll(path, 0o750)
}

// FileExists checks if a file exists.
func FileExists(path string) bool {
	_, err := os.Stat(path)
	return !os.IsNotExist(err)
}

// DirExists checks if a directory exists.
func DirExists(path string) bool {
	info, err := os.Stat(path)
	return !os.IsNotExist(err) && info.IsDir()
}

// GetTestImagePath returns the path to a specific test image file.
func GetTestImagePath(t *testing.T, filename string) string {
	t.Helper()

	testDataDir := GetTestDataDir(t)
	return filepath.Join(testDataDir, "images", filename)
}

// ValidateProjectRoot ensures the directory contains go.mod and required project structure.
func ValidateProjectRoot(root string) error {
	goModPath := filepath.Join(root, "go.mod")
	if !FileExists(goModPath) {
		return fmt.Errorf("go.mod not found at %s", goModPath)
	}

	// Check for key project directories
	requiredDirs := []string{"internal", "cmd"}
	for _, dir := range requiredDirs {
		dirPath := filepath.Join(root, dir)
		if !DirExists(dirPath) {
			return fmt.Errorf("required project directory %s not found at %s", dir, dirPath)
		}
	}

	return nil
}

// GetProjectRootValidated returns the project root with validation.
func GetProjectRootValidated() (string, error) {
	root, err := GetProjectRoot()
	if err != nil {
		return "", err
	}

	if err := ValidateProjectRoot(root); err != nil {
		return "", fmt.Errorf("invalid project root %s: %w", root, err)
	}

	return root, nil
}
