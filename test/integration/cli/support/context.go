package support

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/MeKo-Tech/pogo/internal/testutil"
)

// TestContext holds the state for integration tests.
type TestContext struct {
	// Command execution state
	LastCommand    string
	LastOutput     string
	LastError      error
	LastExitCode   int
	LastStartTime  time.Time
	LastDuration   time.Duration
	LastOutputFile string

	// Test environment
	WorkingDir    string
	TempDir       string
	TempModelsDir string
	EnvVars       []string

	// Server management
	ServerProcess  *os.Process
	ServerPort     int
	ServerHost     string
	HTTPTestServer *HTTPTestServerWrapper

	// HTTP response state
	LastHTTPStatusCode int
	LastHTTPResponse   string
	LastHTTPHeaders    map[string]string

	// Server configuration state
	LastCORSOrigin    string
	LastMaxUploadSize int
	LastTimeout       int

	// Test artifacts
	CreatedFiles       []string
	CreatedDirectories []string
}

// StopServer stops the running server (placeholder implementation).
func (testCtx *TestContext) StopServer() error {
	// Stop httptest server if running
	if testCtx.HTTPTestServer != nil {
		return testCtx.stopTestHTTPServer()
	}

	// Stop process-based server if running
	if testCtx.ServerProcess != nil {
		if err := testCtx.ServerProcess.Kill(); err != nil {
			return fmt.Errorf("failed to kill server process: %w", err)
		}
		testCtx.ServerProcess = nil
	}
	return nil
}

// NewTestContext creates a new test context.
func NewTestContext() (*TestContext, error) {
	// Get current working directory
	workingDir, err := os.Getwd()
	if err != nil {
		return nil, fmt.Errorf("failed to get working directory: %w", err)
	}

	// If we're in a subdirectory (test execution might cd), find project root
	// Look for go.mod file to identify project root
	currentDir := workingDir
	for {
		if _, err := os.Stat(filepath.Join(currentDir, "go.mod")); err == nil {
			workingDir = currentDir
			break
		}
		parentDir := filepath.Dir(currentDir)
		if parentDir == currentDir {
			// Reached filesystem root, use current directory
			break
		}
		currentDir = parentDir
	}

	// Create temporary directory for test artifacts
	tempDir, err := os.MkdirTemp("", "pogo-test-*")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp directory: %w", err)
	}

	ctx := &TestContext{
		WorkingDir:         workingDir,
		TempDir:            tempDir,
		EnvVars:            []string{},
		CreatedFiles:       []string{},
		CreatedDirectories: []string{},
		ServerPort:         8080,
		ServerHost:         "localhost",
	}

	return ctx, nil
}

// Cleanup removes all temporary files and directories created during tests.
func (testCtx *TestContext) Cleanup() error {
	var errors []error

	// Stop server if running
	if testCtx.ServerProcess != nil {
		if err := testCtx.StopServer(); err != nil {
			errors = append(errors, fmt.Errorf("failed to stop server: %w", err))
		}
	}

	// Remove created files
	for _, file := range testCtx.CreatedFiles {
		if err := os.Remove(file); err != nil && !os.IsNotExist(err) {
			errors = append(errors, fmt.Errorf("failed to remove file %s: %w", file, err))
		}
	}

	// Remove created directories
	for _, dir := range testCtx.CreatedDirectories {
		if err := os.RemoveAll(dir); err != nil && !os.IsNotExist(err) {
			errors = append(errors, fmt.Errorf("failed to remove directory %s: %w", dir, err))
		}
	}

	// Remove temp directory
	if err := os.RemoveAll(testCtx.TempDir); err != nil && !os.IsNotExist(err) {
		errors = append(errors, fmt.Errorf("failed to remove temp directory %s: %w", testCtx.TempDir, err))
	}

	if len(errors) > 0 {
		return fmt.Errorf("cleanup errors: %v", errors)
	}

	return nil
}

// AddEnvVar adds an environment variable for command execution.
func (testCtx *TestContext) AddEnvVar(name, value string) {
	testCtx.EnvVars = append(testCtx.EnvVars, fmt.Sprintf("%s=%s", name, value))
}

// TrackFile adds a file to be cleaned up after tests.
func (testCtx *TestContext) TrackFile(filename string) {
	absPath := filename
	if !filepath.IsAbs(filename) {
		absPath = filepath.Join(testCtx.WorkingDir, filename)
	}
	testCtx.CreatedFiles = append(testCtx.CreatedFiles, absPath)
}

// TrackDirectory adds a directory to be cleaned up after tests.
func (testCtx *TestContext) TrackDirectory(dirname string) {
	absPath := dirname
	if !filepath.IsAbs(dirname) {
		absPath = filepath.Join(testCtx.WorkingDir, dirname)
	}
	testCtx.CreatedDirectories = append(testCtx.CreatedDirectories, absPath)
}

// GetTempFile returns a path to a temporary file.
func (testCtx *TestContext) GetTempFile(suffix string) string {
	return filepath.Join(testCtx.TempDir, fmt.Sprintf("test-%d%s", time.Now().UnixNano(), suffix))
}

// GetTempDir returns a path to a temporary directory.
func (testCtx *TestContext) GetTempDir(prefix string) string {
	dirPath := filepath.Join(testCtx.TempDir, fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano()))
	testCtx.TrackDirectory(dirPath)
	return dirPath
}

// getTestImagePath returns the absolute path to a test image file.
func (testCtx *TestContext) getTestImagePath(filename string) (string, error) {
	projectRoot, err := testutil.GetProjectRoot()
	if err != nil {
		return "", fmt.Errorf("failed to find project root: %w", err)
	}

	imagePath := filepath.Join(projectRoot, "testdata", "images", filename)

	// Check if the image exists
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		return "", fmt.Errorf("test image not found: %s", imagePath)
	}

	return imagePath, nil
}
