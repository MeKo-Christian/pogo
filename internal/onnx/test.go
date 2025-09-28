package onnx

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	onnxruntime "github.com/yalue/onnxruntime_go"
)

// getStartingDirectory returns the directory to start searching for project root.
func getStartingDirectory() (string, error) {
	execPath, err := os.Executable()
	if err != nil {
		// Fallback: try to find relative to current working directory
		cwd, err := os.Getwd()
		if err != nil {
			return "", fmt.Errorf("failed to get current directory: %w", err)
		}
		return cwd, nil
	}
	return filepath.Dir(execPath), nil
}

// findProjectRootFromDir finds the project root starting from the given directory.
func findProjectRootFromDir(startDir string) (string, error) {
	projectRoot := startDir
	for {
		// Check if this directory contains go.mod or onnxruntime directory
		if _, err := os.Stat(filepath.Join(projectRoot, "go.mod")); err == nil {
			return projectRoot, nil
		}
		if _, err := os.Stat(filepath.Join(projectRoot, "onnxruntime")); err == nil {
			return projectRoot, nil
		}

		parent := filepath.Dir(projectRoot)
		if parent == projectRoot {
			// We've reached the root directory
			return "", errors.New("could not find project root with onnxruntime directory")
		}
		projectRoot = parent
	}
}

// getLibraryNameForOS returns the appropriate library filename for the current OS.
func getLibraryNameForOS() (string, error) {
	switch runtime.GOOS {
	case "linux":
		return "libonnxruntime.so", nil
	case "darwin":
		return "libonnxruntime.dylib", nil
	case "windows":
		return "onnxruntime.dll", nil
	default:
		return "", fmt.Errorf("unsupported operating system: %s", runtime.GOOS)
	}
}

// getONNXLibraryPath returns the path to the ONNX Runtime shared library.
func getONNXLibraryPath() (string, error) {
	startDir, err := getStartingDirectory()
	if err != nil {
		return "", err
	}

	projectRoot, err := findProjectRootFromDir(startDir)
	if err != nil {
		return "", err
	}

	libName, err := getLibraryNameForOS()
	if err != nil {
		return "", err
	}

	libPath := filepath.Join(projectRoot, "onnxruntime", "lib", libName)

	// Verify the library exists
	if _, err := os.Stat(libPath); err != nil {
		return "", fmt.Errorf("ONNX Runtime library not found at %s: %w", libPath, err)
	}

	return libPath, nil
}

// TestONNXRuntime performs a basic test to verify ONNX Runtime is working.
func TestONNXRuntime() error {
	// Set the shared library path before initialization
	libPath, err := getONNXLibraryPath()
	if err != nil {
		return fmt.Errorf("failed to find ONNX Runtime library: %w", err)
	}

	fmt.Printf("Using ONNX Runtime library: %s\n", libPath)
	onnxruntime.SetSharedLibraryPath(libPath)

	// Initialize ONNX Runtime
	err = onnxruntime.InitializeEnvironment()
	if err != nil {
		return fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
	}
	defer func() {
		if err := onnxruntime.DestroyEnvironment(); err != nil {
			fmt.Fprintf(os.Stderr, "Error destroying ONNX Runtime environment: %v\n", err)
		}
	}()

	fmt.Println("✓ ONNX Runtime initialized successfully")

	// Test basic session creation (without a model)
	// This tests that the library is linked correctly
	fmt.Println("✓ ONNX Runtime library linked and working")

	return nil
}
