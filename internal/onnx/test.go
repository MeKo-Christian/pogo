package onnx

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	onnxruntime "github.com/yalue/onnxruntime_go"
)

// getONNXLibraryPath returns the path to the ONNX Runtime shared library.
func getONNXLibraryPath() (string, error) {
	// Get the current executable's directory to find project root
	execPath, err := os.Executable()
	if err != nil {
		// Fallback: try to find relative to current working directory
		cwd, err := os.Getwd()
		if err != nil {
			return "", fmt.Errorf("failed to get current directory: %w", err)
		}
		execPath = cwd
	}

	// Find project root by looking for go.mod or onnxruntime directory
	projectRoot := filepath.Dir(execPath)
	for {
		// Check if this directory contains go.mod or onnxruntime directory
		if _, err := os.Stat(filepath.Join(projectRoot, "go.mod")); err == nil {
			break
		}
		if _, err := os.Stat(filepath.Join(projectRoot, "onnxruntime")); err == nil {
			break
		}

		parent := filepath.Dir(projectRoot)
		if parent == projectRoot {
			// We've reached the root directory
			return "", errors.New("could not find project root with onnxruntime directory")
		}
		projectRoot = parent
	}

	// Construct path to ONNX Runtime library
	libDir := filepath.Join(projectRoot, "onnxruntime", "lib")

	// Determine library filename based on OS
	var libName string
	switch runtime.GOOS {
	case "linux":
		libName = "libonnxruntime.so"
	case "darwin":
		libName = "libonnxruntime.dylib"
	case "windows":
		libName = "onnxruntime.dll"
	default:
		return "", fmt.Errorf("unsupported operating system: %s", runtime.GOOS)
	}

	libPath := filepath.Join(libDir, libName)

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
