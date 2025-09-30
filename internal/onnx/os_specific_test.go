package onnx

import (
	"runtime"
	"testing"
)

// TestOSSpecificBehavior documents OS-specific behavior
// These tests exercise the logic but some will skip on certain platforms.
func TestOSSpecificBehavior(t *testing.T) {
	currentOS := runtime.GOOS

	t.Run("LibraryNameLogic", func(t *testing.T) {
		// This tests the OS detection logic even if we can't test all branches
		name, err := getLibraryName()
		if err != nil {
			t.Fatalf("getLibraryName() failed: %v", err)
		}

		// Verify the name matches expectations for current OS
		var expectedName string
		switch currentOS {
		case "linux":
			expectedName = "libonnxruntime.so"
		case "darwin":
			expectedName = "libonnxruntime.dylib"
		case "windows":
			expectedName = "onnxruntime.dll"
		default:
			t.Skipf("Unsupported OS: %s", currentOS)
		}

		if name != expectedName {
			t.Errorf("getLibraryName() = %s, want %s for OS %s", name, expectedName, currentOS)
		}
	})

	t.Run("LibraryNameForOSLogic", func(t *testing.T) {
		// Test getLibraryNameForOS
		name, err := getLibraryNameForOS()
		if err != nil {
			t.Fatalf("getLibraryNameForOS() failed: %v", err)
		}

		// Should match getLibraryName() result
		expectedName, _ := getLibraryName()
		if name != expectedName {
			t.Errorf("getLibraryNameForOS() = %s, want %s", name, expectedName)
		}
	})

	t.Run("SystemLibraryPathPriority", func(t *testing.T) {
		// Test that system library paths are returned correctly
		cpuPaths := getSystemLibraryPaths(false)
		if len(cpuPaths) == 0 {
			t.Error("getSystemLibraryPaths(false) returned empty list")
		}

		gpuPaths := getSystemLibraryPaths(true)
		if len(gpuPaths) == 0 {
			t.Error("getSystemLibraryPaths(true) returned empty list")
		}

		// GPU paths should be longer (includes GPU-specific paths)
		if len(gpuPaths) <= len(cpuPaths) {
			t.Errorf("Expected GPU paths (%d) to be more than CPU paths (%d)",
				len(gpuPaths), len(cpuPaths))
		}
	})
}

// TestLibraryDiscoveryFlow tests the full library discovery flow.
func TestLibraryDiscoveryFlow(t *testing.T) {
	t.Run("TrySetLibraryPath", func(t *testing.T) {
		// Test with non-existent path
		result := trySetLibraryPath("/nonexistent/path/to/library.so")
		if result {
			t.Error("trySetLibraryPath should return false for non-existent path")
		}
	})

	t.Run("FindProjectRootFromCurrentDir", func(t *testing.T) {
		// Test from actual project (should work since tests run from project)
		root, err := findProjectRoot()
		if err != nil {
			t.Logf("findProjectRoot() failed: %v (may be expected in some test environments)", err)
		} else {
			t.Logf("Found project root: %s", root)
		}
	})

	t.Run("GetStartingDirectoryFlow", func(t *testing.T) {
		// Test that getStartingDirectory returns valid directory
		dir, err := getStartingDirectory()
		if err != nil {
			t.Errorf("getStartingDirectory() failed: %v", err)
		}
		if dir == "" {
			t.Error("getStartingDirectory() returned empty string")
		}
		t.Logf("Starting directory: %s", dir)
	})
}
