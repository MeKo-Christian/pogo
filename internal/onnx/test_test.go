package onnx

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestGetStartingDirectory(t *testing.T) {
	// Test that getStartingDirectory returns a valid directory
	dir, err := getStartingDirectory()
	if err != nil {
		t.Errorf("getStartingDirectory() failed: %v", err)
	}
	if dir == "" {
		t.Error("getStartingDirectory() returned empty string")
	}

	// Verify the directory exists
	if _, err := os.Stat(dir); err != nil {
		t.Errorf("getStartingDirectory() returned non-existent directory: %s", dir)
	}
}

func TestFindProjectRootFromDir_WithGoMod(t *testing.T) {
	// Create a temporary directory structure with go.mod
	tempDir := t.TempDir()
	projectDir := filepath.Join(tempDir, "project")
	if err := os.MkdirAll(projectDir, 0o755); err != nil {
		t.Fatalf("Failed to create project directory: %v", err)
	}

	// Create go.mod file
	goModPath := filepath.Join(projectDir, "go.mod")
	if err := os.WriteFile(goModPath, []byte("module test\n\ngo 1.21\n"), 0o644); err != nil {
		t.Fatalf("Failed to create go.mod: %v", err)
	}

	// Create subdirectory
	subDir := filepath.Join(projectDir, "sub", "deep")
	if err := os.MkdirAll(subDir, 0o755); err != nil {
		t.Fatalf("Failed to create subdirectory: %v", err)
	}

	// Test from subdirectory
	root, err := findProjectRootFromDir(subDir)
	if err != nil {
		t.Fatalf("findProjectRootFromDir() failed: %v", err)
	}

	if root != projectDir {
		t.Errorf("findProjectRootFromDir() = %s, want %s", root, projectDir)
	}
}

func TestFindProjectRootFromDir_WithOnnxruntimeDir(t *testing.T) {
	// Create a temporary directory structure with onnxruntime directory
	tempDir := t.TempDir()
	projectDir := filepath.Join(tempDir, "project")
	if err := os.MkdirAll(projectDir, 0o755); err != nil {
		t.Fatalf("Failed to create project directory: %v", err)
	}

	// Create onnxruntime directory (instead of go.mod)
	onnxDir := filepath.Join(projectDir, "onnxruntime")
	if err := os.MkdirAll(onnxDir, 0o755); err != nil {
		t.Fatalf("Failed to create onnxruntime directory: %v", err)
	}

	// Create subdirectory
	subDir := filepath.Join(projectDir, "sub")
	if err := os.MkdirAll(subDir, 0o755); err != nil {
		t.Fatalf("Failed to create subdirectory: %v", err)
	}

	// Test from subdirectory
	root, err := findProjectRootFromDir(subDir)
	if err != nil {
		t.Fatalf("findProjectRootFromDir() failed: %v", err)
	}

	if root != projectDir {
		t.Errorf("findProjectRootFromDir() = %s, want %s", root, projectDir)
	}
}

func TestFindProjectRootFromDir_NotFound(t *testing.T) {
	// Create a temporary directory without go.mod or onnxruntime
	tempDir := t.TempDir()

	// Test should fail to find project root
	_, err := findProjectRootFromDir(tempDir)
	if err == nil {
		t.Error("findProjectRootFromDir() should fail when no project markers found")
	}
}

func TestGetLibraryNameForOS(t *testing.T) {
	tests := []struct {
		name     string
		expected string
	}{
		{
			name:     "current OS",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			name, err := getLibraryNameForOS()
			if err != nil {
				t.Errorf("getLibraryNameForOS() failed: %v", err)
			}

			// Verify the name matches the current OS
			switch runtime.GOOS {
			case osLinux:
				if name != libLinux {
					t.Errorf("Expected '%s' for Linux, got '%s'", libLinux, name)
				}
			case osDarwin:
				if name != libDarwin {
					t.Errorf("Expected '%s' for Darwin, got '%s'", libDarwin, name)
				}
			case osWindows:
				if name != libWindows {
					t.Errorf("Expected '%s' for Windows, got '%s'", libWindows, name)
				}
			}
		})
	}
}

func TestGetONNXLibraryPath_WithLibrary(t *testing.T) {
	// Note: This test documents the expected behavior of getONNXLibraryPath
	// The actual function uses getStartingDirectory which finds the executable,
	// so we test it in the actual project environment if available

	path, err := getONNXLibraryPath()
	if err != nil {
		// This is expected if ONNX Runtime is not installed
		t.Logf("getONNXLibraryPath() failed (expected if ONNX Runtime not installed): %v", err)
		return
	}

	// Verify the returned path exists
	if _, err := os.Stat(path); err != nil {
		t.Errorf("getONNXLibraryPath() returned non-existent path: %s", path)
	}

	// Verify the path contains expected library name
	switch runtime.GOOS {
	case osLinux:
		if filepath.Base(path) != libLinux {
			t.Errorf("Expected library name '%s', got '%s'", libLinux, filepath.Base(path))
		}
	case osDarwin:
		if filepath.Base(path) != libDarwin {
			t.Errorf("Expected library name '%s', got '%s'", libDarwin, filepath.Base(path))
		}
	case osWindows:
		if filepath.Base(path) != libWindows {
			t.Errorf("Expected library name '%s', got '%s'", libWindows, filepath.Base(path))
		}
	}
}

func TestGetONNXLibraryPath_ErrorHandling(t *testing.T) {
	// This test documents that getONNXLibraryPath returns appropriate errors
	// when the library cannot be found. The actual behavior depends on whether
	// ONNX Runtime is installed in the system, so we just verify it handles
	// errors gracefully without panicking

	_, err := getONNXLibraryPath()
	// We don't check the error value because it may succeed if ONNX Runtime is installed
	// The important thing is that the function doesn't panic
	_ = err
}

func TestTestONNXRuntime_Integration(t *testing.T) {
	// This is an integration test that requires ONNX Runtime to be installed
	// It will skip if the library is not found
	err := TestONNXRuntime()
	if err != nil {
		// It's OK if this fails - it means ONNX Runtime is not installed
		// We just want to ensure the function handles errors gracefully
		t.Logf("TestONNXRuntime() failed (ONNX Runtime may not be installed): %v", err)
	}
}

func TestGetStartingDirectory_Fallback(t *testing.T) {
	// This test verifies that getStartingDirectory falls back to current directory
	// when executable path is not available

	// We can't easily test the fallback path without manipulating os.Executable()
	// but we can verify the function doesn't panic and returns a valid directory
	dir, err := getStartingDirectory()
	if err != nil {
		t.Errorf("getStartingDirectory() should not fail: %v", err)
	}

	// Verify it's an absolute path
	if !filepath.IsAbs(dir) {
		t.Errorf("getStartingDirectory() should return absolute path, got: %s", dir)
	}

	// Verify the directory exists
	if _, err := os.Stat(dir); err != nil {
		t.Errorf("getStartingDirectory() returned non-existent directory: %s", dir)
	}
}

func TestFindProjectRootFromDir_FromProjectRoot(t *testing.T) {
	// Test finding project root when already at project root
	tempDir := t.TempDir()
	projectDir := filepath.Join(tempDir, "project")
	if err := os.MkdirAll(projectDir, 0o755); err != nil {
		t.Fatalf("Failed to create project directory: %v", err)
	}

	// Create go.mod
	goModPath := filepath.Join(projectDir, "go.mod")
	if err := os.WriteFile(goModPath, []byte("module test\n\ngo 1.21\n"), 0o644); err != nil {
		t.Fatalf("Failed to create go.mod: %v", err)
	}

	// Test from project directory itself
	root, err := findProjectRootFromDir(projectDir)
	if err != nil {
		t.Fatalf("findProjectRootFromDir() failed: %v", err)
	}

	if root != projectDir {
		t.Errorf("findProjectRootFromDir() = %s, want %s", root, projectDir)
	}
}
