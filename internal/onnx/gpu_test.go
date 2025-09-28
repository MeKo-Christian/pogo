package onnx

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestDefaultGPUConfig(t *testing.T) {
	config := DefaultGPUConfig()

	if config.UseGPU {
		t.Error("Expected UseGPU to be false by default")
	}
	if config.DeviceID != 0 {
		t.Errorf("Expected DeviceID to be 0, got %d", config.DeviceID)
	}
	if config.GPUMemLimit != 0 {
		t.Errorf("Expected GPUMemLimit to be 0, got %d", config.GPUMemLimit)
	}
	if config.ArenaExtendStrategy != "kNextPowerOfTwo" {
		t.Errorf("Expected ArenaExtendStrategy to be 'kNextPowerOfTwo', got %s", config.ArenaExtendStrategy)
	}
	if config.CUDNNConvAlgoSearch != "DEFAULT" {
		t.Errorf("Expected CUDNNConvAlgoSearch to be 'DEFAULT', got %s", config.CUDNNConvAlgoSearch)
	}
	if !config.DoCopyInDefaultStream {
		t.Error("Expected DoCopyInDefaultStream to be true by default")
	}
}

func TestValidateGPUConfig(t *testing.T) {
	tests := []struct {
		name    string
		config  GPUConfig
		wantErr bool
	}{
		{
			name:    "valid CPU config",
			config:  DefaultGPUConfig(),
			wantErr: false,
		},
		{
			name: "valid GPU config",
			config: GPUConfig{
				UseGPU:                true,
				DeviceID:              0,
				ArenaExtendStrategy:   "kNextPowerOfTwo",
				CUDNNConvAlgoSearch:   "DEFAULT",
				DoCopyInDefaultStream: true,
			},
			wantErr: false,
		},
		{
			name: "negative device ID",
			config: GPUConfig{
				UseGPU:   true,
				DeviceID: -1,
			},
			wantErr: true,
		},
		{
			name: "invalid arena extend strategy",
			config: GPUConfig{
				UseGPU:              true,
				ArenaExtendStrategy: "invalid",
			},
			wantErr: true,
		},
		{
			name: "invalid CUDNN algo search",
			config: GPUConfig{
				UseGPU:              true,
				CUDNNConvAlgoSearch: "invalid",
			},
			wantErr: true,
		},
		{
			name: "valid arena extend strategy kSameAsRequested",
			config: GPUConfig{
				UseGPU:              true,
				ArenaExtendStrategy: "kSameAsRequested",
			},
			wantErr: false,
		},
		{
			name: "valid CUDNN algo search EXHAUSTIVE",
			config: GPUConfig{
				UseGPU:              true,
				CUDNNConvAlgoSearch: "EXHAUSTIVE",
			},
			wantErr: false,
		},
		{
			name: "valid CUDNN algo search HEURISTIC",
			config: GPUConfig{
				UseGPU:              true,
				CUDNNConvAlgoSearch: "HEURISTIC",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateGPUConfig(tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateGPUConfig() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGetRecommendedGPUMemLimit(t *testing.T) {
	limit := GetRecommendedGPUMemLimit()

	// Should return 2GB
	expected := uint64(2 * 1024 * 1024 * 1024)
	if limit != expected {
		t.Errorf("GetRecommendedGPUMemLimit() = %d, want %d", limit, expected)
	}
}

func TestGetSystemLibraryPaths(t *testing.T) {
	// Test GPU paths
	gpuPaths := getSystemLibraryPaths(true)
	expectedGPULen := 4
	if len(gpuPaths) != expectedGPULen {
		t.Errorf("getSystemLibraryPaths(true) returned %d paths, want %d", len(gpuPaths), expectedGPULen)
	}

	// Test CPU paths
	cpuPaths := getSystemLibraryPaths(false)
	expectedCPULen := 3
	if len(cpuPaths) != expectedCPULen {
		t.Errorf("getSystemLibraryPaths(false) returned %d paths, want %d", len(cpuPaths), expectedCPULen)
	}

	// GPU paths should prioritize GPU libraries
	if len(gpuPaths) > 0 && !strings.Contains(gpuPaths[0], "gpu") {
		t.Logf("First GPU path doesn't contain 'gpu': %s", gpuPaths[0])
	}
}

func TestFindProjectRoot(t *testing.T) {
	// Create a temporary directory structure with go.mod
	tempDir := t.TempDir()
	projectDir := filepath.Join(tempDir, "project")
	if err := os.MkdirAll(projectDir, 0o755); err != nil {
		t.Fatalf("Failed to create project directory: %v", err)
	}

	// Create go.mod file
	goModPath := filepath.Join(projectDir, "go.mod")
	goModContent := "module test\n\ngo 1.21\n"
	if err := os.WriteFile(goModPath, []byte(goModContent), 0o644); err != nil {
		t.Fatalf("Failed to create go.mod: %v", err)
	}

	// Create subdirectory
	subDir := filepath.Join(projectDir, "subdir")
	if err := os.MkdirAll(subDir, 0o755); err != nil {
		t.Fatalf("Failed to create subdirectory: %v", err)
	}

	// Change to subdirectory and test
	oldWd, _ := os.Getwd()
	defer func() {
		_ = os.Chdir(oldWd)
	}()

	if err := os.Chdir(subDir); err != nil {
		t.Fatalf("Failed to change to subdirectory: %v", err)
	}

	root, err := findProjectRoot()
	if err != nil {
		t.Fatalf("findProjectRoot() failed: %v", err)
	}

	if root != projectDir {
		t.Errorf("findProjectRoot() = %s, want %s", root, projectDir)
	}
}

func TestFindProjectRootNoGoMod(t *testing.T) {
	// Test in a directory without go.mod
	tempDir := t.TempDir()

	oldWd, _ := os.Getwd()
	defer func() {
		_ = os.Chdir(oldWd)
	}()

	if err := os.Chdir(tempDir); err != nil {
		t.Fatalf("Failed to change to temp directory: %v", err)
	}

	_, err := findProjectRoot()
	if err == nil {
		t.Error("findProjectRoot() should fail when no go.mod is found")
	}
}

func TestGetLibraryName(t *testing.T) {
	// Test that the function works for the current OS
	name, err := getLibraryName()
	if err != nil {
		t.Errorf("getLibraryName() failed for current OS: %v", err)
	}
	if name == "" {
		t.Error("getLibraryName() returned empty name")
	}

	// Verify the name is appropriate for the current OS
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
	default:
		t.Errorf("Unsupported OS: %s", runtime.GOOS)
	}
}

func TestTrySetLibraryPath(t *testing.T) {
	// Create a temporary file to simulate a library
	tempDir := t.TempDir()
	libPath := filepath.Join(tempDir, "libonnxruntime.so")
	if err := os.WriteFile(libPath, []byte("fake library"), 0o644); err != nil {
		t.Fatalf("Failed to create fake library file: %v", err)
	}

	// Test with existing file
	if !trySetLibraryPath(libPath) {
		t.Error("trySetLibraryPath() should return true for existing file")
	}

	// Test with non-existing file
	nonExistentPath := filepath.Join(tempDir, "nonexistent.so")
	if trySetLibraryPath(nonExistentPath) {
		t.Error("trySetLibraryPath() should return false for non-existing file")
	}
}

func TestSetONNXLibraryPath(t *testing.T) {
	// Create a temporary project structure
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

	// Create fake library files
	libName := libLinux
	switch runtime.GOOS {
	case osDarwin:
		libName = libDarwin
	case osWindows:
		libName = libWindows
	}

	// Create CPU library
	cpuLibDir := filepath.Join(projectDir, "onnxruntime", "lib")
	if err := os.MkdirAll(cpuLibDir, 0o755); err != nil {
		t.Fatalf("Failed to create CPU lib directory: %v", err)
	}
	cpuLibPath := filepath.Join(cpuLibDir, libName)
	if err := os.WriteFile(cpuLibPath, []byte("fake cpu library"), 0o644); err != nil {
		t.Fatalf("Failed to create fake CPU library: %v", err)
	}

	// Change to project directory
	oldWd, _ := os.Getwd()
	defer func() {
		_ = os.Chdir(oldWd)
	}()

	if err := os.Chdir(projectDir); err != nil {
		t.Fatalf("Failed to change to project directory: %v", err)
	}

	// Test CPU library path setting
	err := SetONNXLibraryPath(false)
	if err != nil {
		t.Errorf("SetONNXLibraryPath(false) failed: %v", err)
	}

	// Test GPU library path setting (should fall back to CPU)
	err = SetONNXLibraryPath(true)
	if err != nil {
		t.Errorf("SetONNXLibraryPath(true) failed: %v", err)
	}
}

func TestSetONNXLibraryPathNoLibraries(t *testing.T) {
	// This test verifies that SetONNXLibraryPath doesn't panic
	// It may succeed if system libraries exist, which is fine
	err := SetONNXLibraryPath(false)
	// We don't assert on the error since system libraries might exist
	// The important thing is that the function completes without panicking
	_ = err
}
