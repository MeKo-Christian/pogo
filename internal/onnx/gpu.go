package onnx

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"

	"github.com/yalue/onnxruntime_go"
)

const (
	osLinux    = "linux"
	osDarwin   = "darwin"
	osWindows  = "windows"
	libLinux   = "libonnxruntime.so"
	libDarwin  = "libonnxruntime.dylib"
	libWindows = "onnxruntime.dll"
)

// GPUConfig holds configuration for GPU acceleration using CUDA.
type GPUConfig struct {
	UseGPU                bool   // Enable GPU acceleration
	DeviceID              int    // CUDA device ID (default: 0)
	GPUMemLimit           uint64 // GPU memory limit in bytes (0 = unlimited)
	ArenaExtendStrategy   string // "kNextPowerOfTwo" or "kSameAsRequested" (default: "kNextPowerOfTwo")
	CUDNNConvAlgoSearch   string // "EXHAUSTIVE", "HEURISTIC", or "DEFAULT" (default: "DEFAULT")
	DoCopyInDefaultStream bool   // Use default stream for copy operations (default: true)
}

// DefaultGPUConfig returns default GPU configuration.
func DefaultGPUConfig() GPUConfig {
	return GPUConfig{
		UseGPU:                false,
		DeviceID:              0,
		GPUMemLimit:           0, // Unlimited
		ArenaExtendStrategy:   "kNextPowerOfTwo",
		CUDNNConvAlgoSearch:   "DEFAULT",
		DoCopyInDefaultStream: true,
	}
}

// ConfigureSessionForGPU configures an ONNX Runtime session to use GPU acceleration.
// If GPU configuration fails, it gracefully falls back to CPU-only execution.
func ConfigureSessionForGPU(sessionOptions *onnxruntime_go.SessionOptions, gpuConfig GPUConfig) error {
	if !gpuConfig.UseGPU {
		// GPU not requested, use CPU only
		return nil
	}

	// Create CUDA provider options
	cudaOpts, err := onnxruntime_go.NewCUDAProviderOptions()
	if err != nil {
		return fmt.Errorf("failed to create CUDA provider options (GPU may not be available): %w", err)
	}
	defer func() {
		if destroyErr := cudaOpts.Destroy(); destroyErr != nil {
			// Log but don't fail on cleanup error
			fmt.Printf("Warning: failed to destroy CUDA provider options: %v\n", destroyErr)
		}
	}()

	// Configure CUDA options
	cudaSettings := make(map[string]string)
	cudaSettings["device_id"] = strconv.Itoa(gpuConfig.DeviceID)

	if gpuConfig.GPUMemLimit > 0 {
		cudaSettings["gpu_mem_limit"] = strconv.FormatUint(gpuConfig.GPUMemLimit, 10)
	}

	if gpuConfig.ArenaExtendStrategy != "" {
		cudaSettings["arena_extend_strategy"] = gpuConfig.ArenaExtendStrategy
	}

	if gpuConfig.CUDNNConvAlgoSearch != "" {
		cudaSettings["cudnn_conv_algo_search"] = gpuConfig.CUDNNConvAlgoSearch
	}

	if gpuConfig.DoCopyInDefaultStream {
		cudaSettings["do_copy_in_default_stream"] = "1"
	} else {
		cudaSettings["do_copy_in_default_stream"] = "0"
	}

	// Update CUDA provider with settings
	if err := cudaOpts.Update(cudaSettings); err != nil {
		return fmt.Errorf("failed to update CUDA provider options: %w", err)
	}

	// Append CUDA execution provider (will be tried first)
	if err := sessionOptions.AppendExecutionProviderCUDA(cudaOpts); err != nil {
		return fmt.Errorf("failed to append CUDA execution provider: %w", err)
	}

	return nil
}

// GetRecommendedGPUMemLimit returns a recommended GPU memory limit based on available GPU memory.
// This is typically 80% of available GPU memory to leave room for other processes.
func GetRecommendedGPUMemLimit() uint64 {
	// For now, return a conservative 2GB limit
	// In the future, this could query actual GPU memory via nvidia-ml-py or similar
	return 2 * 1024 * 1024 * 1024 // 2GB
}

// ValidateGPUConfig checks if the GPU configuration is valid.
func ValidateGPUConfig(config GPUConfig) error {
	if !config.UseGPU {
		return nil // Nothing to validate for CPU-only
	}

	if config.DeviceID < 0 {
		return fmt.Errorf("device ID must be non-negative, got %d", config.DeviceID)
	}

	validStrategies := map[string]bool{
		"kNextPowerOfTwo":  true,
		"kSameAsRequested": true,
	}
	if config.ArenaExtendStrategy != "" && !validStrategies[config.ArenaExtendStrategy] {
		return fmt.Errorf("invalid arena extend strategy: %s (must be 'kNextPowerOfTwo' or "+
			"'kSameAsRequested')", config.ArenaExtendStrategy)
	}

	validAlgoSearch := map[string]bool{
		"EXHAUSTIVE": true,
		"HEURISTIC":  true,
		"DEFAULT":    true,
	}
	if config.CUDNNConvAlgoSearch != "" && !validAlgoSearch[config.CUDNNConvAlgoSearch] {
		return fmt.Errorf("invalid CUDNN conv algo search: %s (must be 'EXHAUSTIVE', 'HEURISTIC', or "+
			"'DEFAULT')", config.CUDNNConvAlgoSearch)
	}

	return nil
}

// getSystemLibraryPaths returns system library paths to try, prioritizing GPU or CPU based on useGPU.
func getSystemLibraryPaths(useGPU bool) []string {
	if useGPU {
		return []string{
			"/opt/onnxruntime/gpu/lib/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/libonnxruntime.so",
			"/opt/onnxruntime/cpu/lib/libonnxruntime.so",
		}
	}
	return []string{
		"/usr/local/lib/libonnxruntime.so",
		"/usr/lib/libonnxruntime.so",
		"/opt/onnxruntime/cpu/lib/libonnxruntime.so",
	}
}

// findProjectRoot finds the project root directory by looking for go.mod.
func findProjectRoot() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get current directory: %w", err)
	}

	projectRoot := cwd
	for {
		if _, err := os.Stat(filepath.Join(projectRoot, "go.mod")); err == nil {
			return projectRoot, nil
		}
		parent := filepath.Dir(projectRoot)
		if parent == projectRoot {
			return "", errors.New("could not find project root")
		}
		projectRoot = parent
	}
}

// getLibraryName returns the appropriate library filename for the current OS.
func getLibraryName() (string, error) {
	switch runtime.GOOS {
	case osLinux:
		return libLinux, nil
	case osDarwin:
		return libDarwin, nil
	case osWindows:
		return libWindows, nil
	default:
		return "", fmt.Errorf("unsupported operating system: %s", runtime.GOOS)
	}
}

// trySetLibraryPath attempts to set the ONNX library path if the file exists.
func trySetLibraryPath(path string) bool {
	if _, err := os.Stat(path); err == nil {
		onnxruntime_go.SetSharedLibraryPath(path)
		return true
	}
	return false
}

// SetONNXLibraryPath sets the path to the ONNX Runtime shared library.
// If useGPU is true, it prioritizes GPU libraries.
func SetONNXLibraryPath(useGPU bool) error {
	// Try system paths first
	systemPaths := getSystemLibraryPaths(useGPU)
	for _, path := range systemPaths {
		if trySetLibraryPath(path) {
			return nil
		}
	}

	// Try project-relative path
	projectRoot, err := findProjectRoot()
	if err != nil {
		return err
	}

	libName, err := getLibraryName()
	if err != nil {
		return err
	}

	// Try GPU library first if requested
	if useGPU {
		gpuLibPath := filepath.Join(projectRoot, "onnxruntime", "gpu", "lib", libName)
		if trySetLibraryPath(gpuLibPath) {
			return nil
		}
	}

	// Fallback to CPU library
	libPath := filepath.Join(projectRoot, "onnxruntime", "lib", libName)
	if !trySetLibraryPath(libPath) {
		return fmt.Errorf("ONNX Runtime library not found at %s", libPath)
	}

	return nil
}
