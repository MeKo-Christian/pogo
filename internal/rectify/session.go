package rectify

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	onnxrt "github.com/yalue/onnxruntime_go"
)

// createONNXSession creates and initializes an ONNX session for the given config.
func createONNXSession(cfg Config) (
	*onnxrt.DynamicAdvancedSession,
	onnxrt.InputOutputInfo,
	onnxrt.InputOutputInfo,
	error,
) {
	if err := setONNXLibraryPath(); err != nil {
		return nil, onnxrt.InputOutputInfo{}, onnxrt.InputOutputInfo{}, err
	}

	if !onnxrt.IsInitialized() {
		if err := onnxrt.InitializeEnvironment(); err != nil {
			return nil, onnxrt.InputOutputInfo{}, onnxrt.InputOutputInfo{}, fmt.Errorf("init onnx: %w", err)
		}
	}

	inputs, outputs, err := onnxrt.GetInputOutputInfo(cfg.ModelPath)
	if err != nil {
		return nil, onnxrt.InputOutputInfo{}, onnxrt.InputOutputInfo{}, fmt.Errorf("io info: %w", err)
	}

	if len(inputs) != 1 || len(outputs) != 1 {
		return nil, onnxrt.InputOutputInfo{}, onnxrt.InputOutputInfo{},
			fmt.Errorf("unexpected io (in:%d out:%d)", len(inputs), len(outputs))
	}

	in := inputs[0]
	out := outputs[0]

	opts, err := onnxrt.NewSessionOptions()
	if err != nil {
		return nil, onnxrt.InputOutputInfo{}, onnxrt.InputOutputInfo{}, fmt.Errorf("session opts: %w", err)
	}
	defer func() { _ = opts.Destroy() }()

	if cfg.NumThreads > 0 {
		_ = opts.SetIntraOpNumThreads(cfg.NumThreads)
	}

	sess, err := onnxrt.NewDynamicAdvancedSession(cfg.ModelPath, []string{in.Name}, []string{out.Name}, opts)
	if err != nil {
		return nil, onnxrt.InputOutputInfo{}, onnxrt.InputOutputInfo{}, fmt.Errorf("session: %w", err)
	}

	return sess, in, out, nil
}

// setONNXLibraryPath mirrors orientation's helper to prefer the project-local runtime.
func setONNXLibraryPath() error {
	// Try system paths first
	if path := findSystemONNXLibrary(); path != "" {
		onnxrt.SetSharedLibraryPath(path)
		return nil
	}

	// Try project-relative path
	projectLib, err := findProjectONNXLibrary()
	if err != nil {
		return err
	}
	onnxrt.SetSharedLibraryPath(projectLib)
	return nil
}

func findSystemONNXLibrary() string {
	systemPaths := []string{
		"/usr/local/lib/libonnxruntime.so",
		"/usr/lib/libonnxruntime.so",
		"/opt/onnxruntime/cpu/lib/libonnxruntime.so",
	}
	for _, path := range systemPaths {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return ""
}

func findProjectONNXLibrary() (string, error) {
	root, err := findProjectRoot()
	if err != nil {
		return "", err
	}

	libName, err := getONNXLibraryName()
	if err != nil {
		return "", err
	}

	libPath := filepath.Join(root, "onnxruntime", "lib", libName)
	if _, err := os.Stat(libPath); err != nil {
		return "", fmt.Errorf("ONNX Runtime library not found at %s", libPath)
	}
	return libPath, nil
}

func findProjectRoot() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	root := cwd
	for {
		if _, err := os.Stat(filepath.Join(root, "go.mod")); err == nil {
			return root, nil
		}
		parent := filepath.Dir(root)
		if parent == root {
			return "", errors.New("could not find project root")
		}
		root = parent
	}
}

func getONNXLibraryName() (string, error) {
	switch runtime.GOOS {
	case "linux":
		return "libonnxruntime.so", nil
	case "darwin":
		return "libonnxruntime.dylib", nil
	case "windows":
		return "onnxruntime.dll", nil
	default:
		return "", fmt.Errorf("unsupported OS: %s", runtime.GOOS)
	}
}
