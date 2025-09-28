package detector

import (
	"fmt"

	"github.com/MeKo-Tech/pogo/internal/onnx"
	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/yalue/onnxruntime_go"
)

// setupONNXEnvironment sets up the ONNX Runtime environment.
func setupONNXEnvironment(useGPU bool) error {
	if err := onnx.SetONNXLibraryPath(useGPU); err != nil {
		return fmt.Errorf("failed to set ONNX Runtime library path: %w", err)
	}

	if !onnxruntime_go.IsInitialized() {
		if err := onnxruntime_go.InitializeEnvironment(); err != nil {
			return fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
		}
	}
	return nil
}

// createSession creates the ONNX session with the given configuration.
func createSession(modelPath string, inputInfo, outputInfo onnxruntime_go.InputOutputInfo,
	config Config,
) (*onnxruntime_go.DynamicAdvancedSession, error) {
	sessionOptions, err := onnxruntime_go.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer func() {
		if err := sessionOptions.Destroy(); err != nil {
			fmt.Printf("Failed to destroy session options: %v", err)
		}
	}()

	if err := onnx.ConfigureSessionForGPU(sessionOptions, config.GPU); err != nil {
		return nil, fmt.Errorf("failed to configure GPU: %w", err)
	}

	if config.NumThreads > 0 {
		if err = sessionOptions.SetIntraOpNumThreads(config.NumThreads); err != nil {
			return nil, fmt.Errorf("failed to set thread count: %w", err)
		}
	}

	session, err := onnxruntime_go.NewDynamicAdvancedSession(modelPath,
		[]string{inputInfo.Name}, []string{outputInfo.Name}, sessionOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return session, nil
}

// setupImageConstraints creates image constraints based on config.
func setupImageConstraints(config Config) utils.ImageConstraints {
	return utils.ImageConstraints{
		MaxWidth:  config.MaxImageSize,
		MaxHeight: config.MaxImageSize,
		MinWidth:  32,
		MinHeight: 32,
	}
}
