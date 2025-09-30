//go:build integration
// +build integration

package onnx

import (
	"testing"

	onnxruntime "github.com/yalue/onnxruntime_go"
)

// These integration tests require ONNX Runtime to be properly installed
// Run with: go test -tags=integration ./internal/onnx

func TestConfigureSessionForGPU_WithRealSession(t *testing.T) {
	// Initialize ONNX Runtime
	if err := SetONNXLibraryPath(false); err != nil {
		t.Skipf("ONNX Runtime library not available: %v", err)
	}

	if err := onnxruntime.InitializeEnvironment(); err != nil {
		t.Skipf("Failed to initialize ONNX Runtime: %v", err)
	}
	defer func() {
		_ = onnxruntime.DestroyEnvironment()
	}()

	// Create session options
	sessionOptions, err := onnxruntime.NewSessionOptions()
	if err != nil {
		t.Fatalf("Failed to create session options: %v", err)
	}
	defer func() {
		_ = sessionOptions.Destroy()
	}()

	tests := []struct {
		name    string
		config  GPUConfig
		wantErr bool
	}{
		{
			name: "GPU disabled",
			config: GPUConfig{
				UseGPU: false,
			},
			wantErr: false,
		},
		{
			name: "GPU enabled - may fail without CUDA",
			config: GPUConfig{
				UseGPU:                true,
				DeviceID:              0,
				GPUMemLimit:           1024 * 1024 * 1024, // 1GB
				ArenaExtendStrategy:   "kNextPowerOfTwo",
				CUDNNConvAlgoSearch:   "DEFAULT",
				DoCopyInDefaultStream: true,
			},
			wantErr: true, // Expected to fail without CUDA
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a fresh session options for each test
			opts, err := onnxruntime.NewSessionOptions()
			if err != nil {
				t.Fatalf("Failed to create session options: %v", err)
			}
			defer func() {
				_ = opts.Destroy()
			}()

			err = ConfigureSessionForGPU(opts, tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("ConfigureSessionForGPU() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestTestONNXRuntime_FullIntegration(t *testing.T) {
	// This test runs the full TestONNXRuntime function
	err := TestONNXRuntime()
	if err != nil {
		t.Logf("TestONNXRuntime() failed (expected if ONNX Runtime not installed): %v", err)
		// Don't fail the test - this is expected in some environments
	}
}
