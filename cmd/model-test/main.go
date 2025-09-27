package main

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/yalue/onnxruntime_go"
)

func main() {
	// Set up structured logging
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	// Set the shared library path
	onnxruntime_go.SetSharedLibraryPath("/usr/local/lib/libonnxruntime.so")

	// Initialize ONNX Runtime
	err := onnxruntime_go.InitializeEnvironment()
	if err != nil {
		slog.Error("Failed to initialize ONNX Runtime", "error", err)
		os.Exit(1)
	}
	defer func() {
		if err := onnxruntime_go.DestroyEnvironment(); err != nil {
			slog.Error("Failed to destroy ONNX Runtime environment", "error", err)
		}
	}()

	modelsDir := "models"
	models := []string{
		"PP-OCRv5_mobile_det.onnx",
		"PP-OCRv5_server_det.onnx",
		"PP-OCRv5_mobile_rec.onnx",
		"PP-OCRv5_server_rec.onnx",
		"pplcnet_x1_0_doc_ori.onnx",
		"pplcnet_x0_25_textline_ori.onnx",
		"pplcnet_x1_0_textline_ori.onnx",
		"uvdoc.onnx",
	}

	fmt.Println("Testing ONNX model compatibility...")
	fmt.Println("=====================================")

	for _, modelName := range models {
		modelPath := filepath.Join(modelsDir, modelName)

		// Check if model file exists
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			fmt.Printf("❌ %s: File not found\n", modelName)
			continue
		}

		// Get model metadata and input/output info
		inputs, outputs, err := onnxruntime_go.GetInputOutputInfo(modelPath)
		if err != nil {
			fmt.Printf("❌ %s: Failed to get model info - %v\n", modelName, err)
			continue
		}

		fmt.Printf("✅ %s: Compatible with ONNX Runtime\n", modelName)
		fmt.Printf("   - Inputs: %d\n", len(inputs))
		for i, input := range inputs {
			fmt.Printf("     [%d] %s: %v (type: %s)\n", i, input.Name, input.Dimensions, input.DataType)
		}
		fmt.Printf("   - Outputs: %d\n", len(outputs))
		for i, output := range outputs {
			fmt.Printf("     [%d] %s: %v (type: %s)\n", i, output.Name, output.Dimensions, output.DataType)
		}

		// Also get model metadata
		metadata, err := onnxruntime_go.GetModelMetadata(modelPath)
		if err == nil {
			if producer, err := metadata.GetProducerName(); err == nil && producer != "" {
				fmt.Printf("   - Producer: %s\n", producer)
			}
			if version, err := metadata.GetVersion(); err == nil {
				fmt.Printf("   - Version: %d\n", version)
			}
			if description, err := metadata.GetDescription(); err == nil && description != "" {
				fmt.Printf("   - Description: %s\n", description)
			}
			if err := metadata.Destroy(); err != nil {
				slog.Error("Failed to destroy model metadata", "error", err)
			}
		}

		fmt.Println()
	}

	fmt.Println("Model compatibility test completed!")
}
