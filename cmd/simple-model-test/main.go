package main

import (
	"fmt"
	"os"
	"path/filepath"
)

func main() {
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

	fmt.Println("Checking ONNX model files...")
	fmt.Println("=============================")

	allPresent := true
	for _, modelName := range models {
		modelPath := filepath.Join(modelsDir, modelName)

		// Check if model file exists
		if stat, err := os.Stat(modelPath); os.IsNotExist(err) {
			fmt.Printf("❌ %s: File not found\n", modelName)
			allPresent = false
		} else {
			size := float64(stat.Size()) / (1024 * 1024) // Size in MB
			fmt.Printf("✅ %s: Present (%.1f MB)\n", modelName, size)
		}
	}

	fmt.Println()
	if allPresent {
		fmt.Println("✅ All required ONNX models are present!")
		fmt.Println("Note: ONNX Runtime compatibility will be tested when the runtime is properly configured.")
	} else {
		fmt.Println("❌ Some models are missing. Please run the model download scripts.")
	}
}
