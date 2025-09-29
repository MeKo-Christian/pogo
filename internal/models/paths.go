package models

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// Model name constants to avoid typos and ensure consistency.
const (
	// Detection models.
	DetectionMobile = "PP-OCRv5_mobile_det.onnx"
	DetectionServer = "PP-OCRv5_server_det.onnx"

	// Recognition models.
	RecognitionMobile = "PP-OCRv5_mobile_rec.onnx"
	RecognitionServer = "PP-OCRv5_server_rec.onnx"

	// Layout analysis models.
	LayoutPPLCNetX025Textline = "pplcnet_x0_25_textline_ori.onnx"
	LayoutPPLCNetX10Doc       = "pplcnet_x1_0_doc_ori.onnx"
	LayoutPPLCNetX10Textline  = "pplcnet_x1_0_textline_ori.onnx"
	LayoutUVDoc               = "uvdoc.onnx"
	LayoutDocTR               = "doctr.onnx"

	// Dictionary files.
	DictionaryPPOCRKeysV1 = "ppocr_keys_v1.txt"
)

// Model type categories for organized directory structure.
const (
	TypeDetection    = "detection"
	TypeRecognition  = "recognition"
	TypeLayout       = "layout"
	TypeDictionaries = "dictionaries"
)

// Model variant categories.
const (
	VariantMobile = "mobile"
	VariantServer = "server"
)

// Default models directory.
const DefaultModelsDir = "models"

// Environment variable for models directory override.
const EnvModelsDir = "GO_OAR_OCR_MODELS_DIR"

// findProjectRoot finds the project root by looking for go.mod.
func findProjectRoot() (string, error) {
	// Start from current working directory
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}

	// Walk up the directory tree looking for go.mod
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			// Reached filesystem root without finding go.mod
			break
		}
		dir = parent
	}

	return "", errors.New("could not find project root (go.mod not found)")
}

// ModelInfo contains metadata about a model.
type ModelInfo struct {
	Name        string
	Type        string
	Variant     string
	Description string
	Filename    string
}

// GetModelsDir returns the models directory path from various sources
// Priority: 1. Explicit modelsDir parameter, 2. Environment variable, 3. Project root + default.
func GetModelsDir(modelsDir string) string {
	if modelsDir != "" {
		return modelsDir
	}

	if envDir := os.Getenv(EnvModelsDir); envDir != "" {
		return envDir
	}

	// Use project root + default models directory
	if projectRoot, err := findProjectRoot(); err == nil {
		return filepath.Join(projectRoot, DefaultModelsDir)
	}

	// Fallback to relative path if project root can't be found
	return DefaultModelsDir
}

// ResolveModelPath resolves a model filename to its full path
// Supports both new organized structure and legacy flat structure for backward compatibility.
func ResolveModelPath(modelsDir, modelType, variant, filename string) string {
	baseDir := GetModelsDir(modelsDir)

	// Try new organized structure first
	if modelType != "" {
		var organizedPath string
		if variant != "" && (modelType == TypeDetection || modelType == TypeRecognition) {
			organizedPath = filepath.Join(baseDir, modelType, variant, filename)
		} else {
			organizedPath = filepath.Join(baseDir, modelType, filename)
		}

		// Check if organized path exists
		if _, err := os.Stat(organizedPath); err == nil {
			return organizedPath
		}
	}

	// Fall back to legacy flat structure
	return filepath.Join(baseDir, filename)
}

// GetDetectionModelPath returns the path for a detection model.
func GetDetectionModelPath(modelsDir string, useServer bool) string {
	var filename string
	if useServer {
		filename = DetectionServer
	} else {
		filename = DetectionMobile
	}

	variant := VariantMobile
	if useServer {
		variant = VariantServer
	}

	return ResolveModelPath(modelsDir, TypeDetection, variant, filename)
}

// GetRecognitionModelPath returns the path for a recognition model.
func GetRecognitionModelPath(modelsDir string, useServer bool) string {
	var filename string
	if useServer {
		filename = RecognitionServer
	} else {
		filename = RecognitionMobile
	}

	variant := VariantMobile
	if useServer {
		variant = VariantServer
	}

	return ResolveModelPath(modelsDir, TypeRecognition, variant, filename)
}

// GetDictionaryPath returns the path for a dictionary file.
func GetDictionaryPath(modelsDir, filename string) string {
	return ResolveModelPath(modelsDir, TypeDictionaries, "", filename)
}

// GetLayoutModelPath returns the path for a layout analysis model.
func GetLayoutModelPath(modelsDir, filename string) string {
	return ResolveModelPath(modelsDir, TypeLayout, "", filename)
}

// GetDocTRModelPath returns the path for the DocTR rectification model.
func GetDocTRModelPath(modelsDir string) string {
	return GetLayoutModelPath(modelsDir, LayoutDocTR)
}

// ValidateModelExists checks if a model file exists at the given path.
func ValidateModelExists(modelPath string) error {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return fmt.Errorf("model file not found: %s", modelPath)
	}
	return nil
}

// ListAvailableModels returns information about available models.
func ListAvailableModels() []ModelInfo {
	return []ModelInfo{
		{
			Name:        "mobile-detection",
			Type:        TypeDetection,
			Variant:     VariantMobile,
			Description: "Mobile detection model",
			Filename:    DetectionMobile,
		},
		{
			Name:        "server-detection",
			Type:        TypeDetection,
			Variant:     VariantServer,
			Description: "Server detection model",
			Filename:    DetectionServer,
		},
		{
			Name:        "mobile-recognition",
			Type:        TypeRecognition,
			Variant:     VariantMobile,
			Description: "Mobile recognition model",
			Filename:    RecognitionMobile,
		},
		{
			Name:        "server-recognition",
			Type:        TypeRecognition,
			Variant:     VariantServer,
			Description: "Server recognition model",
			Filename:    RecognitionServer,
		},
		{
			Name:        "pplcnet-x0.25-textline",
			Type:        TypeLayout,
			Variant:     "",
			Description: "PPLCNet x0.25 textline model",
			Filename:    LayoutPPLCNetX025Textline,
		},
		{
			Name:        "pplcnet-x1.0-doc",
			Type:        TypeLayout,
			Variant:     "",
			Description: "PPLCNet x1.0 document model",
			Filename:    LayoutPPLCNetX10Doc,
		},
		{
			Name:        "pplcnet-x1.0-textline",
			Type:        TypeLayout,
			Variant:     "",
			Description: "PPLCNet x1.0 textline model",
			Filename:    LayoutPPLCNetX10Textline,
		},
		{
			Name:        "uvdoc",
			Type:        TypeLayout,
			Variant:     "",
			Description: "UVDoc layout model",
			Filename:    LayoutUVDoc,
		},
		{
			Name:        "doctr",
			Type:        TypeLayout,
			Variant:     "",
			Description: "DocTR document rectification model",
			Filename:    LayoutDocTR,
		},
		{
			Name:        "ppocr-keys-v1",
			Type:        TypeDictionaries,
			Variant:     "",
			Description: "PPOCR character dictionary v1",
			Filename:    DictionaryPPOCRKeysV1,
		},
	}
}

// GetDictionaryPathsForLanguages tries to resolve dictionary files for the given
// language codes. It searches under modelsDir/dictionaries for common naming patterns
// and falls back to the default dictionary if no language-specific file is found.
// The returned list is de-duplicated and ordered by the input languages.
func GetDictionaryPathsForLanguages(modelsDir string, languages []string) []string {
	base := GetModelsDir(modelsDir)
	out := make([]string, 0, len(languages)+1)
	seen := make(map[string]struct{}, len(languages)+1)
	tryAdd := func(p string) {
		if p == "" {
			return
		}
		if _, err := os.Stat(p); err == nil {
			if _, ok := seen[p]; !ok {
				seen[p] = struct{}{}
				out = append(out, p)
			}
		}
	}
	for _, lang := range languages {
		if lang == "" {
			continue
		}
		// Try a few common patterns
		tryAdd(filepath.Join(base, TypeDictionaries, "ppocr_keys_"+lang+".txt"))
		tryAdd(filepath.Join(base, TypeDictionaries, "keys_"+lang+".txt"))
		tryAdd(filepath.Join(base, TypeDictionaries, lang+".txt"))
	}
	// Always ensure a default dictionary exists
	def := GetDictionaryPath(base, DictionaryPPOCRKeysV1)
	if _, err := os.Stat(def); err == nil {
		if _, ok := seen[def]; !ok {
			out = append(out, def)
		}
	}
	return out
}
