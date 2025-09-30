package models

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// helpers to compute default expected paths without deep nesting.
func expectedDetectionDefault(useServer bool) string {
	base := DefaultModelsDir
	if projectRoot, err := findProjectRoot(); err == nil {
		base = filepath.Join(projectRoot, DefaultModelsDir)
		if useServer {
			return filepath.Join(base, TypeDetection, VariantServer, DetectionServer)
		}
		return filepath.Join(base, TypeDetection, VariantMobile, DetectionMobile)
	}
	if useServer {
		return filepath.Join(base, DetectionServer)
	}
	return filepath.Join(base, DetectionMobile)
}

func expectedRecognitionDefault(useServer bool) string {
	base := DefaultModelsDir
	if projectRoot, err := findProjectRoot(); err == nil {
		base = filepath.Join(projectRoot, DefaultModelsDir)
		if useServer {
			return filepath.Join(base, TypeRecognition, VariantServer, RecognitionServer)
		}
		return filepath.Join(base, TypeRecognition, VariantMobile, RecognitionMobile)
	}
	if useServer {
		return filepath.Join(base, RecognitionServer)
	}
	return filepath.Join(base, RecognitionMobile)
}

func TestGetModelsDir(t *testing.T) {
	tests := []struct {
		name           string
		explicitDir    string
		envVar         string
		expectedResult string
	}{
		{
			name:           "explicit directory takes precedence",
			explicitDir:    "/explicit/path",
			envVar:         "/env/path",
			expectedResult: "/explicit/path",
		},
		{
			name:           "environment variable used when no explicit dir",
			explicitDir:    "",
			envVar:         "/env/path",
			expectedResult: "/env/path",
		},
		{
			name:           "default used when neither provided",
			explicitDir:    "",
			envVar:         "",
			expectedResult: "", // Will be set dynamically in the test
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set environment variable
			if tt.envVar != "" {
				require.NoError(t, os.Setenv(EnvModelsDir, tt.envVar))
			} else {
				require.NoError(t, os.Unsetenv(EnvModelsDir))
			}
			defer func() {
				require.NoError(t, os.Unsetenv(EnvModelsDir))
			}()
			result := GetModelsDir(tt.explicitDir)

			expectedResult := tt.expectedResult
			if expectedResult == "" {
				base := DefaultModelsDir
				if projectRoot, err := findProjectRoot(); err == nil {
					base = filepath.Join(projectRoot, DefaultModelsDir)
				}
				expectedResult = base
			}

			assert.Equal(t, expectedResult, result)
		})
	}
}

func TestGetDetectionModelPath(t *testing.T) {
	tests := []struct {
		name      string
		modelsDir string
		useServer bool
		expected  string
	}{
		{
			name:      "mobile detection model with custom dir (falls back to flat)",
			modelsDir: "/custom",
			useServer: false,
			expected:  filepath.Join("/custom", DetectionMobile),
		},
		{
			name:      "server detection model with default dir (uses organized structure)",
			modelsDir: "",
			useServer: true,
			expected:  "", // Will be calculated dynamically
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetDetectionModelPath(tt.modelsDir, tt.useServer)

			expected := tt.expected
			if expected == "" {
				expected = expectedDetectionDefault(tt.useServer)
			}

			assert.Equal(t, expected, result)
		})
	}
}

func TestGetRecognitionModelPath(t *testing.T) {
	tests := []struct {
		name      string
		modelsDir string
		useServer bool
		expected  string
	}{
		{
			name:      "mobile recognition model (uses organized structure)",
			modelsDir: "",
			useServer: false,
			expected:  "", // Will be calculated dynamically
		},
		{
			name:      "server recognition model (falls back to flat)",
			modelsDir: "/test",
			useServer: true,
			expected:  filepath.Join("/test", RecognitionServer),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetRecognitionModelPath(tt.modelsDir, tt.useServer)

			expected := tt.expected
			if expected == "" {
				expected = expectedRecognitionDefault(tt.useServer)
			}

			assert.Equal(t, expected, result)
		})
	}
}

func TestGetDictionaryPath(t *testing.T) {
	// Test with default directory (should use organized structure)
	result := GetDictionaryPath("", DictionaryPPOCRKeysV1)
	var expected string
	if projectRoot, err := findProjectRoot(); err == nil {
		modelsDir := filepath.Join(projectRoot, DefaultModelsDir)
		expected = filepath.Join(modelsDir, TypeDictionaries, DictionaryPPOCRKeysV1)
	} else {
		expected = filepath.Join(DefaultModelsDir, DictionaryPPOCRKeysV1)
	}
	assert.Equal(t, expected, result)

	// Test with custom directory (should use flat structure)
	result = GetDictionaryPath("/custom", DictionaryPPOCRKeysV1)
	expected = filepath.Join("/custom", DictionaryPPOCRKeysV1)
	assert.Equal(t, expected, result)
}

func TestListAvailableModels(t *testing.T) {
	models := ListAvailableModels()
	assert.NotEmpty(t, models)

	// Check that we have the expected model types
	var hasDetection, hasRecognition, hasDictionary bool
	for _, model := range models {
		switch model.Type {
		case TypeDetection:
			hasDetection = true
		case TypeRecognition:
			hasRecognition = true
		case TypeDictionaries:
			hasDictionary = true
		}
	}

	assert.True(t, hasDetection, "Should have detection models")
	assert.True(t, hasRecognition, "Should have recognition models")
	assert.True(t, hasDictionary, "Should have dictionary files")
}

func TestResolveModelPath_BackwardCompatibility(t *testing.T) {
	// Test that it falls back to flat structure when organized structure doesn't exist
	result := ResolveModelPath("/nonexistent", TypeDetection, VariantMobile, DetectionMobile)
	expected := filepath.Join("/nonexistent", DetectionMobile)
	assert.Equal(t, expected, result)
}

func TestResolveModelPath_OrganizedStructure(t *testing.T) {
	// Test path resolution behavior - should use organized structure when models exist
	result := GetDetectionModelPath("", false)
	var expected string
	if projectRoot, err := findProjectRoot(); err == nil {
		modelsDir := filepath.Join(projectRoot, DefaultModelsDir)
		expected = filepath.Join(modelsDir, TypeDetection, VariantMobile, DetectionMobile)
	} else {
		expected = filepath.Join(DefaultModelsDir, DetectionMobile)
	}
	assert.Equal(t, expected, result)

	// Verify the function generates the correct organized path when directory structure exists
	if projectRoot, err := findProjectRoot(); err == nil {
		modelsDir := filepath.Join(projectRoot, DefaultModelsDir)
		organizedPath := ResolveModelPath(modelsDir, TypeDetection, VariantMobile, DetectionMobile)
		expectedOrganized := filepath.Join(modelsDir, TypeDetection, VariantMobile, DetectionMobile)
		assert.Equal(t, expectedOrganized, organizedPath)
	}
}

func TestGetLayoutModelPath(t *testing.T) {
	tests := []struct {
		name      string
		modelsDir string
		filename  string
		expected  string
	}{
		{
			name:      "layout model with custom dir",
			modelsDir: "/custom",
			filename:  LayoutPPLCNetX025Textline,
			expected:  filepath.Join("/custom", LayoutPPLCNetX025Textline),
		},
		{
			name:      "layout model with default dir",
			modelsDir: "",
			filename:  LayoutPPLCNetX10Doc,
			expected:  "", // Will be calculated dynamically
		},
		{
			name:      "uvdoc layout model",
			modelsDir: "/test/models",
			filename:  LayoutUVDoc,
			expected:  filepath.Join("/test/models", LayoutUVDoc),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetLayoutModelPath(tt.modelsDir, tt.filename)

			expected := tt.expected
			if expected == "" {
				base := DefaultModelsDir
				if projectRoot, err := findProjectRoot(); err == nil {
					base = filepath.Join(projectRoot, DefaultModelsDir)
				}
				expected = filepath.Join(base, TypeLayout, tt.filename)
			}

			assert.Equal(t, expected, result)
		})
	}
}

func TestGetDocTRModelPath(t *testing.T) {
	tests := []struct {
		name      string
		modelsDir string
		checkPath func(t *testing.T, result string)
	}{
		{
			name:      "doctr model with custom dir",
			modelsDir: "/custom/models",
			checkPath: func(t *testing.T, result string) {
				t.Helper()
				expected := filepath.Join("/custom/models", LayoutDocTR)
				assert.Equal(t, expected, result)
			},
		},
		{
			name:      "doctr model with default dir",
			modelsDir: "",
			checkPath: func(t *testing.T, result string) {
				t.Helper()
				// The path should either be organized or flat structure depending on what exists
				base := DefaultModelsDir
				if projectRoot, err := findProjectRoot(); err == nil {
					base = filepath.Join(projectRoot, DefaultModelsDir)
				}

				// Check if it's one of the valid paths (organized or flat)
				organizedPath := filepath.Join(base, TypeLayout, LayoutDocTR)
				flatPath := filepath.Join(base, LayoutDocTR)

				// ResolveModelPath tries organized first, falls back to flat
				// Since we're testing behavior, accept either valid path
				assert.True(t, result == organizedPath || result == flatPath,
					"Path should be either organized (%s) or flat (%s), got: %s",
					organizedPath, flatPath, result)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetDocTRModelPath(tt.modelsDir)
			tt.checkPath(t, result)
		})
	}
}

func TestValidateModelExists(t *testing.T) {
	// Create a temporary file for testing
	tmpFile, err := os.CreateTemp("", "model_test_*.onnx")
	require.NoError(t, err)
	tmpPath := tmpFile.Name()
	_ = tmpFile.Close()
	defer func() { _ = os.Remove(tmpPath) }()

	tests := []struct {
		name      string
		modelPath string
		wantErr   bool
	}{
		{
			name:      "existing model file",
			modelPath: tmpPath,
			wantErr:   false,
		},
		{
			name:      "non-existent model file",
			modelPath: "/nonexistent/path/to/model.onnx",
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateModelExists(tt.modelPath)
			if tt.wantErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "model file not found")
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestGetDictionaryPathsForLanguages(t *testing.T) {
	// Create a temporary directory structure for testing
	tmpDir, err := os.MkdirTemp("", "dict_test_*")
	require.NoError(t, err)
	defer func() { _ = os.RemoveAll(tmpDir) }()

	dictDir := filepath.Join(tmpDir, TypeDictionaries)
	require.NoError(t, os.MkdirAll(dictDir, 0o755))

	// Create test dictionary files
	enDict := filepath.Join(dictDir, "ppocr_keys_en.txt")
	require.NoError(t, os.WriteFile(enDict, []byte("test"), 0o644))

	zhDict := filepath.Join(dictDir, "ppocr_keys_zh.txt")
	require.NoError(t, os.WriteFile(zhDict, []byte("test"), 0o644))

	defaultDict := filepath.Join(dictDir, DictionaryPPOCRKeysV1)
	require.NoError(t, os.WriteFile(defaultDict, []byte("test"), 0o644))

	tests := []struct {
		name           string
		modelsDir      string
		languages      []string
		expectedCount  int
		shouldContain  []string
		shouldNotExist []string
	}{
		{
			name:          "single language with existing dictionary",
			modelsDir:     tmpDir,
			languages:     []string{"en"},
			expectedCount: 2, // en + default
			shouldContain: []string{enDict, defaultDict},
		},
		{
			name:          "multiple languages with existing dictionaries",
			modelsDir:     tmpDir,
			languages:     []string{"en", "zh"},
			expectedCount: 3, // en + zh + default
			shouldContain: []string{enDict, zhDict, defaultDict},
		},
		{
			name:          "non-existent language falls back to default",
			modelsDir:     tmpDir,
			languages:     []string{"fr"},
			expectedCount: 1, // only default
			shouldContain: []string{defaultDict},
		},
		{
			name:          "empty languages returns default",
			modelsDir:     tmpDir,
			languages:     []string{},
			expectedCount: 1, // only default
			shouldContain: []string{defaultDict},
		},
		{
			name:          "duplicate languages deduplicated",
			modelsDir:     tmpDir,
			languages:     []string{"en", "en"},
			expectedCount: 2, // en + default (no duplicates)
			shouldContain: []string{enDict, defaultDict},
		},
		{
			name:           "empty string language ignored",
			modelsDir:      tmpDir,
			languages:      []string{"", "en"},
			expectedCount:  2, // en + default
			shouldContain:  []string{enDict, defaultDict},
			shouldNotExist: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetDictionaryPathsForLanguages(tt.modelsDir, tt.languages)

			assert.Len(t, result, tt.expectedCount, "Expected %d paths, got %d", tt.expectedCount, len(result))

			for _, expectedPath := range tt.shouldContain {
				assert.Contains(t, result, expectedPath, "Result should contain %s", expectedPath)
			}

			// Check for duplicates
			seen := make(map[string]bool)
			for _, path := range result {
				assert.False(t, seen[path], "Found duplicate path: %s", path)
				seen[path] = true
			}
		})
	}
}

func TestGetDictionaryPathsForLanguages_NoDefaultFallback(t *testing.T) {
	// Test case where no dictionaries exist at all
	tmpDir, err := os.MkdirTemp("", "dict_empty_test_*")
	require.NoError(t, err)
	defer func() { _ = os.RemoveAll(tmpDir) }()

	result := GetDictionaryPathsForLanguages(tmpDir, []string{"en", "zh"})
	assert.Empty(t, result, "Should return empty slice when no dictionaries exist")
}

func TestGetDictionaryPathsForLanguages_AlternativePatterns(t *testing.T) {
	// Test alternative naming patterns for dictionaries
	tmpDir, err := os.MkdirTemp("", "dict_patterns_test_*")
	require.NoError(t, err)
	defer func() { _ = os.RemoveAll(tmpDir) }()

	dictDir := filepath.Join(tmpDir, TypeDictionaries)
	require.NoError(t, os.MkdirAll(dictDir, 0o755))

	// Create dictionaries with alternative naming patterns
	keysEnDict := filepath.Join(dictDir, "keys_en.txt")
	require.NoError(t, os.WriteFile(keysEnDict, []byte("test"), 0o644))

	enTxtDict := filepath.Join(dictDir, "en.txt")
	require.NoError(t, os.WriteFile(enTxtDict, []byte("test"), 0o644))

	defaultDict := filepath.Join(dictDir, DictionaryPPOCRKeysV1)
	require.NoError(t, os.WriteFile(defaultDict, []byte("test"), 0o644))

	// Test that all patterns are tried
	result := GetDictionaryPathsForLanguages(tmpDir, []string{"en"})

	// Should find at least the keys_en.txt pattern (first to be tried) and default
	assert.Contains(t, result, defaultDict)
	// At least one of the en patterns should be found
	hasEnPattern := false
	for _, path := range result {
		if path == keysEnDict || path == enTxtDict {
			hasEnPattern = true
			break
		}
	}
	assert.True(t, hasEnPattern, "Should find at least one en dictionary pattern")
}

func TestResolveModelPath_WithoutVariant(t *testing.T) {
	// Test ResolveModelPath when variant is empty (layout models)
	result := ResolveModelPath("/test", TypeLayout, "", LayoutUVDoc)
	expected := filepath.Join("/test", LayoutUVDoc)
	assert.Equal(t, expected, result)
}

func TestResolveModelPath_EmptyModelType(t *testing.T) {
	// Test ResolveModelPath when modelType is empty (fallback to flat structure)
	result := ResolveModelPath("/test", "", "", "some_model.onnx")
	expected := filepath.Join("/test", "some_model.onnx")
	assert.Equal(t, expected, result)
}

func TestFindProjectRoot(t *testing.T) {
	// Test that findProjectRoot succeeds in the current project
	root, err := findProjectRoot()
	if err == nil {
		// If we're in a Go project, verify go.mod exists
		goModPath := filepath.Join(root, "go.mod")
		_, statErr := os.Stat(goModPath)
		assert.NoError(t, statErr, "go.mod should exist at project root")
	}
	// If err != nil, we're not in a Go project, which is also valid
}

func TestListAvailableModels_Structure(t *testing.T) {
	models := ListAvailableModels()

	// Verify model structure
	for _, model := range models {
		assert.NotEmpty(t, model.Name, "Model should have a name")
		assert.NotEmpty(t, model.Type, "Model should have a type")
		assert.NotEmpty(t, model.Filename, "Model should have a filename")
		assert.NotEmpty(t, model.Description, "Model should have a description")
		// Variant can be empty for some model types (layout, dictionaries)
	}

	// Check for specific expected models
	var foundMobileDetection, foundServerRecognition, foundDocTR bool
	for _, model := range models {
		if model.Name == "mobile-detection" {
			foundMobileDetection = true
			assert.Equal(t, TypeDetection, model.Type)
			assert.Equal(t, VariantMobile, model.Variant)
		}
		if model.Name == "server-recognition" {
			foundServerRecognition = true
			assert.Equal(t, TypeRecognition, model.Type)
			assert.Equal(t, VariantServer, model.Variant)
		}
		if model.Name == "doctr" {
			foundDocTR = true
			assert.Equal(t, TypeLayout, model.Type)
			assert.Empty(t, model.Variant)
		}
	}

	assert.True(t, foundMobileDetection, "Should have mobile-detection model")
	assert.True(t, foundServerRecognition, "Should have server-recognition model")
	assert.True(t, foundDocTR, "Should have doctr model")
}

func TestModelConstants(t *testing.T) {
	// Test that model constants are defined and non-empty
	assert.NotEmpty(t, DetectionMobile)
	assert.NotEmpty(t, DetectionServer)
	assert.NotEmpty(t, RecognitionMobile)
	assert.NotEmpty(t, RecognitionServer)
	assert.NotEmpty(t, LayoutPPLCNetX025Textline)
	assert.NotEmpty(t, LayoutPPLCNetX10Doc)
	assert.NotEmpty(t, LayoutPPLCNetX10Textline)
	assert.NotEmpty(t, LayoutUVDoc)
	assert.NotEmpty(t, LayoutDocTR)
	assert.NotEmpty(t, DictionaryPPOCRKeysV1)

	// Test type constants
	assert.NotEmpty(t, TypeDetection)
	assert.NotEmpty(t, TypeRecognition)
	assert.NotEmpty(t, TypeLayout)
	assert.NotEmpty(t, TypeDictionaries)

	// Test variant constants
	assert.NotEmpty(t, VariantMobile)
	assert.NotEmpty(t, VariantServer)

	// Test environment variable constant
	assert.NotEmpty(t, EnvModelsDir)
	assert.NotEmpty(t, DefaultModelsDir)
}
