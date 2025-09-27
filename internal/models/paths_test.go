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
