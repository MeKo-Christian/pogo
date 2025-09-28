package detector

import (
	"path/filepath"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/stretchr/testify/assert"
)

const testModelsDir = "/test/models"

func TestConfig_UpdateModelPath(t *testing.T) {
	tests := []struct {
		name           string
		initialConfig  Config
		modelsDir      string
		expectedSuffix string
	}{
		{
			name: "Update with mobile model",
			initialConfig: Config{
				ModelPath:      "old/path/model.onnx",
				UseServerModel: false,
			},
			modelsDir:      testModelsDir,
			expectedSuffix: models.DetectionMobile,
		},
		{
			name: "Update with server model",
			initialConfig: Config{
				ModelPath:      "old/path/model.onnx",
				UseServerModel: true,
			},
			modelsDir:      testModelsDir,
			expectedSuffix: models.DetectionServer,
		},
		{
			name: "Update with empty models dir",
			initialConfig: Config{
				ModelPath:      "old/path/model.onnx",
				UseServerModel: false,
			},
			modelsDir:      "",
			expectedSuffix: models.DetectionMobile,
		},
		{
			name: "Update preserves other config fields",
			initialConfig: Config{
				ModelPath:      "old/path/model.onnx",
				UseServerModel: false,
				DbThresh:       0.4,
				DbBoxThresh:    0.6,
				MaxImageSize:   1024,
				NumThreads:     4,
			},
			modelsDir:      "/custom/models",
			expectedSuffix: models.DetectionMobile,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := tt.initialConfig
			originalDbThresh := config.DbThresh
			originalDbBoxThresh := config.DbBoxThresh
			originalMaxImageSize := config.MaxImageSize
			originalNumThreads := config.NumThreads

			// Call UpdateModelPath
			config.UpdateModelPath(tt.modelsDir)

			// Verify the model path was updated correctly
			expectedPath := models.GetDetectionModelPath(tt.modelsDir, tt.initialConfig.UseServerModel)
			assert.Equal(t, expectedPath, config.ModelPath)

			// Verify the path contains the expected suffix
			assert.Contains(t, config.ModelPath, tt.expectedSuffix)

			// Verify UseServerModel flag is preserved
			assert.Equal(t, tt.initialConfig.UseServerModel, config.UseServerModel)

			// Verify other config fields are preserved
			if originalDbThresh != 0 {
				assert.InDelta(t, originalDbThresh, config.DbThresh, 1e-6)
			}
			if originalDbBoxThresh != 0 {
				assert.InDelta(t, originalDbBoxThresh, config.DbBoxThresh, 1e-6)
			}
			if originalMaxImageSize != 0 {
				assert.Equal(t, originalMaxImageSize, config.MaxImageSize)
			}
			if originalNumThreads != 0 {
				assert.Equal(t, originalNumThreads, config.NumThreads)
			}
		})
	}
}

func TestConfig_UpdateModelPath_PathStructure(t *testing.T) {
	config := Config{
		ModelPath:      "old/path/model.onnx",
		UseServerModel: false,
	}

	modelsDir := testModelsDir
	config.UpdateModelPath(modelsDir)

	// Verify the path starts with the models directory
	assert.Contains(t, config.ModelPath, modelsDir)

	// Verify it's an .onnx file
	assert.Equal(t, ".onnx", filepath.Ext(config.ModelPath))

	// Verify it contains the mobile model filename
	assert.Contains(t, config.ModelPath, models.DetectionMobile)
}

func TestConfig_UpdateModelPath_ServerVsMobile(t *testing.T) {
	mobileConfig := Config{UseServerModel: false}
	serverConfig := Config{UseServerModel: true}

	modelsDir := testModelsDir

	mobileConfig.UpdateModelPath(modelsDir)
	serverConfig.UpdateModelPath(modelsDir)

	// Verify mobile and server configs get different paths
	assert.NotEqual(t, mobileConfig.ModelPath, serverConfig.ModelPath)

	// Verify mobile path contains "mobile"
	assert.Contains(t, mobileConfig.ModelPath, "mobile")

	// Verify server path contains "server"
	assert.Contains(t, serverConfig.ModelPath, "server")
}

func TestConfig_UpdateModelPath_Integration(t *testing.T) {
	// Test integration with the models package
	config := Config{
		ModelPath:      "old/path/model.onnx",
		UseServerModel: false,
	}

	modelsDir := "/integration/test"
	config.UpdateModelPath(modelsDir)

	// Verify the result matches what models.GetDetectionModelPath returns
	expectedPath := models.GetDetectionModelPath(modelsDir, false)
	assert.Equal(t, expectedPath, config.ModelPath)

	// Test with server model
	config.UseServerModel = true
	config.UpdateModelPath(modelsDir)

	expectedServerPath := models.GetDetectionModelPath(modelsDir, true)
	assert.Equal(t, expectedServerPath, config.ModelPath)
}

func TestConfig_UpdateModelPath_Idempotent(t *testing.T) {
	config := Config{
		ModelPath:      "old/path/model.onnx",
		UseServerModel: false,
		DbThresh:       0.3,
		DbBoxThresh:    0.5,
	}

	modelsDir := testModelsDir

	// Call UpdateModelPath multiple times
	config.UpdateModelPath(modelsDir)
	firstPath := config.ModelPath
	firstDbThresh := config.DbThresh

	config.UpdateModelPath(modelsDir)
	secondPath := config.ModelPath
	secondDbThresh := config.DbThresh

	// Results should be identical
	assert.Equal(t, firstPath, secondPath)
	assert.InDelta(t, firstDbThresh, secondDbThresh, 1e-6)
}
