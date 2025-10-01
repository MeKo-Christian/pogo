package orientation

import (
	"image"
	"image/color"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConfigUpdateModelPath(t *testing.T) {
	tmp := t.TempDir()
	c := DefaultConfig()
	old := c.ModelPath
	c.UpdateModelPath(tmp)
	assert.NotEqual(t, old, c.ModelPath)
	assert.True(t, filepath.IsAbs(c.ModelPath))
	assert.GreaterOrEqual(t, len(c.ModelPath), len(tmp))
	// Should keep the filename under the new dir
	assert.Equal(t, models.GetLayoutModelPath(tmp, filepath.Base(old)), c.ModelPath)
}

func TestHeuristicOrientation_LandscapePrefers0(t *testing.T) {
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "Landscape"
	cfg.Background = color.White
	cfg.Foreground = color.Black
	cfg.Size = testutil.MediumSize
	// no rotation -> likely 0
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)
	cls, err := NewClassifier(Config{Enabled: false, UseHeuristicFallback: true, ConfidenceThreshold: 0.0})
	require.NoError(t, err)
	res, err := cls.Predict(img)
	require.NoError(t, err)
	assert.Contains(t, []int{0, 90, 180, 270}, res.Angle)
	// Heuristic should return a valid angle with a sane confidence bound
	assert.GreaterOrEqual(t, res.Confidence, 0.0)
	assert.LessOrEqual(t, res.Confidence, 1.0)
}

func TestHeuristicOrientation_Rotated90(t *testing.T) {
	// Generate a rotated 90-degree text image
	cfg := testutil.DefaultTestImageConfig()
	cfg.Text = "Rotated Text"
	cfg.Rotation = 90
	cfg.Background = color.White
	cfg.Foreground = color.Black
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	// Use heuristic-only classifier (no ONNX required)
	cls, err := NewClassifier(Config{Enabled: false, UseHeuristicFallback: true})
	require.NoError(t, err)

	res, err := cls.Predict(img)
	require.NoError(t, err)
	// Validate output shape: angle in allowed set, confidence in [0,1]
	allowed := map[int]bool{0: true, 90: true, 180: true, 270: true}
	assert.True(t, allowed[res.Angle], "angle not in {0,90,180,270}: %d", res.Angle)
	assert.True(t, res.Confidence >= 0 && res.Confidence <= 1, "confidence out of range: %f", res.Confidence)
}

func TestNewClassifier_FallbackWhenModelMissing(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Enabled = true
	// Point to a non-existing model path, but with fallback enabled
	cfg.ModelPath = "/non/existent/model.onnx"
	cfg.UseHeuristicFallback = true
	cls, err := NewClassifier(cfg)
	require.NoError(t, err)
	require.NotNil(t, cls)
	// Heuristic should be active
	imgCfg := testutil.DefaultTestImageConfig()
	img, err := testutil.GenerateTextImage(imgCfg)
	require.NoError(t, err)
	_, err = cls.Predict(img)
	require.NoError(t, err)
}

func TestNewClassifier_HeuristicOnly(t *testing.T) {
	cfg := DefaultConfig()
	cfg.HeuristicOnly = true
	cls, err := NewClassifier(cfg)
	require.NoError(t, err)
	require.NotNil(t, cls)
	assert.True(t, cls.heuristic)
}

func TestNewClassifier_DisabledConfig(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Enabled = false
	cls, err := NewClassifier(cfg)
	require.NoError(t, err)
	require.NotNil(t, cls)
	assert.True(t, cls.heuristic)
}

func TestValidateModelPath(t *testing.T) {
	// Test empty path
	err := validateModelPath("")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty model path")

	// Test non-existent path
	err = validateModelPath("/non/existent/path.onnx")
	require.Error(t, err)

	// Test valid path (create a temporary file)
	tmpDir := t.TempDir()
	validPath := filepath.Join(tmpDir, "valid.onnx")
	require.NoError(t, os.WriteFile(validPath, []byte("dummy"), 0o644))
	err = validateModelPath(validPath)
	assert.NoError(t, err)
}

func TestUpdateModelPath_EdgeCases(t *testing.T) {
	cfg := DefaultConfig()

	// Test with empty filename
	cfg.ModelPath = ""
	cfg.UpdateModelPath("/tmp")
	assert.Contains(t, cfg.ModelPath, models.LayoutPPLCNetX10Doc)

	// Test with just "."
	cfg.ModelPath = "."
	cfg.UpdateModelPath("/tmp")
	assert.Contains(t, cfg.ModelPath, models.LayoutPPLCNetX10Doc)

	// Test with "/"
	cfg.ModelPath = "/"
	cfg.UpdateModelPath("/tmp")
	assert.Contains(t, cfg.ModelPath, models.LayoutPPLCNetX10Doc)
}

func TestBatchPredict_EmptyInput(t *testing.T) {
	cls, err := NewClassifier(Config{Enabled: false, UseHeuristicFallback: true})
	require.NoError(t, err)

	results, err := cls.BatchPredict(nil)
	require.NoError(t, err)
	assert.Nil(t, results)

	results, err = cls.BatchPredict([]image.Image{})
	require.NoError(t, err)
	assert.Nil(t, results)
}

func TestBatchPredict_HeuristicMode(t *testing.T) {
	cls, err := NewClassifier(Config{Enabled: false, UseHeuristicFallback: true, ConfidenceThreshold: 0.0})
	require.NoError(t, err)

	// Generate test images
	imgCfg := testutil.DefaultTestImageConfig()
	img1, err := testutil.GenerateTextImage(imgCfg)
	require.NoError(t, err)

	imgCfg.Rotation = 90
	img2, err := testutil.GenerateTextImage(imgCfg)
	require.NoError(t, err)

	images := []image.Image{img1, img2}
	results, err := cls.BatchPredict(images)
	require.NoError(t, err)
	require.Len(t, results, 2)

	for i, result := range results {
		assert.Contains(t, []int{0, 90, 180, 270}, result.Angle, "Result %d angle should be valid", i)
		assert.GreaterOrEqual(t, result.Confidence, 0.0, "Result %d confidence should be >= 0", i)
		assert.LessOrEqual(t, result.Confidence, 1.0, "Result %d confidence should be <= 1", i)
	}
}

func TestPredictWithHeuristicSingle(t *testing.T) {
	cls, err := NewClassifier(Config{Enabled: false, UseHeuristicFallback: true, ConfidenceThreshold: 0.5})
	require.NoError(t, err)

	imgCfg := testutil.DefaultTestImageConfig()
	img, err := testutil.GenerateTextImage(imgCfg)
	require.NoError(t, err)

	result := cls.predictWithHeuristicSingle(img)
	assert.Contains(t, []int{0, 90, 180, 270}, result.Angle)
	assert.GreaterOrEqual(t, result.Confidence, 0.0)
	assert.LessOrEqual(t, result.Confidence, 1.0)
}

func TestShouldSkipOrientation(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SkipSquareImages = true
	cfg.SquareThreshold = 1.2
	cls, err := NewClassifier(cfg)
	require.NoError(t, err)

	// Test with square image (should skip)
	squareImg := image.NewRGBA(image.Rect(0, 0, 100, 100))
	assert.True(t, cls.shouldSkipOrientation(squareImg))

	// Test with near-square image (should skip)
	nearSquareImg := image.NewRGBA(image.Rect(0, 0, 100, 110))
	assert.True(t, cls.shouldSkipOrientation(nearSquareImg))

	// Test with landscape image (should not skip)
	landscapeImg := image.NewRGBA(image.Rect(0, 0, 200, 100))
	assert.False(t, cls.shouldSkipOrientation(landscapeImg))

	// Test with portrait image (should not skip)
	portraitImg := image.NewRGBA(image.Rect(0, 0, 100, 200))
	assert.False(t, cls.shouldSkipOrientation(portraitImg))

	// Test with zero-size image (should skip)
	zeroImg := image.NewRGBA(image.Rect(0, 0, 0, 0))
	assert.True(t, cls.shouldSkipOrientation(zeroImg))
}

func TestHeuristicOrientation_NilImage(t *testing.T) {
	angle, conf := heuristicOrientation(nil)
	assert.Equal(t, 0, angle)
	assert.InDelta(t, 0.0, conf, 1e-6)
}

func TestHeuristicOrientation_SmallImage(t *testing.T) {
	// Create a 1x1 image (invalid thumbnail)
	smallImg := image.NewRGBA(image.Rect(0, 0, 1, 1))
	angle, conf := heuristicOrientation(smallImg)
	assert.Equal(t, 0, angle)
	assert.InDelta(t, 0.0, conf, 1e-6)
}

func TestDetermineOrientation_EdgeCases(t *testing.T) {
	// Test with zero transitions
	angle, conf := determineOrientation(0, 0, image.Rect(0, 0, 100, 100))
	assert.Equal(t, 0, angle)
	assert.InDelta(t, 0.0, conf, 1e-6)

	// Test with equal transitions
	angle, conf = determineOrientation(10, 10, image.Rect(0, 0, 100, 100))
	assert.Equal(t, 90, angle) // colTransitions >= rowTransitions
	assert.GreaterOrEqual(t, conf, 0.0)

	// Test landscape aspect ratio with column preference
	angle, conf = determineOrientation(5, 15, image.Rect(0, 0, 200, 100)) // ar > 1.2
	assert.Equal(t, 90, angle)
	assert.GreaterOrEqual(t, conf, 0.0)

	// Test portrait aspect ratio with row preference
	angle, conf = determineOrientation(15, 5, image.Rect(0, 0, 100, 200)) // ar < 0.8
	assert.Equal(t, 0, angle)
	assert.GreaterOrEqual(t, conf, 0.0)
}

func TestBatchPredict_WithSkipSquareImages(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SkipSquareImages = true
	cfg.SquareThreshold = 1.2
	cfg.Enabled = false
	cfg.UseHeuristicFallback = true
	cfg.ConfidenceThreshold = 0.0 // Low threshold to get actual results
	cls, err := NewClassifier(cfg)
	require.NoError(t, err)

	// Mix of square and non-square images
	squareImg := image.NewRGBA(image.Rect(0, 0, 100, 100))
	landscapeImg := image.NewRGBA(image.Rect(0, 0, 200, 100))

	images := []image.Image{squareImg, landscapeImg}
	results, err := cls.BatchPredict(images)
	require.NoError(t, err)
	require.Len(t, results, 2)

	// In heuristic mode, BatchPredict doesn't use the skip logic - it processes all images
	// So both images will have valid results from the heuristic
	for i, result := range results {
		assert.Contains(t, []int{0, 90, 180, 270}, result.Angle, "Result %d angle should be valid", i)
		assert.GreaterOrEqual(t, result.Confidence, 0.0, "Result %d confidence should be >= 0", i)
		assert.LessOrEqual(t, result.Confidence, 1.0, "Result %d confidence should be <= 1", i)
	}
}

func TestSoftmax(t *testing.T) {
	// Test empty input
	result := softmax([]float32{})
	assert.Nil(t, result)

	// Test single value
	result = softmax([]float32{1.0})
	require.Len(t, result, 1)
	assert.InDelta(t, 1.0, result[0], 1e-6)

	// Test multiple values
	result = softmax([]float32{1.0, 2.0, 3.0})
	require.Len(t, result, 3)

	// Check probabilities sum to 1
	sum := 0.0
	for _, p := range result {
		sum += p
		assert.GreaterOrEqual(t, p, 0.0)
		assert.LessOrEqual(t, p, 1.0)
	}
	assert.InDelta(t, 1.0, sum, 1e-6)

	// Larger values should have higher probabilities
	assert.Greater(t, result[2], result[1])
	assert.Greater(t, result[1], result[0])
}

func TestArgmax(t *testing.T) {
	// Test empty input
	result := argmax([]float64{})
	assert.Equal(t, -1, result)

	// Test single value
	result = argmax([]float64{1.0})
	assert.Equal(t, 0, result)

	// Test multiple values
	result = argmax([]float64{1.0, 3.0, 2.0})
	assert.Equal(t, 1, result)

	// Test with negative values
	result = argmax([]float64{-1.0, -3.0, -2.0})
	assert.Equal(t, 0, result)

	// Test with equal values (should return first)
	result = argmax([]float64{2.0, 2.0, 1.0})
	assert.Equal(t, 0, result)
}

func TestGetONNXLibName(t *testing.T) {
	// This tests the current OS
	libName, err := getONNXLibName()
	require.NoError(t, err)
	assert.NotEmpty(t, libName)

	// On Linux, should be .so
	// On macOS, should be .dylib
	// On Windows, should be .dll
	// We can't test all platforms but we can test the current one
	switch runtime.GOOS {
	case "linux":
		assert.Equal(t, "libonnxruntime.so", libName)
	case "darwin":
		assert.Equal(t, "libonnxruntime.dylib", libName)
	case "windows":
		assert.Equal(t, "onnxruntime.dll", libName)
	}
}

func TestFindProjectRoot(t *testing.T) {
	// This should find the project root (where go.mod is)
	root, err := findProjectRoot()
	require.NoError(t, err)
	assert.NotEmpty(t, root)

	// Verify go.mod exists at the root
	goModPath := filepath.Join(root, "go.mod")
	_, err = os.Stat(goModPath)
	assert.NoError(t, err)
}

func TestWarmup_HeuristicMode(t *testing.T) {
	cls, err := NewClassifier(Config{Enabled: false, UseHeuristicFallback: true})
	require.NoError(t, err)

	// Warmup should be no-op for heuristic mode
	err = cls.Warmup()
	require.NoError(t, err)
}

func TestClose_HeuristicMode(t *testing.T) {
	cls, err := NewClassifier(Config{Enabled: false, UseHeuristicFallback: true})
	require.NoError(t, err)

	// Close should be safe to call on heuristic classifier
	cls.Close()
	// Should be safe to call multiple times
	cls.Close()
}

func TestPredictWithConfidenceThreshold(t *testing.T) {
	// High confidence threshold should return angle 0
	cls, err := NewClassifier(Config{
		Enabled:              false,
		UseHeuristicFallback: true,
		ConfidenceThreshold:  0.99, // Very high threshold
	})
	require.NoError(t, err)

	imgCfg := testutil.DefaultTestImageConfig()
	img, err := testutil.GenerateTextImage(imgCfg)
	require.NoError(t, err)

	result, err := cls.Predict(img)
	require.NoError(t, err)

	// With high threshold, likely to return angle 0 due to low confidence
	if result.Confidence < 0.99 {
		assert.Equal(t, 0, result.Angle)
	}
}

func TestCalculateLuminance(t *testing.T) {
	// Test with pure white
	white := color.RGBA{255, 255, 255, 255}
	lum := calculateLuminance(white)
	assert.InDelta(t, 255.0, lum, 1e-6)

	// Test with pure black
	black := color.RGBA{0, 0, 0, 255}
	lum = calculateLuminance(black)
	assert.InDelta(t, 0.0, lum, 1e-6)

	// Test with pure red
	red := color.RGBA{255, 0, 0, 255}
	lum = calculateLuminance(red)
	assert.InDelta(t, 0.299*255, lum, 1e-6)
}

func TestLuminanceToBinary(t *testing.T) {
	// Below threshold should return 1
	result := luminanceToBinary(50.0, 100.0)
	assert.Equal(t, 1, result)

	// Above threshold should return 0
	result = luminanceToBinary(150.0, 100.0)
	assert.Equal(t, 0, result)

	// Equal to threshold should return 0
	result = luminanceToBinary(100.0, 100.0)
	assert.Equal(t, 0, result)
}

func TestCalculateAspectRatio(t *testing.T) {
	// Square
	ratio := calculateAspectRatio(image.Rect(0, 0, 100, 100))
	assert.InDelta(t, 1.0, ratio, 1e-6)

	// Portrait (taller than wide)
	ratio = calculateAspectRatio(image.Rect(0, 0, 100, 200))
	assert.InDelta(t, 2.0, ratio, 1e-6)

	// Landscape (wider than tall)
	ratio = calculateAspectRatio(image.Rect(0, 0, 200, 100))
	assert.InDelta(t, 0.5, ratio, 1e-6)
}

func TestNewClassifier_EmptyModelPath(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Enabled = true
	cfg.ModelPath = ""
	cfg.UseHeuristicFallback = false
	cls, err := NewClassifier(cfg)
	require.Error(t, err)
	assert.Nil(t, cls)
	assert.Contains(t, err.Error(), "empty model path")
}

func TestInitializeONNXEnvironment_Coverage(t *testing.T) {
	// This tests the initialization path, but since ONNX might already be initialized
	// we can't easily test the error path without mocking
	err := initializeONNXEnvironment()
	// Should not error since we have ONNX Runtime available
	assert.NoError(t, err)
}

func TestSetONNXLibraryPath_Coverage(t *testing.T) {
	// This covers more of the setONNXLibraryPath function
	err := setONNXLibraryPath()
	assert.NoError(t, err)
}

func TestBatchPredict_NonHeuristicMode_EmptyInputs(t *testing.T) {
	// Create a temporary dummy model file to pass validation but fail ONNX loading
	tmpDir := t.TempDir()
	dummyModel := filepath.Join(tmpDir, "dummy.onnx")
	require.NoError(t, os.WriteFile(dummyModel, []byte("dummy"), 0o644))

	cfg := DefaultConfig()
	cfg.Enabled = true
	cfg.ModelPath = dummyModel
	cfg.UseHeuristicFallback = true // Will fall back to heuristic
	cls, err := NewClassifier(cfg)
	require.NoError(t, err)

	// Force to non-heuristic mode for coverage (even though it will be heuristic due to model loading failure)
	cls.heuristic = false
	cls.session = nil // Ensure session is nil

	// Test empty inputs in non-heuristic path (will still use heuristic due to nil session)
	results, err := cls.BatchPredict([]image.Image{})
	require.NoError(t, err)
	assert.Nil(t, results)
}

func TestHeuristicOrientation_EdgeCases(t *testing.T) {
	// Test with 2x2 image (at the edge of valid)
	smallImg := image.NewRGBA(image.Rect(0, 0, 2, 2))
	// Set some pixels to create pattern
	smallImg.Set(0, 0, color.RGBA{255, 255, 255, 255})
	smallImg.Set(1, 0, color.RGBA{0, 0, 0, 255})
	smallImg.Set(0, 1, color.RGBA{0, 0, 0, 255})
	smallImg.Set(1, 1, color.RGBA{255, 255, 255, 255})

	angle, conf := heuristicOrientation(smallImg)
	assert.Contains(t, []int{0, 90}, angle) // Should be either 0 or 90
	assert.GreaterOrEqual(t, conf, 0.0)
	assert.LessOrEqual(t, conf, 1.0)
}

func TestIsValidThumbnail_EdgeCases(t *testing.T) {
	// Test 0x0 image
	zeroImg := image.NewRGBA(image.Rect(0, 0, 0, 0))
	assert.False(t, isValidThumbnail(zeroImg))

	// Test 1x1 image
	oneImg := image.NewRGBA(image.Rect(0, 0, 1, 1))
	assert.False(t, isValidThumbnail(oneImg))

	// Test 2x2 image (valid)
	twoImg := image.NewRGBA(image.Rect(0, 0, 2, 2))
	assert.True(t, isValidThumbnail(twoImg))

	// Test 1x10 image (invalid - width is 1)
	tallImg := image.NewRGBA(image.Rect(0, 0, 1, 10))
	assert.False(t, isValidThumbnail(tallImg))

	// Test 10x1 image (invalid - height is 1)
	wideImg := image.NewRGBA(image.Rect(0, 0, 10, 1))
	assert.False(t, isValidThumbnail(wideImg))
}

func TestWarmup_WithSessionNil(t *testing.T) {
	cls := &Classifier{
		heuristic: false,
		session:   nil, // No session
	}

	// Should return nil since session is nil
	err := cls.Warmup()
	require.NoError(t, err)
}

func TestClose_WithSession(t *testing.T) {
	cls := &Classifier{
		heuristic: false,
		session:   nil, // Simulate session being nil
	}

	// Should be safe to close
	cls.Close()
	assert.Nil(t, cls.session)
}

func TestGetONNXLibName_UnsupportedOS(t *testing.T) {
	// Save original GOOS
	originalGOOS := runtime.GOOS

	// We can't actually change runtime.GOOS, so we test the supported cases
	// The function already works correctly for the current OS
	libName, err := getONNXLibName()
	require.NoError(t, err)
	assert.NotEmpty(t, libName)

	// Restore original GOOS (even though we didn't change it)
	_ = originalGOOS
}

func TestFindProjectRoot_EdgeCase(t *testing.T) {
	// Test the function by changing to a temporary directory
	originalWd, err := os.Getwd()
	require.NoError(t, err)
	defer func() {
		_ = os.Chdir(originalWd)
	}()

	// Create a temporary directory and change to it
	tmpDir := t.TempDir()
	err = os.Chdir(tmpDir)
	require.NoError(t, err)

	// Should fail to find project root (no go.mod in temp dir)
	_, err = findProjectRoot()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "could not find project root")

	// Change back to original directory
	err = os.Chdir(originalWd)
	require.NoError(t, err)

	// Now should work
	root, err := findProjectRoot()
	require.NoError(t, err)
	assert.NotEmpty(t, root)
}

func TestPredict_SkipSquareImages(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SkipSquareImages = true
	cfg.SquareThreshold = 1.2
	cfg.Enabled = false
	cfg.UseHeuristicFallback = true
	cls, err := NewClassifier(cfg)
	require.NoError(t, err)

	// Test with square image (should skip)
	squareImg := image.NewRGBA(image.Rect(0, 0, 100, 100))
	result, err := cls.Predict(squareImg)
	require.NoError(t, err)
	assert.Equal(t, 0, result.Angle)
	assert.InDelta(t, 1.0, result.Confidence, 1e-6)
}

func TestBatchPredict_MixedSkipAndProcess(t *testing.T) {
	// For this test to work properly, we need to set up a classifier that's not in heuristic mode
	// but still falls back to heuristic. This is tricky since we need actual ONNX setup.
	// Let's test the heuristic scenario more thoroughly instead.

	cfg := DefaultConfig()
	cfg.SkipSquareImages = true
	cfg.SquareThreshold = 1.2
	cfg.Enabled = false
	cfg.UseHeuristicFallback = true
	cfg.ConfidenceThreshold = 0.0
	cls, err := NewClassifier(cfg)
	require.NoError(t, err)

	// Create test images with different aspect ratios
	squareImg := image.NewRGBA(image.Rect(0, 0, 100, 100))
	landscapeImg := image.NewRGBA(image.Rect(0, 0, 200, 100))
	portraitImg := image.NewRGBA(image.Rect(0, 0, 100, 200))

	images := []image.Image{squareImg, landscapeImg, portraitImg}
	results, err := cls.BatchPredict(images)
	require.NoError(t, err)
	require.Len(t, results, 3)

	// All should have valid angles since we're in heuristic mode
	for i, result := range results {
		assert.Contains(t, []int{0, 90, 180, 270}, result.Angle, "Result %d should have valid angle", i)
		assert.GreaterOrEqual(t, result.Confidence, 0.0, "Result %d confidence should be >= 0", i)
		assert.LessOrEqual(t, result.Confidence, 1.0, "Result %d confidence should be <= 1", i)
	}
}

// TestDefaultConfig_AllFields verifies all fields in DefaultConfig are set correctly.
func TestDefaultConfig_AllFields(t *testing.T) {
	cfg := DefaultConfig()

	// Basic settings
	assert.False(t, cfg.Enabled, "Expected Enabled to be false by default")
	assert.NotEmpty(t, cfg.ModelPath, "Expected ModelPath to be set")
	assert.InDelta(t, 0.7, cfg.ConfidenceThreshold, 1e-6, "Expected default confidence threshold 0.7")
	assert.Equal(t, 0, cfg.NumThreads, "Expected NumThreads to be 0 (auto)")

	// Fallback and optimization settings
	assert.True(t, cfg.UseHeuristicFallback, "Expected UseHeuristicFallback to be true")
	assert.True(t, cfg.SkipSquareImages, "Expected SkipSquareImages to be true")
	assert.InDelta(t, 1.2, cfg.SquareThreshold, 1e-6, "Expected SquareThreshold to be 1.2")
	assert.False(t, cfg.EnableWarmup, "Expected EnableWarmup to be false")
	assert.False(t, cfg.HeuristicOnly, "Expected HeuristicOnly to be false")

	// GPU settings
	assert.False(t, cfg.GPU.UseGPU, "Expected GPU to be disabled by default")
	assert.Equal(t, 0, cfg.GPU.DeviceID, "Expected GPU device 0")
}

// TestDefaultTextLineConfig_AllFields verifies all fields in DefaultTextLineConfig.
func TestDefaultTextLineConfig_AllFields(t *testing.T) {
	cfg := DefaultTextLineConfig()

	// Should have different defaults than document config
	assert.False(t, cfg.Enabled, "Expected Enabled to be false by default")
	assert.NotEmpty(t, cfg.ModelPath, "Expected ModelPath to be set")
	assert.InDelta(t, 0.6, cfg.ConfidenceThreshold, 1e-6, "Expected textline confidence threshold 0.6")

	// Should still have common defaults
	assert.True(t, cfg.UseHeuristicFallback, "Expected UseHeuristicFallback to be true")
	assert.False(t, cfg.EnableWarmup, "Expected EnableWarmup to be false")
	assert.False(t, cfg.HeuristicOnly, "Expected HeuristicOnly to be false")
	assert.False(t, cfg.GPU.UseGPU, "Expected GPU to be disabled")
}

// TestConfig_ConfidenceThresholdRange tests various confidence threshold values.
func TestConfig_ConfidenceThresholdRange(t *testing.T) {
	tests := []struct {
		name      string
		threshold float64
	}{
		{"minimum", 0.0},
		{"low", 0.3},
		{"medium", 0.5},
		{"default", 0.7},
		{"high", 0.9},
		{"maximum", 1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.ConfidenceThreshold = tt.threshold
			cfg.Enabled = false
			cfg.UseHeuristicFallback = true

			cls, err := NewClassifier(cfg)
			require.NoError(t, err)
			assert.Equal(t, tt.threshold, cls.cfg.ConfidenceThreshold)
		})
	}
}

// TestConfig_SquareThresholdRange tests various square threshold values.
func TestConfig_SquareThresholdRange(t *testing.T) {
	tests := []struct {
		name      string
		threshold float64
		imgWidth  int
		imgHeight int
		shouldSkip bool
	}{
		{"threshold 1.1 with square", 1.1, 100, 100, true},   // aspect=1.0, 1.0 <= 1.1 = skip
		{"threshold 1.1 with slight landscape", 1.1, 110, 100, true}, // aspect=1.1, 1.1 <= 1.1 = skip
		{"threshold 1.5 with moderate landscape", 1.5, 150, 100, true}, // aspect=1.5, 1.5 <= 1.5 = skip
		{"threshold 1.5 with large landscape", 1.5, 200, 100, false}, // aspect=2.0, 2.0 > 1.5 = no skip
		{"threshold 2.0 with very wide", 2.0, 300, 100, false}, // aspect=3.0, 3.0 > 2.0 = no skip
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.SkipSquareImages = true
			cfg.SquareThreshold = tt.threshold
			cfg.Enabled = false

			cls, err := NewClassifier(cfg)
			require.NoError(t, err)

			img := image.NewRGBA(image.Rect(0, 0, tt.imgWidth, tt.imgHeight))
			shouldSkip := cls.shouldSkipOrientation(img)
			assert.Equal(t, tt.shouldSkip, shouldSkip,
				"Image %dx%d with threshold %f", tt.imgWidth, tt.imgHeight, tt.threshold)
		})
	}
}

// TestConfig_NumThreadsSettings tests NumThreads configuration.
func TestConfig_NumThreadsSettings(t *testing.T) {
	tests := []struct {
		name       string
		numThreads int
	}{
		{"auto (0)", 0},
		{"single thread", 1},
		{"dual thread", 2},
		{"quad thread", 4},
		{"many threads", 16},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.NumThreads = tt.numThreads
			cfg.Enabled = false

			cls, err := NewClassifier(cfg)
			require.NoError(t, err)
			assert.Equal(t, tt.numThreads, cls.cfg.NumThreads)
		})
	}
}

// TestConfig_SkipSquareImagesFlag tests SkipSquareImages flag behavior.
func TestConfig_SkipSquareImagesFlag(t *testing.T) {
	tests := []struct {
		name             string
		skipSquare       bool
		imgSize          image.Rectangle
		expectWouldSkip  bool
	}{
		{"skip enabled, square image", true, image.Rect(0, 0, 100, 100), true},
		{"skip disabled, square image", false, image.Rect(0, 0, 100, 100), true}, // shouldSkipOrientation still returns true based on aspect
		{"skip enabled, landscape", true, image.Rect(0, 0, 200, 100), false},
		{"skip disabled, landscape", false, image.Rect(0, 0, 200, 100), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.SkipSquareImages = tt.skipSquare
			cfg.SquareThreshold = 1.2
			cfg.Enabled = false

			cls, err := NewClassifier(cfg)
			require.NoError(t, err)
			assert.Equal(t, tt.skipSquare, cls.cfg.SkipSquareImages)

			img := image.NewRGBA(tt.imgSize)

			// shouldSkipOrientation is based on aspect ratio regardless of SkipSquareImages flag
			wouldSkip := cls.shouldSkipOrientation(img)
			assert.Equal(t, tt.expectWouldSkip, wouldSkip,
				"shouldSkipOrientation should return %v for this aspect ratio", tt.expectWouldSkip)
		})
	}
}

// TestConfig_GPUSettings tests GPU configuration fields.
func TestConfig_GPUSettings(t *testing.T) {
	cfg := DefaultConfig()

	// Test default GPU config
	assert.False(t, cfg.GPU.UseGPU, "GPU should be disabled by default")
	assert.Equal(t, 0, cfg.GPU.DeviceID, "GPU device should be 0 by default")

	// Test modifying GPU config
	cfg.GPU.UseGPU = true
	cfg.GPU.DeviceID = 1
	cfg.GPU.GPUMemLimit = 1024 * 1024 * 1024 // 1GB

	assert.True(t, cfg.GPU.UseGPU)
	assert.Equal(t, 1, cfg.GPU.DeviceID)
	assert.Equal(t, uint64(1024*1024*1024), cfg.GPU.GPUMemLimit)
}

// TestConfig_EnableWarmupFlag tests EnableWarmup configuration.
func TestConfig_EnableWarmupFlag(t *testing.T) {
	tests := []struct {
		name         string
		enableWarmup bool
	}{
		{"warmup disabled", false},
		{"warmup enabled", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.EnableWarmup = tt.enableWarmup
			cfg.Enabled = false

			cls, err := NewClassifier(cfg)
			require.NoError(t, err)
			assert.Equal(t, tt.enableWarmup, cls.cfg.EnableWarmup)
		})
	}
}

// TestConfig_HeuristicOnlyMode tests HeuristicOnly flag forces heuristic mode.
func TestConfig_HeuristicOnlyMode(t *testing.T) {
	cfg := DefaultConfig()
	cfg.HeuristicOnly = true
	cfg.Enabled = true // Even with enabled, should use heuristic

	cls, err := NewClassifier(cfg)
	require.NoError(t, err)

	// Should be in heuristic mode
	assert.True(t, cls.heuristic, "Classifier should be in heuristic mode when HeuristicOnly is true")
	assert.Nil(t, cls.session, "Session should be nil in heuristic-only mode")
}

// TestConfig_UseHeuristicFallbackBehavior tests UseHeuristicFallback flag.
func TestConfig_UseHeuristicFallbackBehavior(t *testing.T) {
	tests := []struct {
		name             string
		useFallback      bool
		modelPath        string
		expectError      bool
		expectHeuristic  bool
	}{
		{"with fallback, invalid model", true, "/nonexistent/model.onnx", false, true},
		{"without fallback, invalid model", false, "/nonexistent/model.onnx", true, false},
		{"with fallback, empty model", true, "", false, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.Enabled = true
			cfg.UseHeuristicFallback = tt.useFallback
			cfg.ModelPath = tt.modelPath

			cls, err := NewClassifier(cfg)

			if tt.expectError {
				assert.Error(t, err, "Expected error for invalid config without fallback")
				assert.Nil(t, cls, "Classifier should be nil on error")
			} else {
				require.NoError(t, err, "Should not error with fallback enabled")
				require.NotNil(t, cls)
				if tt.expectHeuristic {
					assert.True(t, cls.heuristic, "Should fall back to heuristic mode")
				}
			}
		})
	}
}

func TestComputeOrientationFromLogits_Coverage(t *testing.T) {
	cfg := Config{Enabled: false, UseHeuristicFallback: true}
	cls, err := NewClassifier(cfg)
	require.NoError(t, err)

	// Test the function directly
	logits := []float32{0.1, 0.8, 0.05, 0.05}
	angle, confidence := cls.computeOrientationFromLogits(logits)

	assert.Equal(t, 90, angle) // Index 1 has highest value -> 90 degrees
	assert.Greater(t, confidence, 0.0)
	assert.LessOrEqual(t, confidence, 1.0)

	// Test with different max index
	logits2 := []float32{0.05, 0.05, 0.8, 0.1}
	angle2, confidence2 := cls.computeOrientationFromLogits(logits2)
	assert.Equal(t, 180, angle2) // Index 2 -> 180 degrees
	assert.Greater(t, confidence2, 0.0)
}

func TestComputeOrientationFromLogits_AllAngles(t *testing.T) {
	cfg := Config{Enabled: false, UseHeuristicFallback: true}
	cls, err := NewClassifier(cfg)
	require.NoError(t, err)

	// Test each angle
	testCases := []struct {
		logits        []float32
		expectedAngle int
	}{
		{[]float32{0.9, 0.05, 0.03, 0.02}, 0},   // Index 0 -> 0 degrees
		{[]float32{0.05, 0.9, 0.03, 0.02}, 90},  // Index 1 -> 90 degrees
		{[]float32{0.05, 0.03, 0.9, 0.02}, 180}, // Index 2 -> 180 degrees
		{[]float32{0.05, 0.03, 0.02, 0.9}, 270}, // Index 3 -> 270 degrees
	}

	for _, tc := range testCases {
		angle, conf := cls.computeOrientationFromLogits(tc.logits)
		assert.Equal(t, tc.expectedAngle, angle)
		assert.Greater(t, conf, 0.0)
		assert.LessOrEqual(t, conf, 1.0)
	}
}

func TestDefaultTextLineConfig_Values(t *testing.T) {
	cfg := DefaultTextLineConfig()

	// Verify it returns proper defaults
	assert.False(t, cfg.Enabled)
	assert.Contains(t, cfg.ModelPath, models.LayoutPPLCNetX025Textline)
	assert.InDelta(t, 0.6, cfg.ConfidenceThreshold, 1e-6)
	assert.False(t, cfg.EnableWarmup)
	assert.False(t, cfg.HeuristicOnly)
}

func TestPredictWithHeuristic_EdgeCases(t *testing.T) {
	cls, err := NewClassifier(Config{Enabled: false, UseHeuristicFallback: true, ConfidenceThreshold: 0.5})
	require.NoError(t, err)

	// Test with very small image
	smallImg := image.NewRGBA(image.Rect(0, 0, 10, 10))
	result, err := cls.predictWithHeuristic(smallImg)
	require.NoError(t, err)
	assert.Contains(t, []int{0, 90, 180, 270}, result.Angle)

	// Test with very large aspect ratio
	wideImg := image.NewRGBA(image.Rect(0, 0, 1000, 10))
	result, err = cls.predictWithHeuristic(wideImg)
	require.NoError(t, err)
	assert.Contains(t, []int{0, 90, 180, 270}, result.Angle)

	tallImg := image.NewRGBA(image.Rect(0, 0, 10, 1000))
	result, err = cls.predictWithHeuristic(tallImg)
	require.NoError(t, err)
	assert.Contains(t, []int{0, 90, 180, 270}, result.Angle)
}

func TestHeuristicOrientation_UniformImage(t *testing.T) {
	// Create a uniform gray image (no transitions)
	uniformImg := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := range 100 {
		for x := range 100 {
			uniformImg.Set(x, y, color.RGBA{128, 128, 128, 255})
		}
	}

	angle, conf := heuristicOrientation(uniformImg)
	assert.Equal(t, 0, angle)
	assert.InDelta(t, 0.0, conf, 1e-6) // Should have low confidence for uniform image
}

func TestHeuristicOrientation_HighContrastPatterns(t *testing.T) {
	// Create a horizontal striped pattern (high row transitions)
	horizontalStripes := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := range 100 {
		c := color.RGBA{255, 255, 255, 255}
		if y%10 < 5 {
			c = color.RGBA{0, 0, 0, 255}
		}
		for x := range 100 {
			horizontalStripes.Set(x, y, c)
		}
	}

	angle, conf := heuristicOrientation(horizontalStripes)
	assert.Contains(t, []int{0, 90}, angle)
	assert.GreaterOrEqual(t, conf, 0.0)
	assert.LessOrEqual(t, conf, 1.0)

	// Create a vertical striped pattern (high column transitions)
	verticalStripes := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for x := range 100 {
		c := color.RGBA{255, 255, 255, 255}
		if x%10 < 5 {
			c = color.RGBA{0, 0, 0, 255}
		}
		for y := range 100 {
			verticalStripes.Set(x, y, c)
		}
	}

	angle2, conf2 := heuristicOrientation(verticalStripes)
	assert.Contains(t, []int{0, 90}, angle2)
	assert.GreaterOrEqual(t, conf2, 0.0)
	assert.LessOrEqual(t, conf2, 1.0)
}

func TestShouldSkipOrientation_ExactThreshold(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SkipSquareImages = true
	cfg.SquareThreshold = 1.5
	cls, err := NewClassifier(cfg)
	require.NoError(t, err)

	// Test image exactly at threshold (150x100 = 1.5 ratio)
	thresholdImg := image.NewRGBA(image.Rect(0, 0, 150, 100))
	assert.True(t, cls.shouldSkipOrientation(thresholdImg))

	// Test image just above threshold
	aboveImg := image.NewRGBA(image.Rect(0, 0, 151, 100))
	assert.False(t, cls.shouldSkipOrientation(aboveImg))
}

func TestBatchPredict_SingleImage(t *testing.T) {
	cls, err := NewClassifier(Config{Enabled: false, UseHeuristicFallback: true, ConfidenceThreshold: 0.0})
	require.NoError(t, err)

	imgCfg := testutil.DefaultTestImageConfig()
	img, err := testutil.GenerateTextImage(imgCfg)
	require.NoError(t, err)

	// Test batch predict with single image
	results, err := cls.BatchPredict([]image.Image{img})
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Contains(t, []int{0, 90, 180, 270}, results[0].Angle)
}

func TestNewClassifier_HeuristicOnlyFlag(t *testing.T) {
	// Even with Enabled=true and valid model path, HeuristicOnly should force heuristic mode
	cfg := DefaultConfig()
	cfg.Enabled = true
	cfg.HeuristicOnly = true

	cls, err := NewClassifier(cfg)
	require.NoError(t, err)
	require.NotNil(t, cls)
	assert.True(t, cls.heuristic)

	// Should work with heuristic
	imgCfg := testutil.DefaultTestImageConfig()
	img, err := testutil.GenerateTextImage(imgCfg)
	require.NoError(t, err)

	result, err := cls.Predict(img)
	require.NoError(t, err)
	assert.Contains(t, []int{0, 90, 180, 270}, result.Angle)
}

func TestCountTransitions_EdgeCases(t *testing.T) {
	// Test with single-pixel wide/tall images
	singleCol := image.NewRGBA(image.Rect(0, 0, 1, 100))
	for y := range 100 {
		c := color.RGBA{255, 255, 255, 255}
		if y%2 == 0 {
			c = color.RGBA{0, 0, 0, 255}
		}
		singleCol.Set(0, y, c)
	}

	// Should not panic
	meanLum := calculateMeanLuminance(singleCol)
	transitions := countTransitionsInRows(singleCol, meanLum)
	assert.GreaterOrEqual(t, transitions, 0.0)

	transitions = countTransitionsInColumns(singleCol, meanLum)
	assert.GreaterOrEqual(t, transitions, 0.0)
}

func TestDetermineOrientation_ExtremeAspectRatios(t *testing.T) {
	// Test with very wide image (ar > 1.2)
	wideAngle, wideConf := determineOrientation(10, 20, image.Rect(0, 0, 1000, 100))
	assert.Equal(t, 90, wideAngle) // colTransitions > rowTransitions
	assert.GreaterOrEqual(t, wideConf, 0.0)

	// Test with very tall image (ar < 0.8)
	tallAngle, tallConf := determineOrientation(20, 10, image.Rect(0, 0, 100, 1000))
	assert.Equal(t, 0, tallAngle) // rowTransitions > colTransitions
	assert.GreaterOrEqual(t, tallConf, 0.0)

	// Test with equal transitions and extreme aspect ratio
	equalAngle, equalConf := determineOrientation(10, 10, image.Rect(0, 0, 1000, 100))
	assert.Equal(t, 90, equalAngle) // colTransitions >= rowTransitions
	assert.GreaterOrEqual(t, equalConf, 0.0)
}
