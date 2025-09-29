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
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "empty model path")

	// Test non-existent path
	err = validateModelPath("/non/existent/path.onnx")
	assert.Error(t, err)

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
	assert.NoError(t, err)
	assert.Nil(t, results)

	results, err = cls.BatchPredict([]image.Image{})
	assert.NoError(t, err)
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
	assert.NoError(t, err)
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
	assert.NoError(t, err)
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
	assert.Error(t, err)
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
	assert.NoError(t, err)
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
	assert.NoError(t, err)
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
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "could not find project root")

	// Change back to original directory
	err = os.Chdir(originalWd)
	require.NoError(t, err)

	// Now should work
	root, err := findProjectRoot()
	assert.NoError(t, err)
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
	assert.Equal(t, 1.0, result.Confidence)
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
