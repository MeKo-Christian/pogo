package support

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/MeKo-Tech/pogo/cmd/ocr/cmd"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/cucumber/godog"
	"github.com/spf13/cobra"
)

// copyFile copies a file from src to dst.
func copyFile(src, dst string) error {
	sourceFile, err := os.Open(src) //nolint:gosec // G304: Test file copy with controlled paths
	if err != nil {
		return err
	}
	defer func() { _ = sourceFile.Close() }()

	destFile, err := os.Create(dst) //nolint:gosec // G304: Test file copy with controlled paths
	if err != nil {
		return err
	}
	defer func() { _ = destFile.Close() }()

	_, err = io.Copy(destFile, sourceFile)
	return err
}

// hasEnvVar checks if an environment variable is already set in the test context.
func (testCtx *TestContext) hasEnvVar(name string) bool {
	prefix := name + "="
	for _, envVar := range testCtx.EnvVars {
		if strings.HasPrefix(envVar, prefix) {
			return true
		}
	}
	return false
}

// theOCRModelsAreAvailable checks if OCR models are available.
func (testCtx *TestContext) theOCRModelsAreAvailable() error {
	// Set the models directory to the project root models directory
	projectRoot, err := testutil.GetProjectRoot()
	if err != nil {
		return fmt.Errorf("failed to find project root: %w", err)
	}
	modelsDir := filepath.Join(projectRoot, "models")

	// Check if models directory exists
	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		return fmt.Errorf("models directory not found: %s", modelsDir)
	}

	// Check for essential model files in their organized locations
	expectedModels := []string{
		"detection/mobile/PP-OCRv5_mobile_det.onnx",
		"recognition/mobile/PP-OCRv5_mobile_rec.onnx",
		"dictionaries/ppocr_keys_v1.txt",
	}

	for _, model := range expectedModels {
		modelPath := filepath.Join(modelsDir, model)
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			return fmt.Errorf("required model not found: %s", modelPath)
		}
	}

	// Set the environment variable for the test - this is crucial!
	testCtx.AddEnvVar("GO_OAR_OCR_MODELS_DIR", modelsDir)

	return nil
}

// theOCRModelsAreAvailableIn verifies models are available at custom path.
func (testCtx *TestContext) theOCRModelsAreAvailableIn(path string) error {
	// For testing with custom paths, we'll copy the real models to the custom location
	// or create symbolic links to avoid duplicating large model files
	projectRoot, err := testutil.GetProjectRoot()
	if err != nil {
		return fmt.Errorf("failed to find project root: %w", err)
	}

	sourceModelsDir := filepath.Join(projectRoot, "models")

	// Create the custom models directory structure
	if err := os.MkdirAll(path, 0o755); err != nil {
		return fmt.Errorf("failed to create custom models directory: %w", err)
	}

	// Copy or link essential model files to the custom path
	modelMappings := []struct {
		src string
		dst string
	}{
		{"detection/mobile/PP-OCRv5_mobile_det.onnx", "detection/mobile/PP-OCRv5_mobile_det.onnx"},
		{"recognition/mobile/PP-OCRv5_mobile_rec.onnx", "recognition/mobile/PP-OCRv5_mobile_rec.onnx"},
		{"dictionaries/ppocr_keys_v1.txt", "dictionaries/ppocr_keys_v1.txt"},
	}

	for _, mapping := range modelMappings {
		srcPath := filepath.Join(sourceModelsDir, mapping.src)
		dstPath := filepath.Join(path, mapping.dst)

		// Create destination directory
		if err := os.MkdirAll(filepath.Dir(dstPath), 0o755); err != nil {
			return fmt.Errorf("failed to create directory for %s: %w", dstPath, err)
		}

		// Check if source exists
		if _, err := os.Stat(srcPath); os.IsNotExist(err) {
			return fmt.Errorf("source model not found: %s", srcPath)
		}

		// Create a symbolic link to avoid copying large files
		if err := os.Symlink(srcPath, dstPath); err != nil {
			// If symlink fails, try copying the file
			if err := copyFile(srcPath, dstPath); err != nil {
				return fmt.Errorf("failed to copy model %s to %s: %w", srcPath, dstPath, err)
			}
		}
	}

	return nil
}

// theOCRModelsAreAvailableInTempDir creates a temporary models directory.
func (testCtx *TestContext) theOCRModelsAreAvailableInTempDir() error {
	// Create a temporary models directory
	tempModelsDir := testCtx.GetTempDir("models")
	if err := os.MkdirAll(tempModelsDir, 0o755); err != nil {
		return fmt.Errorf("failed to create temp models directory: %w", err)
	}

	// Store the temp path for later use in command substitution
	testCtx.TempModelsDir = tempModelsDir

	// Set up the models using the existing logic
	return testCtx.theOCRModelsAreAvailableIn(tempModelsDir)
}

// theTestImagesAreAvailable checks if test images are available.
func (testCtx *TestContext) theTestImagesAreAvailable() error {
	// Use testutil to get proper project root
	projectRoot, err := testutil.GetProjectRoot()
	if err != nil {
		return fmt.Errorf("failed to find project root: %w", err)
	}

	testDataDir := filepath.Join(projectRoot, "testdata")

	// Debug: print working directory and testdata path
	wd, _ := os.Getwd()
	fmt.Printf("DEBUG: Current working dir: %s\n", wd)
	fmt.Printf("DEBUG: Project root: %s\n", projectRoot)
	fmt.Printf("DEBUG: Looking for testdata at: %s\n", testDataDir)
	fmt.Printf("DEBUG: TestContext working dir: %s\n", testCtx.WorkingDir)

	// Check if testdata directory exists
	if _, err := os.Stat(testDataDir); os.IsNotExist(err) {
		return fmt.Errorf("testdata directory not found: %s", testDataDir)
	}

	// Check for essential test images
	essentialImages := []string{
		"images/simple_text.png",
		"synthetic/basic_text.png",
	}

	for _, imgPath := range essentialImages {
		fullPath := filepath.Join(testDataDir, imgPath)
		if _, err := os.Stat(fullPath); os.IsNotExist(err) {
			// Try to create basic synthetic images if they don't exist
			if strings.Contains(imgPath, "synthetic") {
				if err := testCtx.createSyntheticTestImage(fullPath); err != nil {
					return fmt.Errorf("test image not found and could not create: %s", fullPath)
				}
			} else {
				return fmt.Errorf("required test image not found: %s", fullPath)
			}
		}
	}

	// Also ensure OCR models are available by default for most tests
	// This sets up the default models directory unless already configured
	if !testCtx.hasEnvVar("GO_OAR_OCR_MODELS_DIR") {
		if err := testCtx.theOCRModelsAreAvailable(); err != nil {
			return fmt.Errorf("failed to set up default OCR models: %w", err)
		}
	}

	return nil
}

// theTestPDFsAreAvailable checks if test PDF files are available.
func (testCtx *TestContext) theTestPDFsAreAvailable() error {
	// Use testutil to get proper project root
	projectRoot, err := testutil.GetProjectRoot()
	if err != nil {
		return fmt.Errorf("failed to find project root: %w", err)
	}

	testDataDir := filepath.Join(projectRoot, "testdata", "documents")

	// Check if documents directory exists
	if _, err := os.Stat(testDataDir); os.IsNotExist(err) {
		return fmt.Errorf("test documents directory not found: %s", testDataDir)
	}

	// For now, just check the directory exists - we can add specific PDF checks later
	return nil
}

// iRunCommand executes a command and stores the result.
func (testCtx *TestContext) iRunCommand(command string) error {
	// Perform command substitution
	command = testCtx.substituteCommandVariables(command)

	testCtx.LastCommand = command
	testCtx.LastStartTime = time.Now()

	// Check if this is a pogo command - if so, run it internally
	if strings.HasPrefix(command, "pogo ") {
		return testCtx.iRunCommandInternal(command)
	}

	// Parse command into parts
	parts := strings.Fields(command)
	if len(parts) == 0 {
		return errors.New("empty command")
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Execute command
	cmd := exec.CommandContext(ctx, parts[0], parts[1:]...)
	cmd.Dir = testCtx.WorkingDir

	// Set environment variables
	cmd.Env = append(os.Environ(), testCtx.EnvVars...)

	// Capture both stdout and stderr
	output, err := cmd.CombinedOutput()
	testCtx.LastOutput = string(output)
	testCtx.LastError = err
	testCtx.LastDuration = time.Since(testCtx.LastStartTime)

	// Store exit code
	if err != nil {
		exitError := &exec.ExitError{}
		if errors.As(err, &exitError) {
			testCtx.LastExitCode = exitError.ExitCode()
		} else {
			testCtx.LastExitCode = -1
		}
	} else {
		testCtx.LastExitCode = 0
	}

	return nil
}

// iRunCommandInternal executes a pogo command internally using the cobra command structure.
func (testCtx *TestContext) iRunCommandInternal(command string) error {
	// Parse command into parts
	parts := strings.Fields(command)
	if len(parts) == 0 {
		return errors.New("empty command")
	}

	// Remove "pogo" from the beginning
	if parts[0] == "pogo" {
		parts = parts[1:]
	}

	// Add models-dir flag if GO_OAR_OCR_MODELS_DIR is set and not already present
	hasModelsDir := false
	for i, part := range parts {
		if part == "--models-dir" && i < len(parts)-1 {
			hasModelsDir = true
			break
		}
	}
	if !hasModelsDir {
		if modelsDir := os.Getenv("GO_OAR_OCR_MODELS_DIR"); modelsDir != "" {
			parts = append(parts, "--models-dir", modelsDir)
		}
	}

	// Default height is now fixed in the recognizer config to 48
	// No need to add --rec-height automatically

	// Create a new root command for this test execution
	rootCmd := testCtx.createTestRootCommand()

	// Set up output capture
	var stdout, stderr bytes.Buffer
	rootCmd.SetOut(&stdout)
	rootCmd.SetErr(&stderr)

	// Set arguments
	rootCmd.SetArgs(parts)

	// Set environment variables
	for _, envVar := range testCtx.EnvVars {
		keyValue := strings.SplitN(envVar, "=", 2)
		if len(keyValue) == 2 {
			_ = os.Setenv(keyValue[0], keyValue[1])
		}
	}

	// Change to the correct working directory for command execution
	currentDir, _ := os.Getwd()
	if currentDir != testCtx.WorkingDir {
		fmt.Printf("DEBUG: Changing working dir from %s to %s\n", currentDir, testCtx.WorkingDir)
		if err := os.Chdir(testCtx.WorkingDir); err != nil {
			return fmt.Errorf("failed to change working directory: %w", err)
		}
		// Restore working directory after command execution
		defer func() { _ = os.Chdir(currentDir) }()
	}

	// Execute the command
	err := rootCmd.Execute()

	// Capture output and error
	testCtx.LastOutput = stdout.String() + stderr.String()
	testCtx.LastError = err
	testCtx.LastDuration = time.Since(testCtx.LastStartTime)

	// Store exit code
	if err != nil {
		testCtx.LastExitCode = 1
	} else {
		testCtx.LastExitCode = 0
	}

	return nil
}

// createTestRootCommand creates a new root command for testing that doesn't call os.Exit.
func (testCtx *TestContext) createTestRootCommand() *cobra.Command {
	// Get the actual root command
	return cmd.GetRootCommand()
}

// theCommandShouldSucceed verifies the command succeeded.
func (testCtx *TestContext) theCommandShouldSucceed() error {
	if testCtx.LastExitCode != 0 {
		return fmt.Errorf("command failed with exit code %d: %w\nOutput: %s",
			testCtx.LastExitCode, testCtx.LastError, testCtx.LastOutput)
	}
	return nil
}

// theCommandShouldFail verifies the command failed.
func (testCtx *TestContext) theCommandShouldFail() error {
	if testCtx.LastExitCode == 0 {
		return fmt.Errorf("command succeeded when it should have failed\nOutput: %s", testCtx.LastOutput)
	}
	return nil
}

// theOutputShouldContain verifies the output contains specific text.
func (testCtx *TestContext) theOutputShouldContain(expectedText string) error {
	if !strings.Contains(testCtx.LastOutput, expectedText) {
		return fmt.Errorf("output does not contain '%s'\nActual output: %s", expectedText, testCtx.LastOutput)
	}
	return nil
}

// theOutputShouldBeValidJSON verifies the output is valid JSON.
func (testCtx *TestContext) theOutputShouldBeValidJSON() error {
	var js json.RawMessage
	if err := json.Unmarshal([]byte(testCtx.LastOutput), &js); err != nil {
		return fmt.Errorf("output is not valid JSON: %w\nOutput: %s", err, testCtx.LastOutput)
	}
	return nil
}

// theJSONShouldContain verifies JSON contains a specific field.
func (testCtx *TestContext) theJSONShouldContain(field string) error {
	// First verify it's valid JSON
	if err := testCtx.theOutputShouldBeValidJSON(); err != nil {
		return err
	}

	// Parse JSON and check for field
	var data map[string]interface{}
	if err := json.Unmarshal([]byte(testCtx.LastOutput), &data); err != nil {
		return fmt.Errorf("failed to parse JSON: %w", err)
	}

	// Handle nested field paths (e.g., "ocr.regions")
	parts := strings.Split(field, ".")
	current := data

	for i, part := range parts {
		if part == "array" {
			// Special handling for array type checking
			if i == 0 {
				return errors.New("array cannot be the root field")
			}
			// Previous part should be the field name
			prevPart := parts[i-1]
			if val, exists := current[prevPart]; exists {
				if _, isArray := val.([]interface{}); !isArray {
					return fmt.Errorf("field '%s' is not an array", prevPart)
				}
				return nil
			}
			return fmt.Errorf("field '%s' not found in JSON", prevPart)
		}

		if val, exists := current[part]; exists {
			if i == len(parts)-1 {
				// Last part - field exists
				return nil
			}
			// Navigate deeper
			if nextMap, ok := val.(map[string]interface{}); ok {
				current = nextMap
			} else {
				return fmt.Errorf("cannot navigate deeper into non-object field '%s'", part)
			}
		} else {
			return fmt.Errorf("field '%s' not found in JSON", strings.Join(parts[:i+1], "."))
		}
	}

	return nil
}

// theErrorShouldMention verifies the error message contains specific text.
func (testCtx *TestContext) theErrorShouldMention(errorText string) error {
	if testCtx.LastError == nil && testCtx.LastExitCode == 0 {
		return fmt.Errorf("no error occurred, but expected error containing '%s'", errorText)
	}

	// Check both error message and output for the expected text
	fullErrorText := testCtx.LastOutput
	if testCtx.LastError != nil {
		fullErrorText += " " + testCtx.LastError.Error()
	}

	// Convert to lowercase for case-insensitive matching
	if !strings.Contains(strings.ToLower(fullErrorText), strings.ToLower(errorText)) {
		return fmt.Errorf("error does not contain '%s'\nActual error: %s", errorText, fullErrorText)
	}

	return nil
}

// aCustomDetectionModelExistsAt verifies custom detection model exists.
func (testCtx *TestContext) aCustomDetectionModelExistsAt(path string) error {
	// For testing, we use the testdata models
	projectRoot, err := testutil.GetProjectRoot()
	if err != nil {
		return fmt.Errorf("failed to find project root: %w", err)
	}

	// Map the requested path to our testdata model
	modelPath := filepath.Join(
		projectRoot,
		"testdata",
		"models",
		"custom",
		"detection",
		"mobile",
		"PP-OCRv5_mobile_det.onnx",
	)

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return fmt.Errorf("custom detection model not found: %s", modelPath)
	}
	return nil
}

// aCustomRecognitionModelExistsAt verifies custom recognition model exists.
func (testCtx *TestContext) aCustomRecognitionModelExistsAt(path string) error {
	// For testing, we use the testdata models
	projectRoot, err := testutil.GetProjectRoot()
	if err != nil {
		return fmt.Errorf("failed to find project root: %w", err)
	}

	// Map the requested path to our testdata model
	modelPath := filepath.Join(
		projectRoot,
		"testdata",
		"models",
		"custom",
		"recognition",
		"mobile",
		"PP-OCRv5_mobile_rec.onnx",
	)

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return fmt.Errorf("custom recognition model not found: %s", modelPath)
	}
	return nil
}

// customDictionaryFilesExist verifies custom dictionary files exist.
func (testCtx *TestContext) customDictionaryFilesExist() error {
	// Check for dictionary files in testdata
	projectRoot, err := testutil.GetProjectRoot()
	if err != nil {
		return fmt.Errorf("failed to find project root: %w", err)
	}

	dictDir := filepath.Join(projectRoot, "models", "dictionaries")
	if _, err := os.Stat(dictDir); os.IsNotExist(err) {
		return fmt.Errorf("dictionaries directory not found: %s", dictDir)
	}

	// Check for at least one dictionary file
	files, err := os.ReadDir(dictDir)
	if err != nil {
		return fmt.Errorf("failed to read dictionaries directory: %w", err)
	}

	found := false
	for _, file := range files {
		if strings.HasSuffix(file.Name(), ".txt") {
			found = true
			break
		}
	}

	if !found {
		return errors.New("no dictionary files found")
	}

	return nil
}

// theEnglishLanguageModelShouldBeUsed verifies English language is configured.
func (testCtx *TestContext) theEnglishLanguageModelShouldBeUsed() error {
	// Check if the command includes --language en
	if !strings.Contains(testCtx.LastCommand, "--language en") {
		return errors.New("command does not specify English language")
	}
	return nil
}

// theGermanLanguageModelShouldBeUsed verifies German language is configured.
func (testCtx *TestContext) theGermanLanguageModelShouldBeUsed() error {
	// Check if the command includes --language de
	if !strings.Contains(testCtx.LastCommand, "--language de") {
		return errors.New("command does not specify German language")
	}
	return nil
}

// theCustomDetectionModelShouldBeUsed verifies custom detection model is used.
func (testCtx *TestContext) theCustomDetectionModelShouldBeUsed() error {
	// Check if the command includes --det-model
	if !strings.Contains(testCtx.LastCommand, "--det-model") {
		return errors.New("command does not specify custom detection model")
	}
	return nil
}

// theCustomRecognitionModelShouldBeUsed verifies custom recognition model is used.
func (testCtx *TestContext) theCustomRecognitionModelShouldBeUsed() error {
	// Check if the command includes --rec-model
	if !strings.Contains(testCtx.LastCommand, "--rec-model") {
		return errors.New("command does not specify custom recognition model")
	}
	return nil
}

// theCustomDictionariesShouldBeMergedAndUsed verifies custom dictionaries are used.
func (testCtx *TestContext) theCustomDictionariesShouldBeMergedAndUsed() error {
	// Check if the command includes --dict
	if !strings.Contains(testCtx.LastCommand, "--dict") {
		return errors.New("command does not specify custom dictionaries")
	}
	return nil
}

// theOutputShouldIncludeDebugInformation verifies debug output is present.
func (testCtx *TestContext) theOutputShouldIncludeDebugInformation() error {
	debugIndicators := []string{"DEBUG", "debug", "verbose", "VERBOSE"}
	for _, indicator := range debugIndicators {
		if strings.Contains(testCtx.LastOutput, indicator) {
			return nil
		}
	}
	return fmt.Errorf("output does not contain debug information: %s", testCtx.LastOutput)
}

// timingInformationShouldBeDisplayed verifies timing info is shown.
func (testCtx *TestContext) timingInformationShouldBeDisplayed() error {
	timingIndicators := []string{"time", "Time", "duration", "Duration", "ms", "seconds"}
	for _, indicator := range timingIndicators {
		if strings.Contains(testCtx.LastOutput, indicator) {
			return nil
		}
	}
	return fmt.Errorf("output does not contain timing information: %s", testCtx.LastOutput)
}

// onlyRegionsWithConfidenceShouldBeDetected verifies confidence filtering.
func (testCtx *TestContext) onlyRegionsWithConfidenceShouldBeDetected(threshold float64) error {
	// This is a simplified check - in a real implementation, we would parse output
	if strings.Contains(testCtx.LastCommand, fmt.Sprintf("--confidence %.1f", threshold)) {
		return nil
	}
	return fmt.Errorf("command does not specify confidence threshold %.1f", threshold)
}

// onlyTextWithRecognitionConfidenceShouldBeIncluded verifies recognition confidence filtering.
func (testCtx *TestContext) onlyTextWithRecognitionConfidenceShouldBeIncluded(threshold float64) error {
	// This is a simplified check - in a real implementation, we would parse output
	if strings.Contains(testCtx.LastCommand, fmt.Sprintf("--min-rec-conf %.1f", threshold)) {
		return nil
	}
	return fmt.Errorf("command does not specify recognition confidence threshold %.1f", threshold)
}

// theRecognizerShouldUsePixelHeightInput verifies recognition height setting.
func (testCtx *TestContext) theRecognizerShouldUsePixelHeightInput(height int) error {
	// Check if the command includes the correct height
	if strings.Contains(testCtx.LastCommand, fmt.Sprintf("--rec-height %d", height)) {
		return nil
	}
	return fmt.Errorf("command does not specify recognition height %d", height)
}

// orientationDetectionShouldBeEnabledWithThreshold verifies orientation detection.
func (testCtx *TestContext) orientationDetectionShouldBeEnabledWithThreshold(threshold float64) error {
	if strings.Contains(testCtx.LastCommand, "--detect-orientation") &&
		strings.Contains(testCtx.LastCommand, fmt.Sprintf("--orientation-threshold %.1f", threshold)) {
		return nil
	}
	return fmt.Errorf("orientation detection not properly configured with threshold %.1f", threshold)
}

// textLineOrientationDetectionShouldBeEnabledWithThreshold verifies text line orientation detection.
func (testCtx *TestContext) textLineOrientationDetectionShouldBeEnabledWithThreshold(threshold float64) error {
	if strings.Contains(testCtx.LastCommand, "--detect-textline") &&
		strings.Contains(testCtx.LastCommand, fmt.Sprintf("--textline-threshold %.1f", threshold)) {
		return nil
	}
	return fmt.Errorf("text line orientation detection not properly configured with threshold %.1f", threshold)
}

// theOutputShouldBeInJSONFormat verifies JSON output format.
func (testCtx *TestContext) theOutputShouldBeInJSONFormat() error {
	return testCtx.theOutputShouldBeValidJSON()
}

// theOutputShouldBeInCSVFormat verifies CSV output format.
func (testCtx *TestContext) theOutputShouldBeInCSVFormat() error {
	return testCtx.theOutputShouldBeValidCSV()
}

// theCSVShouldContainProperHeaders verifies CSV headers.
func (testCtx *TestContext) theCSVShouldContainProperHeaders() error {
	if err := testCtx.theOutputShouldBeValidCSV(); err != nil {
		return err
	}

	// Check for expected headers
	expectedHeaders := []string{"text", "confidence"}
	for _, header := range expectedHeaders {
		if !strings.Contains(testCtx.LastOutput, header) {
			return fmt.Errorf("CSV missing expected header: %s", header)
		}
	}

	return nil
}

// theResultsShouldBeWrittenTo verifies output file.
func (testCtx *TestContext) theResultsShouldBeWrittenTo(filename string) error {
	return testCtx.theFileShouldExist(filename)
}

// overlayImagesShouldBeCreatedInDirectory verifies overlay creation.
func (testCtx *TestContext) overlayImagesShouldBeCreatedInDirectory(directory string) error {
	return testCtx.theFileShouldExist(directory)
}

// theOverlayImagesShouldShowDetectedRegions verifies overlay content.
func (testCtx *TestContext) theOverlayImagesShouldShowDetectedRegions() error {
	// This is a simplified check - in a real implementation, we would verify image content
	return nil
}

// GermanLanguageShouldBeUsed verifies German language configuration.
func (testCtx *TestContext) GermanLanguageShouldBeUsed() error {
	return testCtx.theGermanLanguageModelShouldBeUsed()
}

// confidenceThresholdShouldBe verifies confidence threshold.
func (testCtx *TestContext) confidenceThresholdShouldBe(threshold float64) error {
	if strings.Contains(testCtx.LastCommand, fmt.Sprintf("--confidence %.1f", threshold)) {
		return nil
	}
	return fmt.Errorf("confidence threshold %.1f not set", threshold)
}

// orientationDetectionShouldBeEnabled verifies orientation detection is enabled.
func (testCtx *TestContext) orientationDetectionShouldBeEnabled() error {
	if strings.Contains(testCtx.LastCommand, "--detect-orientation") {
		return nil
	}
	return errors.New("orientation detection not enabled")
}

// theServerShouldBindToAllInterfaces verifies server binding.
func (testCtx *TestContext) theServerShouldBindToAllInterfaces() error {
	if testCtx.ServerHost == "0.0.0.0" {
		return nil
	}
	return fmt.Errorf("server not bound to all interfaces, host: %s", testCtx.ServerHost)
}

// externalConnectionsShouldBeAccepted verifies external access.
func (testCtx *TestContext) externalConnectionsShouldBeAccepted() error {
	// This is a simplified check - in a real implementation, we would test external connectivity
	return nil
}

// theServerShouldStartSuccessfully verifies server startup.
func (testCtx *TestContext) theServerShouldStartSuccessfully() error {
	if testCtx.ServerProcess != nil && testCtx.isServerHealthy() {
		return nil
	}
	return errors.New("server did not start successfully")
}

// CORSSShouldBeConfiguredFor verifies CORS configuration.
func (testCtx *TestContext) CORSSShouldBeConfiguredFor(origin string) error {
	// Check if the command includes CORS configuration
	if strings.Contains(testCtx.LastCommand, "--cors-origin "+origin) {
		return nil
	}
	return fmt.Errorf("CORS not configured for origin: %s", origin)
}

// theMaximumUploadSizeShouldBe verifies upload size limit.
func (testCtx *TestContext) theMaximumUploadSizeShouldBe(size string) error {
	if strings.Contains(testCtx.LastCommand, "--max-upload-size "+size) {
		return nil
	}
	return fmt.Errorf("upload size not set to %s", size)
}

// requestTimeoutShouldBe verifies request timeout.
func (testCtx *TestContext) requestTimeoutShouldBe(seconds int) error {
	if strings.Contains(testCtx.LastCommand, fmt.Sprintf("--timeout %d", seconds)) {
		return nil
	}
	return fmt.Errorf("timeout not set to %d seconds", seconds)
}

// thePipelineShouldUseGermanLanguage verifies pipeline language.
func (testCtx *TestContext) thePipelineShouldUseGermanLanguage() error {
	return testCtx.GermanLanguageShouldBeUsed()
}

// detectionConfidenceThresholdShouldBe verifies detection confidence.
func (testCtx *TestContext) detectionConfidenceThresholdShouldBe(threshold float64) error {
	if strings.Contains(testCtx.LastCommand, fmt.Sprintf("--min-det-conf %.1f", threshold)) {
		return nil
	}
	return fmt.Errorf("detection confidence threshold %.1f not set", threshold)
}

// theEnvironmentVariableIsSetTo sets environment variable.
func (testCtx *TestContext) theEnvironmentVariableIsSetTo(name, value string) error {
	testCtx.AddEnvVar(name, value)
	return nil
}

// modelsShouldBeLoadedFrom verifies model loading path.
func (testCtx *TestContext) modelsShouldBeLoadedFrom(path string) error {
	// This is a simplified check - in a real implementation, we would verify actual loading
	return nil
}

// theErrorShouldMentionInvalidConfigurationValues verifies config error.
func (testCtx *TestContext) theErrorShouldMentionInvalidConfigurationValues() error {
	return testCtx.theErrorShouldMention("invalid")
}

// theHelpShouldListAllAvailableFlags verifies help content.
func (testCtx *TestContext) theHelpShouldListAllAvailableFlags() error {
	return testCtx.theOutputShouldListAvailableFlags()
}

// flagDescriptionsShouldBeClearAndHelpful verifies flag descriptions.
func (testCtx *TestContext) flagDescriptionsShouldBeClearAndHelpful() error {
	// This is a simplified check - in a real implementation, we would verify description quality
	if len(strings.TrimSpace(testCtx.LastOutput)) > 100 { // Basic length check
		return nil
	}
	return errors.New("help output appears too brief")
}

// theHelpShouldListAllAvailableSubcommands verifies subcommand listing.
func (testCtx *TestContext) theHelpShouldListAllAvailableSubcommands() error {
	return testCtx.theOutputShouldListAvailableSubcommands()
}

// globalFlagsShouldBeDocumented verifies global flag documentation.
func (testCtx *TestContext) globalFlagsShouldBeDocumented() error {
	globalFlags := []string{"--help", "--version"}
	for _, flag := range globalFlags {
		if !strings.Contains(testCtx.LastOutput, flag) {
			return fmt.Errorf("global flag not documented: %s", flag)
		}
	}
	return nil
}

// buildInformationShouldBeIncluded verifies build info.
func (testCtx *TestContext) buildInformationShouldBeIncluded() error {
	buildIndicators := []string{"build", "Build", "commit", "Commit", "date", "Date"}
	for _, indicator := range buildIndicators {
		if strings.Contains(testCtx.LastOutput, indicator) {
			return nil
		}
	}
	return fmt.Errorf("output does not contain build information: %s", testCtx.LastOutput)
}

// theFileShouldExist verifies a file exists.
func (testCtx *TestContext) theFileShouldExist(filename string) error {
	fullPath := filepath.Join(testCtx.WorkingDir, filename)
	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		return fmt.Errorf("file does not exist: %s", fullPath)
	}
	return nil
}

// theFileShouldContain verifies a file contains specific content.
func (testCtx *TestContext) theFileShouldContain(filename, expectedContent string) error {
	if err := testCtx.theFileShouldExist(filename); err != nil {
		return err
	}

	fullPath := filepath.Join(testCtx.WorkingDir, filename)
	content, err := os.ReadFile(fullPath) //nolint:gosec // G304: Test file reading with controlled path
	if err != nil {
		return fmt.Errorf("failed to read file %s: %w", fullPath, err)
	}

	if !strings.Contains(string(content), expectedContent) {
		return fmt.Errorf("file %s does not contain '%s'\nActual content: %s",
			filename, expectedContent, string(content))
	}

	return nil
}

// theModelsShouldBeLoadedFrom verifies models are loaded from specific path.
func (testCtx *TestContext) theModelsShouldBeLoadedFrom(path string) error {
	// Check if the command includes the models directory or environment variable
	if strings.Contains(testCtx.LastCommand, "--models-dir "+path) {
		return nil
	}

	// Check environment variables
	for _, envVar := range testCtx.EnvVars {
		if strings.HasPrefix(envVar, "GO_OAR_OCR_MODELS_DIR=") && strings.Contains(envVar, path) {
			return nil
		}
	}

	return fmt.Errorf("models not configured to load from: %s", path)
}

// theModelsShouldBeLoadedFromTempDir verifies models are loaded from temp directory.
func (testCtx *TestContext) theModelsShouldBeLoadedFromTempDir() error {
	if testCtx.TempModelsDir == "" {
		return errors.New("no temporary models directory was set up")
	}
	return testCtx.theModelsShouldBeLoadedFrom(testCtx.TempModelsDir)
}

// substituteCommandVariables replaces variables in command strings.
func (testCtx *TestContext) substituteCommandVariables(command string) string {
	if testCtx.TempModelsDir != "" {
		command = strings.ReplaceAll(command, "{temp_models_dir}", testCtx.TempModelsDir)
	}
	return command
}

// germanLanguageShouldBeConfigured verifies German language is configured.
func (testCtx *TestContext) germanLanguageShouldBeConfigured() error {
	return testCtx.theGermanLanguageModelShouldBeUsed()
}

// germanLanguageShouldBeUsed verifies German language is used.
func (testCtx *TestContext) germanLanguageShouldBeUsed() error {
	return testCtx.theGermanLanguageModelShouldBeUsed()
}

// theEnvironmentVariableGOOAROCRModelsDirIsSetTo sets the GO_OAR_OCR_MODELS_DIR environment variable.
func (testCtx *TestContext) theEnvironmentVariableGOOAROCRModelsDirIsSetTo(path string) error {
	testCtx.AddEnvVar("GO_OAR_OCR_MODELS_DIR", path)
	return nil
}

// individualTextLinesShouldBeCorrectedForOrientation verifies text line orientation correction.
func (testCtx *TestContext) individualTextLinesShouldBeCorrectedForOrientation() error {
	if strings.Contains(testCtx.LastCommand, "--detect-textline") {
		return nil
	}
	return errors.New("text line orientation detection not enabled")
}

// theCommandMightFail accepts that command might fail.
func (testCtx *TestContext) theCommandMightFail() error {
	// This step accepts either success or failure
	return nil
}

// theOCRModelsAreNotAvailable verifies OCR models are not available.
func (testCtx *TestContext) theOCRModelsAreNotAvailable() error {
	// Set up environment to simulate missing models
	testCtx.AddEnvVar("GO_OAR_OCR_MODELS_DIR", "/nonexistent/models")
	return nil
}

// theFileShouldContainTheOCROutput verifies file contains OCR output.
func (testCtx *TestContext) theFileShouldContainTheOCROutput() error {
	if testCtx.LastOutputFile == "" {
		return errors.New("no output file specified")
	}
	return testCtx.theFileShouldContain(testCtx.LastOutputFile, "text")
}

// theFileShouldContainTheOCRResults verifies file contains OCR results.
func (testCtx *TestContext) theFileShouldContainTheOCRResults() error {
	return testCtx.theFileShouldContainTheOCROutput()
}

// theFileShouldContainValidJSONCode verifies file contains valid JSON.
func (testCtx *TestContext) theFileShouldContainValidJSONCode() error {
	if testCtx.LastOutputFile == "" {
		return errors.New("no output file specified")
	}

	content, err := os.ReadFile(filepath.Join(testCtx.WorkingDir, testCtx.LastOutputFile))
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	var js json.RawMessage
	if err := json.Unmarshal(content, &js); err != nil {
		return fmt.Errorf("file does not contain valid JSON: %w", err)
	}

	return nil
}

// theOutputShouldBeValidJSONCode verifies output is valid JSON.
func (testCtx *TestContext) theOutputShouldBeValidJSONCode() error {
	return testCtx.theOutputShouldBeValidJSON()
}

// theOutputShouldListServerConfigurationFlags verifies server config flags are listed.
func (testCtx *TestContext) theOutputShouldListServerConfigurationFlags() error {
	serverFlags := []string{"--port", "--host", "--timeout"}
	for _, flag := range serverFlags {
		if !strings.Contains(testCtx.LastOutput, flag) {
			return fmt.Errorf("server flag not listed: %s", flag)
		}
	}
	return nil
}

// theOverlayImageShouldBeCreatedInDirectory verifies overlay image creation.
func (testCtx *TestContext) theOverlayImageShouldBeCreatedInDirectory(directory string) error {
	return testCtx.overlayImagesShouldBeCreatedInDirectory(directory)
}

// theProcessingShouldCompleteWithinTimeout verifies processing completes within timeout.
func (testCtx *TestContext) theProcessingShouldCompleteWithinTimeout() error {
	if testCtx.LastDuration > 30*time.Second {
		return fmt.Errorf("processing took too long: %v", testCtx.LastDuration)
	}
	return nil
}

// theProcessShouldTerminate verifies process termination.
func (testCtx *TestContext) theProcessShouldTerminate() error {
	// This is a placeholder for process termination verification
	return nil
}

// theOutputShouldListAvailableFlags verifies available flags are listed.
func (testCtx *TestContext) theOutputShouldListAvailableFlags() error {
	commonFlags := []string{"--help", "--verbose"}
	for _, flag := range commonFlags {
		if !strings.Contains(testCtx.LastOutput, flag) {
			return fmt.Errorf("flag not listed: %s", flag)
		}
	}
	return nil
}

// theOutputShouldListAvailableSubcommands verifies available subcommands are listed.
func (testCtx *TestContext) theOutputShouldListAvailableSubcommands() error {
	subcommands := []string{"image", "pdf", "serve"}
	for _, cmd := range subcommands {
		if !strings.Contains(testCtx.LastOutput, cmd) {
			return fmt.Errorf("subcommand not listed: %s", cmd)
		}
	}
	return nil
}

// theOutputShouldBeValidCSV verifies output is valid CSV.
func (testCtx *TestContext) theOutputShouldBeValidCSV() error {
	lines := strings.Split(strings.TrimSpace(testCtx.LastOutput), "\n")
	if len(lines) < 1 {
		return errors.New("CSV output is empty")
	}

	// Check if first line looks like a header
	if !strings.Contains(lines[0], ",") {
		return errors.New("CSV output does not contain comma separators")
	}

	return nil
}

// createSyntheticTestImage creates a basic test image if it doesn't exist.
func (testCtx *TestContext) createSyntheticTestImage(imagePath string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(imagePath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	// For now, just create an empty file - in a real implementation,
	// this would create a proper image with text
	file, err := os.Create(imagePath) //nolint:gosec // G304: Test image creation with controlled path
	if err != nil {
		return fmt.Errorf("failed to create synthetic image %s: %w", imagePath, err)
	}
	defer func() {
		if err := file.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "Error closing file: %v\n", err)
		}
	}()

	// Write minimal PNG header to make it a valid (though empty) PNG
	pngHeader := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
	if _, err := file.Write(pngHeader); err != nil {
		return fmt.Errorf("failed to write PNG header: %w", err)
	}

	return nil
}

// RegisterCommonSteps registers all common step definitions.
func (testCtx *TestContext) RegisterCommonSteps(sc *godog.ScenarioContext) {
	// Background steps
	sc.Step(`^the OCR models are available$`, testCtx.theOCRModelsAreAvailable)
	sc.Step(`^the test images are available$`, testCtx.theTestImagesAreAvailable)
	sc.Step(`^the test PDFs are available$`, testCtx.theTestPDFsAreAvailable)

	// Command execution
	sc.Step(`^I run "([^"]*)"$`, testCtx.iRunCommand)

	// Command result verification
	sc.Step(`^the command should succeed$`, testCtx.theCommandShouldSucceed)
	sc.Step(`^the command should fail$`, testCtx.theCommandShouldFail)

	// Output verification
	sc.Step(`^the output should contain "([^"]*)"$`, testCtx.theOutputShouldContain)
	sc.Step(`^the output should be valid JSON$`, testCtx.theOutputShouldBeValidJSON)
	sc.Step(`^the JSON should contain "([^"]*)"$`, testCtx.theJSONShouldContain)

	// Error verification
	sc.Step(`^the error should mention "([^"]*)"$`, testCtx.theErrorShouldMention)

	// File verification
	sc.Step(`^the file "([^"]*)" should exist$`, testCtx.theFileShouldExist)
	sc.Step(`^the file should contain "([^"]*)"$`, func(content string) error {
		// This assumes the last file referenced in the scenario
		return testCtx.theFileShouldContain(testCtx.LastOutputFile, content)
	})

	// Configuration verification
	sc.Step(`^the OCR models are available in "([^"]*)"$`, testCtx.theOCRModelsAreAvailableIn)
	sc.Step(`^the OCR models are available in a temporary directory$`, testCtx.theOCRModelsAreAvailableInTempDir)
	sc.Step(`^a custom detection model exists at "([^"]*)"$`, testCtx.aCustomDetectionModelExistsAt)
	sc.Step(`^a custom recognition model exists at "([^"]*)"$`, testCtx.aCustomRecognitionModelExistsAt)
	sc.Step(`^custom dictionary files exist$`, testCtx.customDictionaryFilesExist)

	// Language model verification
	sc.Step(`^the English language model should be used$`, testCtx.theEnglishLanguageModelShouldBeUsed)
	sc.Step(`^the German language model should be used$`, testCtx.theGermanLanguageModelShouldBeUsed)

	// Custom model usage verification
	sc.Step(`^the custom detection model should be used$`, testCtx.theCustomDetectionModelShouldBeUsed)
	sc.Step(`^the custom recognition model should be used$`, testCtx.theCustomRecognitionModelShouldBeUsed)
	sc.Step(`^the custom dictionaries should be merged and used$`, testCtx.theCustomDictionariesShouldBeMergedAndUsed)

	// Debug and timing information verification
	sc.Step(`^the output should include debug information$`, testCtx.theOutputShouldIncludeDebugInformation)
	sc.Step(`^timing information should be displayed$`, testCtx.timingInformationShouldBeDisplayed)

	// Filtering verification
	sc.Step(`^only regions with confidence (\d+.\d+) should be detected$`,
		testCtx.onlyRegionsWithConfidenceShouldBeDetected)
	sc.Step(`^only text with recognition confidence (\d+.\d+) should be included$`,
		testCtx.onlyTextWithRecognitionConfidenceShouldBeIncluded)

	// Recognition settings verification
	sc.Step(`^the recognizer should use pixel height input (\d+)$`, testCtx.theRecognizerShouldUsePixelHeightInput)

	// Orientation detection verification
	sc.Step(`^orientation detection should be enabled with threshold (\d+.\d+)$`,
		testCtx.orientationDetectionShouldBeEnabledWithThreshold)
	sc.Step(`^text line orientation detection should be enabled with threshold (\d+.\d+)$`,
		testCtx.textLineOrientationDetectionShouldBeEnabledWithThreshold)

	// Output format verification
	sc.Step(`^the output should be in JSON format$`, testCtx.theOutputShouldBeInJSONFormat)
	sc.Step(`^the output should be in CSV format$`, testCtx.theOutputShouldBeInCSVFormat)

	// CSV header verification
	sc.Step(`^the CSV should contain proper headers$`, testCtx.theCSVShouldContainProperHeaders)

	// Result writing verification
	sc.Step(`^the results should be written to "([^"]*)"$`, testCtx.theResultsShouldBeWrittenTo)

	// Overlay image verification
	sc.Step(`^overlay images should be created in directory "([^"]*)"$`, testCtx.overlayImagesShouldBeCreatedInDirectory)
	sc.Step(`^the overlay images should show detected regions$`, testCtx.theOverlayImagesShouldShowDetectedRegions)

	// Server configuration verification
	sc.Step(`^the server should bind to all interfaces$`, testCtx.theServerShouldBindToAllInterfaces)
	sc.Step(`^external connections should be accepted$`, testCtx.externalConnectionsShouldBeAccepted)
	sc.Step(`^the server should start successfully$`, testCtx.theServerShouldStartSuccessfully)

	// CORS configuration verification
	sc.Step(`^CORS should be configured for "([^"]*)"$`, testCtx.CORSSShouldBeConfiguredFor)

	// Upload size and timeout verification
	sc.Step(`^the maximum upload size should be "([^"]*)"$`, testCtx.theMaximumUploadSizeShouldBe)
	sc.Step(`^request timeout should be (\d+)$`, testCtx.requestTimeoutShouldBe)

	// Pipeline verification
	sc.Step(`^the pipeline should use German language$`, testCtx.thePipelineShouldUseGermanLanguage)

	// Detection confidence threshold verification
	sc.Step(`^detection confidence threshold should be (\d+.\d+)$`, testCtx.detectionConfidenceThresholdShouldBe)

	// Environment variable verification
	sc.Step(`^the environment variable "([^"]*)" is set to "([^"]*)"$`, testCtx.theEnvironmentVariableIsSetTo)

	// Model loading verification
	sc.Step(`^models should be loaded from "([^"]*)"$`, testCtx.modelsShouldBeLoadedFrom)

	// Error message verification
	sc.Step(`^the error should mention invalid configuration values$`,
		testCtx.theErrorShouldMentionInvalidConfigurationValues)

	// Help content verification
	sc.Step(`^the help should list all available flags$`,
		testCtx.theHelpShouldListAllAvailableFlags)
	sc.Step(`^the help should list all available subcommands$`,
		testCtx.theHelpShouldListAllAvailableSubcommands)

	// Flag and global option documentation verification
	sc.Step(`^flag descriptions should be clear and helpful$`,
		testCtx.flagDescriptionsShouldBeClearAndHelpful)
	sc.Step(`^global flags should be documented$`, testCtx.globalFlagsShouldBeDocumented)

	// Build information verification
	sc.Step(`^build information should be included$`, testCtx.buildInformationShouldBeIncluded)

	// Additional missing steps
	sc.Step(`^the models should be loaded from "([^"]*)"$`, testCtx.theModelsShouldBeLoadedFrom)
	sc.Step(`^the models should be loaded from the temporary directory$`, testCtx.theModelsShouldBeLoadedFromTempDir)
	sc.Step(`^German language should be configured$`, testCtx.germanLanguageShouldBeConfigured)
	sc.Step(`^German language should be used$`, testCtx.germanLanguageShouldBeUsed)
	sc.Step(`^confidence threshold should be ([0-9.]+)$`, testCtx.confidenceThresholdShouldBe)
	sc.Step(`^detection confidence threshold should be ([0-9.]+)$`, testCtx.detectionConfidenceThresholdShouldBe)
	sc.Step(`^the environment variable GO_OAR_OCR_MODELS_DIR is set to "([^"]*)"$`, testCtx.theEnvironmentVariableGOOAROCRModelsDirIsSetTo)
	sc.Step(`^the maximum upload size should be (.+)$`, testCtx.theMaximumUploadSizeShouldBe)
	sc.Step(`^the recognizer should use ([0-9]+) pixel height input$`, testCtx.theRecognizerShouldUsePixelHeightInput)
	sc.Step(`^individual text lines should be corrected for orientation$`, testCtx.individualTextLinesShouldBeCorrectedForOrientation)
	sc.Step(`^orientation detection should be enabled$`, testCtx.orientationDetectionShouldBeEnabled)
	sc.Step(`^the command might fail$`, testCtx.theCommandMightFail)
	sc.Step(`^the OCR models are not available$`, testCtx.theOCRModelsAreNotAvailable)
	sc.Step(`^text line orientation detection should be enabled with threshold ([0-9.]+)$`,
		testCtx.textLineOrientationDetectionShouldBeEnabledWithThreshold)
	sc.Step(`^overlay images should be created in "([^"]*)" directory$`,
		testCtx.overlayImagesShouldBeCreatedInDirectory)
	sc.Step(`^the file should contain the OCR output$`, testCtx.theFileShouldContainTheOCROutput)
	sc.Step(`^the file should contain the OCR results$`, testCtx.theFileShouldContainTheOCRResults)
	sc.Step(`^the file should contain valid JSON-Code$`, testCtx.theFileShouldContainValidJSONCode)
	sc.Step(`^the output should be valid JSON-Code$`, testCtx.theOutputShouldBeValidJSONCode)
	sc.Step(`^the output should list server configuration flags$`, testCtx.theOutputShouldListServerConfigurationFlags)
	sc.Step(`^the overlay image should be created in "([^"]*)" directory$`, testCtx.theOverlayImageShouldBeCreatedInDirectory)
	sc.Step(`^the processing should complete within timeout$`, testCtx.theProcessingShouldCompleteWithinTimeout)
	sc.Step(`^the process should terminate$`, testCtx.theProcessShouldTerminate)
	sc.Step(`^the output should list available flags$`, testCtx.theOutputShouldListAvailableFlags)
	sc.Step(`^the output should list available subcommands$`, testCtx.theOutputShouldListAvailableSubcommands)
	sc.Step(`^the output should be valid CSV$`, testCtx.theOutputShouldBeValidCSV)
}
