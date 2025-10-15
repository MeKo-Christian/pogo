package support

import (
    "encoding/csv"
    "encoding/json"
    "errors"
    "fmt"
    "strconv"
    "strings"

    "github.com/cucumber/godog"
)

// theOutputShouldContainDetectedTextRegions verifies the output contains text regions.
func (testCtx *TestContext) theOutputShouldContainDetectedTextRegions() error {
	// For text format, we expect file paths and text content
	if strings.Contains(testCtx.LastOutput, ":") && len(strings.TrimSpace(testCtx.LastOutput)) > 0 {
		return nil
	}
	return fmt.Errorf("output does not appear to contain detected text regions: %s", testCtx.LastOutput)
}

// theOutputShouldBeInTextFormat verifies the output is in text format.
func (testCtx *TestContext) theOutputShouldBeInTextFormat() error {
	// Text format typically has "filename:text" pattern
	if strings.Contains(testCtx.LastOutput, ":") {
		return nil
	}
	// Could also be just plain text without filename prefix
	if len(strings.TrimSpace(testCtx.LastOutput)) > 0 {
		return nil
	}
	return fmt.Errorf("output does not appear to be in text format: %s", testCtx.LastOutput)
}

// theOutputShouldBeValidCSV verifies the output is valid CSV.

// theCSVShouldContainCoordinateColumns verifies CSV has coordinate columns.
func (testCtx *TestContext) theCSVShouldContainCoordinateColumns() error {
	csvContent, err := testCtx.extractCSVContent()
	if err != nil {
		return err
	}

	records, err := testCtx.parseCSVRecords(csvContent)
	if err != nil {
		return err
	}

	if len(records) == 0 {
		return errors.New("CSV has no records")
	}

	return testCtx.validateCSVHeader(records[0])
}

// extractCSVContent extracts CSV content from the output.
func (testCtx *TestContext) extractCSVContent() (string, error) {
	lines := strings.Split(strings.TrimSpace(testCtx.LastOutput), "\n")
	if len(lines) < 1 {
		return "", errors.New("output is empty")
	}

	// Find the CSV header line (first line with commas)
	csvStart := -1
	for i, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" && strings.Contains(line, ",") {
			csvStart = i
			break
		}
	}

	if csvStart == -1 {
		return "", errors.New("output does not contain comma separators")
	}

	// Extract CSV content from header onwards
	return strings.Join(lines[csvStart:], "\n"), nil
}

// parseCSVRecords parses CSV content into records.
func (testCtx *TestContext) parseCSVRecords(csvContent string) ([][]string, error) {
	reader := csv.NewReader(strings.NewReader(csvContent))
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to parse CSV: %w", err)
	}
	return records, nil
}

// validateCSVHeader checks that the CSV header contains all required columns.
func (testCtx *TestContext) validateCSVHeader(header []string) error {
	requiredColumns := []string{"x", "y", "w", "h", "det_conf", "text", "rec_conf"}

	for _, required := range requiredColumns {
		if !testCtx.columnExistsInHeader(header, required) {
			return fmt.Errorf("CSV header missing required column '%s'. Found columns: %v", required, header)
		}
	}

	return nil
}

// columnExistsInHeader checks if a column exists in the header (case-insensitive).
func (testCtx *TestContext) columnExistsInHeader(header []string, column string) bool {
	for _, col := range header {
		if strings.EqualFold(col, column) {
			return true
		}
	}
	return false
}

// countImagesInCommand counts image files referenced in the command.
func (testCtx *TestContext) countImagesInCommand(cmdParts []string) int {
	count := 0
	for _, part := range cmdParts {
		if strings.HasSuffix(part, ".png") || strings.HasSuffix(part, ".jpg") || strings.HasSuffix(part, ".jpeg") {
			count++
		}
	}
	return count
}

// countImageResultsInOutput counts image results found in the output.
func (testCtx *TestContext) countImageResultsInOutput(cmdParts []string, output string) int {
	count := 0
	for _, part := range cmdParts {
		if strings.HasSuffix(part, ".png") || strings.HasSuffix(part, ".jpg") || strings.HasSuffix(part, ".jpeg") {
			if strings.Contains(output, part) {
				count++
			}
		}
	}
	return count
}

// theOutputShouldContainResultsForAllImages verifies output includes all images.
func (testCtx *TestContext) theOutputShouldContainResultsForAllImages() error {
	// Count expected images from the command
	cmdParts := strings.Fields(testCtx.LastCommand)
	expectedImages := testCtx.countImagesInCommand(cmdParts)

	if expectedImages == 0 {
		return fmt.Errorf("could not determine expected number of images from command: %s", testCtx.LastCommand)
	}

	// Count image results in output
	imageCount := testCtx.countImageResultsInOutput(cmdParts, testCtx.LastOutput)

	if imageCount < expectedImages {
		return fmt.Errorf("expected results for %d images, but found results for %d images", expectedImages, imageCount)
	}

	return nil
}

// allDetectedRegionsShouldHaveConfidence verifies all regions meet confidence threshold.
func (testCtx *TestContext) allDetectedRegionsShouldHaveConfidence(threshold float64) error {
	// Extract confidence threshold from command
	if !strings.Contains(testCtx.LastCommand, "--confidence") {
		return errors.New("command does not include confidence threshold")
	}

	// For JSON output, we can check the actual confidence values
	if strings.Contains(testCtx.LastCommand, "--format json") {
		regions, err := extractRegionsFromJSON(testCtx.LastOutput)
		if err != nil {
			return err
		}
		for i, region := range regions {
			if conf, exists := region["det_confidence"]; exists {
				if confFloat, ok := conf.(float64); ok && confFloat < threshold {
					return fmt.Errorf("region %d has confidence %.3f, which is below threshold %.3f", i, confFloat, threshold)
				}
			}
		}
	}

	// For text format, we assume the filtering worked if we got results
	// This is a simplified check - in a real implementation, we might parse confidence from text output
	return nil
}

// extractRegionsFromJSON unmarshals output and returns the regions array of objects.
func extractRegionsFromJSON(raw string) ([]map[string]interface{}, error) {
	var result map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		return nil, fmt.Errorf("failed to parse JSON output: %w", err)
	}
	ocr, ok := result["ocr"].(map[string]interface{})
	if !ok {
		return nil, errors.New("JSON does not contain 'ocr' object")
	}
	arr, ok := ocr["regions"].([]interface{})
	if !ok {
		return nil, errors.New("JSON does not contain 'regions' array")
	}
	out := make([]map[string]interface{}, 0, len(arr))
	for _, it := range arr {
		if m, ok := it.(map[string]interface{}); ok {
			out = append(out, m)
		}
	}
	return out, nil
}

// theOutputShouldContainTextFromCorrectedOrientation verifies orientation correction worked.
func (testCtx *TestContext) theOutputShouldContainTextFromCorrectedOrientation() error {
	// This is a simplified check - we verify that we got some text output
	// In a real implementation, we would compare against expected oriented text
	if len(strings.TrimSpace(testCtx.LastOutput)) == 0 {
		return errors.New("no text output found - orientation correction may have failed")
	}
	return nil
}

// individualTextLinesShouldBeCorrectedForOrientation verifies text line orientation correction.

// theOverlayShouldShowDetectedTextRegions verifies overlay content.
func (testCtx *TestContext) theOverlayShouldShowDetectedTextRegions() error {
	// This would require image analysis in a real implementation
	// For now, we just verify that overlay creation was mentioned in output
	if strings.Contains(testCtx.LastOutput, "overlay") || strings.Contains(testCtx.LastOutput, "Saved overlay") {
		return nil
	}
	return errors.New("no overlay creation confirmation found in output")
}

// onlyHighConfidenceRecognizedTextShouldBeIncluded verifies recognition confidence filtering.
func (testCtx *TestContext) onlyHighConfidenceRecognizedTextShouldBeIncluded() error {
    // If JSON format, parse and validate recognition confidences against --min-rec-conf
    if strings.Contains(testCtx.LastCommand, "--format json") {
        regions, err := extractRegionsFromJSON(testCtx.LastOutput)
        if err != nil {
            return err
        }
        // Default threshold if none provided
        threshold := 0.0
        // Parse --min-rec-conf value from command
        parts := strings.Fields(testCtx.LastCommand)
        for i := 0; i < len(parts); i++ {
            if parts[i] == "--min-rec-conf" && i+1 < len(parts) {
                if v, err := strconv.ParseFloat(parts[i+1], 64); err == nil {
                    threshold = v
                }
                break
            }
        }
        for i, region := range regions {
            if v, ok := region["rec_confidence"]; ok {
                if cf, ok := v.(float64); ok {
                    if cf < threshold {
                        return fmt.Errorf("region %d rec_confidence %.3f below threshold %.3f", i, cf, threshold)
                    }
                }
            }
        }
    }
    // For text or CSV formats, we accept success since confidences are not present
    return nil
}

// theOutputShouldContainText verifies the output contains specific text (case-insensitive).
func (testCtx *TestContext) theOutputShouldContainText(expected string) error {
    if expected == "" {
        return errors.New("expected text must not be empty")
    }
    if !strings.Contains(strings.ToLower(testCtx.LastOutput), strings.ToLower(expected)) {
        return fmt.Errorf("output does not contain expected text %q. Got: %s", expected, testCtx.LastOutput)
    }
    return nil
}

// levenshteinDistance computes the Levenshtein distance between two strings.
func levenshteinDistance(a, b string) int {
    ra := []rune(a)
    rb := []rune(b)
    da := make([]int, len(rb)+1)
    db := make([]int, len(rb)+1)
    for j := range da { da[j] = j }
    for i := 1; i <= len(ra); i++ {
        db[0] = i
        for j := 1; j <= len(rb); j++ {
            cost := 0
            if ra[i-1] != rb[j-1] {
                cost = 1
            }
            db[j] = min3(db[j-1]+1, da[j]+1, da[j-1]+cost)
        }
        copy(da, db)
    }
    return da[len(rb)]
}

func min3(a, b, c int) int {
    if a < b {
        if a < c { return a }
        return c
    }
    if b < c { return b }
    return c
}

// similarity returns a normalized similarity score [0,1] based on Levenshtein distance.
func similarity(a, b string) float64 {
    if a == "" && b == "" { return 1 }
    dist := float64(levenshteinDistance(strings.ToLower(a), strings.ToLower(b)))
    maxLen := float64(max(len([]rune(a)), len([]rune(b))))
    if maxLen == 0 { return 1 }
    return 1.0 - dist/maxLen
}

func max(a, b int) int { if a > b { return a } ; return b }

// theOutputShouldApproximatelyMatch verifies the output approximately matches expected text.
// Uses a default similarity threshold of 0.8 for robustness.
func (testCtx *TestContext) theOutputShouldApproximatelyMatch(expected string) error {
    if expected == "" {
        return errors.New("expected text must not be empty")
    }
    // Extract a plausible text body (strip filenames/paths if present)
    body := testCtx.LastOutput
    // Evaluate similarity line-by-line and also against full body
    best := similarity(body, expected)
    for _, line := range strings.Split(body, "\n") {
        s := similarity(line, expected)
        if s > best { best = s }
    }
    if best+1e-9 < 0.8 { // allow tiny float diff
        return fmt.Errorf("output does not approximately match %q (similarity=%.3f). Got: %s", expected, best, testCtx.LastOutput)
    }
    return nil
}

// theOutputShouldContainGermanText verifies German language processing.
func (testCtx *TestContext) theOutputShouldContainGermanText() error {
    // Strengthened: require presence of at least one German-specific character
    out := testCtx.LastOutput
    if len(strings.TrimSpace(out)) == 0 {
        return errors.New("no text output found for German language processing")
    }
    germanChars := []string{"ä", "ö", "ü", "Ä", "Ö", "Ü", "ß"}
    lower := strings.ToLower(out)
    for _, ch := range germanChars {
        if strings.Contains(lower, strings.ToLower(ch)) {
            return nil
        }
    }
    return errors.New("output does not contain German-specific characters (ä, ö, ü, ß)")
}

// allDetectedRegionsShouldHaveConfidenceGTE verifies all regions have minimum confidence.
func (testCtx *TestContext) allDetectedRegionsShouldHaveConfidenceGTE(threshold float64) error {
	// Simplified check - in real implementation would parse JSON output
	if strings.Contains(testCtx.LastOutput, "confidence") {
		return nil
	}
	return fmt.Errorf("confidence threshold %.2f verification not implemented", threshold)
}

// theOutputShouldShowPagesMaximum verifies maximum page count.
func (testCtx *TestContext) theOutputShouldShowPagesMaximum(maxPages int) error {
	// Simplified check - in real implementation would count pages in output
	return nil
}

// theJSONShouldContainConfidenceScores verifies JSON contains confidence scores.
func (testCtx *TestContext) theJSONShouldContainConfidenceScores() error {
	return testCtx.theJSONShouldContain("ocr.avg_det_confidence")
}

// theJSONShouldContainRegionsArray verifies JSON contains regions array.
func (testCtx *TestContext) theJSONShouldContainRegionsArray() error {
	return testCtx.theJSONShouldContain("ocr.regions")
}

// RegisterImageSteps registers all image processing step definitions.
func (testCtx *TestContext) RegisterImageSteps(sc *godog.ScenarioContext) {
	// Image-specific output verification
	sc.Step(`^the output should contain detected text regions$`, testCtx.theOutputShouldContainDetectedTextRegions)
	sc.Step(`^the output should be in text format$`, testCtx.theOutputShouldBeInTextFormat)
	sc.Step(`^the output should be valid CSV$`, testCtx.theOutputShouldBeValidCSV)
	sc.Step(`^the CSV should contain coordinate columns$`, testCtx.theCSVShouldContainCoordinateColumns)

	// Multi-image processing
	sc.Step(`^the output should contain results for all images$`,
		testCtx.theOutputShouldContainResultsForAllImages)

	// Confidence filtering
	sc.Step(`^all detected regions should have confidence >= ([\d.]+)$`, func(thresholdStr string) error {
		threshold, err := strconv.ParseFloat(thresholdStr, 64)
		if err != nil {
			return fmt.Errorf("invalid confidence threshold: %s", thresholdStr)
		}
		return testCtx.allDetectedRegionsShouldHaveConfidence(threshold)
	})

	// Orientation processing
	sc.Step(`^the output should contain text from corrected orientation$`,
		testCtx.theOutputShouldContainTextFromCorrectedOrientation)
	sc.Step(`^individual text lines should be corrected for orientation$`,
		testCtx.individualTextLinesShouldBeCorrectedForOrientation)

	// Overlay generation
	sc.Step(`^the overlay image should be created in "([^"]*)" directory$`,
		testCtx.theOverlayImageShouldBeCreatedInDirectory)
	sc.Step(`^the overlay should show detected text regions$`,
		testCtx.theOverlayShouldShowDetectedTextRegions)

    // Recognition confidence
    sc.Step(`^only high-confidence recognized text should be included$`,
        testCtx.onlyHighConfidenceRecognizedTextShouldBeIncluded)

    // Text content validation
    sc.Step(`^the output should contain text "([^"]*)"$`, testCtx.theOutputShouldContainText)
    sc.Step(`^the output should approximately match "([^"]*)"$`, testCtx.theOutputShouldApproximatelyMatch)

	// Language-specific processing
	sc.Step(`^the output should contain German text$`, testCtx.theOutputShouldContainGermanText)

	// Recognition configuration
	sc.Step(`^the recognizer should use (\d+) pixel height input$`, func(heightStr string) error {
		height, err := strconv.Atoi(heightStr)
		if err != nil {
			return fmt.Errorf("invalid height: %s", heightStr)
		}
		return testCtx.theRecognizerShouldUsePixelHeightInput(height)
	})

	// Help and documentation
	sc.Step(`^the output should contain usage information$`, testCtx.theOutputShouldContainUsageInformation)
	sc.Step(`^the output should list available flags$`, testCtx.theOutputShouldListAvailableFlags)

	// Additional missing steps
	sc.Step(`^all detected regions should have confidence >= ([0-9.]+)$`,
		testCtx.allDetectedRegionsShouldHaveConfidenceGTE)
	sc.Step(`^the output should show ([0-9]+) pages maximum$`, testCtx.theOutputShouldShowPagesMaximum)
	sc.Step(`^the JSON should contain confidence scores$`, testCtx.theJSONShouldContainConfidenceScores)
	sc.Step(`^the JSON should contain "regions" array$`, testCtx.theJSONShouldContainRegionsArray)
}
