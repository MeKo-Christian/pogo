package testutil

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestFixture represents a test fixture with input and expected output.
type TestFixture struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputFile   string                 `json:"input_file"`
	Expected    interface{}            `json:"expected"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// OCRExpectedResult represents expected OCR output for testing.
type OCRExpectedResult struct {
	TextRegions []TextRegion `json:"text_regions"`
	FullText    string       `json:"full_text"`
	Confidence  float64      `json:"confidence"`
}

// TextRegion represents a detected text region.
type TextRegion struct {
	Text        string      `json:"text"`
	Confidence  float64     `json:"confidence"`
	BoundingBox BoundingBox `json:"bounding_box"`
	Polygon     []Point     `json:"polygon,omitempty"`
}

// BoundingBox represents a rectangular bounding box.
type BoundingBox struct {
	X      int `json:"x"`
	Y      int `json:"y"`
	Width  int `json:"width"`
	Height int `json:"height"`
}

// Point represents a 2D coordinate.
type Point struct {
	X int `json:"x"`
	Y int `json:"y"`
}

// LoadFixture loads a test fixture from JSON file.
func LoadFixture(t *testing.T, name string) TestFixture {
	t.Helper()

	fixturesDir := GetFixturesDir(t)
	fixturePath := filepath.Join(fixturesDir, name+".json")

	data, err := os.ReadFile(fixturePath) //nolint:gosec // G304: Reading test fixture files with controlled paths
	require.NoError(t, err, "Failed to read fixture file: %s", fixturePath)

	var fixture TestFixture
	err = json.Unmarshal(data, &fixture)
	require.NoError(t, err, "Failed to unmarshal fixture JSON")

	return fixture
}

// SaveFixture saves a test fixture to JSON file.
func SaveFixture(t *testing.T, fixture TestFixture) {
	t.Helper()

	fixturesDir := GetFixturesDir(t)
	require.NoError(t, EnsureDir(fixturesDir))

	fixturePath := filepath.Join(fixturesDir, fixture.Name+".json")

	data, err := json.MarshalIndent(fixture, "", "  ")
	require.NoError(t, err, "Failed to marshal fixture to JSON")

	err = os.WriteFile(fixturePath, data, 0o600)
	require.NoError(t, err, "Failed to write fixture file: %s", fixturePath)
}

// createSimpleFixture creates a simple word fixture.
func createSimpleFixture(t *testing.T) TestFixture {
	t.Helper()

	return TestFixture{
		Name:        "simple_hello",
		Description: "Simple single word 'Hello' detection and recognition",
		InputFile:   "images/simple/simple_1_Hello.png",
		Expected: OCRExpectedResult{
			TextRegions: []TextRegion{
				{
					Text:       "Hello",
					Confidence: 0.95,
					BoundingBox: BoundingBox{
						X:      280,
						Y:      230,
						Width:  80,
						Height: 20,
					},
				},
			},
			FullText:   "Hello",
			Confidence: 0.95,
		},
		Metadata: map[string]interface{}{
			"image_size": map[string]int{
				"width":  640,
				"height": 480,
			},
			"font": "basic",
		},
	}
}

// createMultilineFixture creates a multiline document fixture.
func createMultilineFixture(t *testing.T) TestFixture {
	t.Helper()

	return TestFixture{
		Name:        "multiline_document",
		Description: "Multiline text document detection and recognition",
		InputFile:   "images/multiline/multiline_document.png",
		Expected: OCRExpectedResult{
			TextRegions: []TextRegion{
				{
					Text:       "This is a",
					Confidence: 0.92,
					BoundingBox: BoundingBox{
						X:      400,
						Y:      300,
						Width:  120,
						Height: 15,
					},
				},
				{
					Text:       "multiline text sample",
					Confidence: 0.90,
					BoundingBox: BoundingBox{
						X:      350,
						Y:      330,
						Width:  220,
						Height: 15,
					},
				},
				{
					Text:       "for OCR testing",
					Confidence: 0.93,
					BoundingBox: BoundingBox{
						X:      380,
						Y:      360,
						Width:  160,
						Height: 15,
					},
				},
				{
					Text:       "purposes",
					Confidence: 0.94,
					BoundingBox: BoundingBox{
						X:      450,
						Y:      390,
						Width:  90,
						Height: 15,
					},
				},
			},
			FullText:   "This is a multiline text sample for OCR testing purposes",
			Confidence: 0.92,
		},
		Metadata: map[string]interface{}{
			"image_size": map[string]int{
				"width":  1024,
				"height": 768,
			},
			"text_lines": 4,
		},
	}
}

// createRotatedFixture creates a rotated text fixture.
func createRotatedFixture(t *testing.T) TestFixture {
	t.Helper()

	return TestFixture{
		Name:        "rotated_90",
		Description: "90-degree rotated text detection and recognition",
		InputFile:   "images/rotated/rotated_90.png",
		Expected: OCRExpectedResult{
			TextRegions: []TextRegion{
				{
					Text:       "Rotated Text",
					Confidence: 0.88,
					BoundingBox: BoundingBox{
						X:      300,
						Y:      200,
						Width:  25,
						Height: 140,
					},
				},
			},
			FullText:   "Rotated Text",
			Confidence: 0.88,
		},
		Metadata: map[string]interface{}{
			"rotation": 90,
			"image_size": map[string]int{
				"width":  640,
				"height": 480,
			},
		},
	}
}

// CreateSampleFixtures creates sample test fixtures.
func CreateSampleFixtures(t *testing.T) {
	t.Helper()

	// Simple word fixture
	simpleFixture := createSimpleFixture(t)
	SaveFixture(t, simpleFixture)

	// Multiline document fixture
	multilineFixture := createMultilineFixture(t)
	SaveFixture(t, multilineFixture)

	// Rotated text fixture
	rotatedFixture := createRotatedFixture(t)
	SaveFixture(t, rotatedFixture)
}

// GetFixtureInputPath returns the full path to a fixture's input file.
func GetFixtureInputPath(t *testing.T, fixture TestFixture) string {
	t.Helper()

	testDataDir := GetTestDataDir(t)
	return filepath.Join(testDataDir, fixture.InputFile)
}

// ValidateFixture validates that a fixture's input file exists.
func ValidateFixture(t *testing.T, fixture TestFixture) {
	t.Helper()

	inputPath := GetFixtureInputPath(t, fixture)
	require.True(t, FileExists(inputPath), "Fixture input file does not exist: %s", inputPath)
}
