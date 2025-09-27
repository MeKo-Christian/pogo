package testutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCreateSampleFixtures(t *testing.T) {
	// First generate test images if they don't exist
	GenerateTestImages(t)

	// Create sample fixtures
	CreateSampleFixtures(t)

	// Verify fixtures were created
	fixturesDir := GetFixturesDir(t)
	assert.True(t, DirExists(fixturesDir))

	// Check that fixture files exist
	assert.True(t, FileExists(fixturesDir+"/simple_hello.json"))
	assert.True(t, FileExists(fixturesDir+"/multiline_document.json"))
	assert.True(t, FileExists(fixturesDir+"/rotated_90.json"))
}

func TestLoadFixture(t *testing.T) {
	// First create fixtures
	GenerateTestImages(t)
	CreateSampleFixtures(t)

	// Load a fixture
	fixture := LoadFixture(t, "simple_hello")
	assert.Equal(t, "simple_hello", fixture.Name)
	assert.Equal(t, "Simple single word 'Hello' detection and recognition", fixture.Description)
	assert.Equal(t, "images/simple/simple_1_Hello.png", fixture.InputFile)
	assert.NotNil(t, fixture.Expected)
}

func TestSaveAndLoadFixture(t *testing.T) {
	// Create a test fixture
	fixture := TestFixture{
		Name:        "test_fixture",
		Description: "Test fixture for unit testing",
		InputFile:   "test/input.png",
		Expected: OCRExpectedResult{
			TextRegions: []TextRegion{
				{
					Text:       "Test",
					Confidence: 0.99,
					BoundingBox: BoundingBox{
						X:      10,
						Y:      20,
						Width:  50,
						Height: 15,
					},
				},
			},
			FullText:   "Test",
			Confidence: 0.99,
		},
	}

	// Save fixture
	SaveFixture(t, fixture)

	// Load it back
	loadedFixture := LoadFixture(t, "test_fixture")
	assert.Equal(t, fixture.Name, loadedFixture.Name)
	assert.Equal(t, fixture.Description, loadedFixture.Description)
	assert.Equal(t, fixture.InputFile, loadedFixture.InputFile)
}

func TestValidateFixture(t *testing.T) {
	// Generate test images first
	GenerateTestImages(t)
	CreateSampleFixtures(t)

	// Load a fixture
	fixture := LoadFixture(t, "simple_hello")

	// This should not panic since the input file should exist
	require.NotPanics(t, func() {
		ValidateFixture(t, fixture)
	})
}

func TestGetFixtureInputPath(t *testing.T) {
	fixture := TestFixture{
		InputFile: "images/simple/test.png",
	}

	path := GetFixtureInputPath(t, fixture)
	assert.Contains(t, path, "testdata/images/simple/test.png")
}
