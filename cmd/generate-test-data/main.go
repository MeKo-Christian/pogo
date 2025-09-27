package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/png"
	"log/slog"
	"os"

	"github.com/MeKo-Tech/pogo/internal/testutil"
)

func main() {
	// Set up structured logging
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	var (
		generateImages   = flag.Bool("images", true, "Generate synthetic test images")
		generateFixtures = flag.Bool("fixtures", true, "Generate test fixtures")
		verbose          = flag.Bool("v", false, "Verbose output")
		help             = flag.Bool("h", false, "Show help")
	)

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [OPTIONS]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Generate test data for pogo testing.\n\n")
		fmt.Fprintf(os.Stderr, "OPTIONS:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nEXAMPLES:\n")
		fmt.Fprintf(os.Stderr, "  %s                 # Generate all test data\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -images         # Generate only images\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -fixtures       # Generate only fixtures\n", os.Args[0])
	}

	flag.Parse()

	if *help {
		flag.Usage()
		return
	}

	slog.Info("Starting test data generation...")

	if *verbose {
		slog.Info("Options", "images", *generateImages, "fixtures", *generateFixtures)
	}

	// Get project root to ensure we're in the right place
	root, err := testutil.GetProjectRoot()
	if err != nil {
		slog.Error("Failed to find project root", "error", err)
		os.Exit(1)
	}

	if *verbose {
		slog.Info("Project root", "path", root)
	}

	// Change to project root
	if err := os.Chdir(root); err != nil {
		slog.Error("Failed to change to project root", "error", err)
		os.Exit(1)
	}

	if *generateImages {
		slog.Info("Generating synthetic test images...")

		// Create a dummy test to trigger image generation
		// This is a bit of a hack, but it reuses our existing test infrastructure
		if err := generateTestImages(); err != nil {
			slog.Error("Failed to generate test images", "error", err)
			os.Exit(1)
		}

		slog.Info("✓ Generated synthetic test images")
	}

	if *generateFixtures {
		slog.Info("Generating test fixtures...")

		if err := generateTestFixtures(); err != nil {
			slog.Error("Failed to generate test fixtures", "error", err)
			os.Exit(1)
		}

		slog.Info("✓ Generated test fixtures")
	}

	slog.Info("Test data generation completed successfully!")
}

// generateTestImages generates synthetic test images.
func generateTestImages() error {
	// Create test images using our image generation utilities
	config := testutil.DefaultTestImageConfig()

	// Generate simple images
	simpleDir := "testdata/images/simple"
	if err := testutil.EnsureDir(simpleDir); err != nil {
		return fmt.Errorf("failed to create simple images directory: %w", err)
	}

	words := []string{"Hello", "World", "OCR", "Test", "123", "Sample"}
	for i, word := range words {
		config.Text = word
		config.Size = testutil.SmallSize

		img, err := testutil.GenerateTextImage(config)
		if err != nil {
			return fmt.Errorf("failed to generate image for word '%s': %w", word, err)
		}

		imagePath := fmt.Sprintf("%s/simple_%d_%s.png", simpleDir, i+1, word)

		// Create a temporary file to save the image
		file, err := os.Create(imagePath) //nolint:gosec // G304: Test data generation uses controlled paths
		if err != nil {
			return fmt.Errorf("failed to create image file: %w", err)
		}

		// Use our image saving logic (but without the testing.T)
		if err := saveImageToFile(img, file); err != nil {
			_ = file.Close() // gosec: ignore error on cleanup
			return fmt.Errorf("failed to save image: %w", err)
		}
		if err := file.Close(); err != nil {
			return fmt.Errorf("failed to close file: %w", err)
		}
	}

	// Generate multiline images
	multilineDir := "testdata/images/multiline"
	if err := testutil.EnsureDir(multilineDir); err != nil {
		return fmt.Errorf("failed to create multiline images directory: %w", err)
	}

	config.Size = testutil.LargeSize
	config.Multiline = true

	img, err := testutil.GenerateTextImage(config)
	if err != nil {
		return fmt.Errorf("failed to generate multiline image: %w", err)
	}

	file, err := os.Create(multilineDir + "/multiline_document.png")
	if err != nil {
		return fmt.Errorf("failed to create multiline image file: %w", err)
	}
	defer file.Close()

	if err := saveImageToFile(img, file); err != nil {
		return fmt.Errorf("failed to save multiline image: %w", err)
	}

	// Generate rotated images
	rotatedDir := "testdata/images/rotated"
	if err := testutil.EnsureDir(rotatedDir); err != nil {
		return fmt.Errorf("failed to create rotated images directory: %w", err)
	}

	rotations := []float64{0, 90, 180, 270, 45, -45}
	for _, rotation := range rotations {
		config.Text = "Rotated Text"
		config.Size = testutil.MediumSize
		config.Rotation = rotation
		config.Multiline = false

		img, err := testutil.GenerateTextImage(config)
		if err != nil {
			return fmt.Errorf("failed to generate rotated image for angle %.1f: %w", rotation, err)
		}

		imagePath := fmt.Sprintf("%s/rotated_%.0f.png", rotatedDir, rotation)
		file, err := os.Create(imagePath) //nolint:gosec // G304: Test data generation uses controlled paths
		if err != nil {
			return fmt.Errorf("failed to create rotated image file: %w", err)
		}

		if err := saveImageToFile(img, file); err != nil {
			_ = file.Close() // gosec: ignore error on cleanup
			return fmt.Errorf("failed to save rotated image: %w", err)
		}
		if err := file.Close(); err != nil {
			return fmt.Errorf("failed to close file: %w", err)
		}
	}

	// Generate scanned document simulation
	scannedDir := "testdata/images/scanned"
	if err := testutil.EnsureDir(scannedDir); err != nil {
		return fmt.Errorf("failed to create scanned images directory: %w", err)
	}

	config.Text = "Scanned Document Sample"
	config.Size = testutil.LargeSize
	config.Rotation = 0
	config.Multiline = false

	img, err = testutil.GenerateTextImage(config)
	if err != nil {
		return fmt.Errorf("failed to generate scanned document image: %w", err)
	}

	file, err = os.Create(scannedDir + "/scanned_document.png")
	if err != nil {
		return fmt.Errorf("failed to create scanned document file: %w", err)
	}
	defer file.Close()

	if err := saveImageToFile(img, file); err != nil {
		return fmt.Errorf("failed to save scanned document: %w", err)
	}

	return nil
}

// generateTestFixtures generates test fixtures.
func generateTestFixtures() error {
	fixturesDir := "testdata/fixtures"
	if err := testutil.EnsureDir(fixturesDir); err != nil {
		return fmt.Errorf("failed to create fixtures directory: %w", err)
	}

	// Create sample fixtures using our testutil structures
	fixtures := []testutil.TestFixture{
		{
			Name:        "simple_hello",
			Description: "Simple single word 'Hello' detection and recognition",
			InputFile:   "images/simple/simple_1_Hello.png",
			Expected: testutil.OCRExpectedResult{
				TextRegions: []testutil.TextRegion{
					{
						Text:       "Hello",
						Confidence: 0.95,
						BoundingBox: testutil.BoundingBox{
							X:      130,
							Y:      115,
							Width:  60,
							Height: 15,
						},
					},
				},
				FullText:   "Hello",
				Confidence: 0.95,
			},
		},
		{
			Name:        "multiline_document",
			Description: "Multiline text document detection and recognition",
			InputFile:   "images/multiline/multiline_document.png",
			Expected: testutil.OCRExpectedResult{
				TextRegions: []testutil.TextRegion{
					{
						Text:       "This is a",
						Confidence: 0.92,
						BoundingBox: testutil.BoundingBox{
							X:      450,
							Y:      350,
							Width:  80,
							Height: 15,
						},
					},
				},
				FullText:   "This is a multiline text sample for OCR testing purposes",
				Confidence: 0.92,
			},
		},
	}

	for _, fixture := range fixtures {
		if err := saveFixture(fixture, fixturesDir); err != nil {
			return fmt.Errorf("failed to save fixture '%s': %w", fixture.Name, err)
		}
	}

	return nil
}

// Helper functions that don't require testing.T

func saveImageToFile(img image.Image, file *os.File) error {
	return png.Encode(file, img)
}

func saveFixture(fixture testutil.TestFixture, dir string) error {
	data, err := json.MarshalIndent(fixture, "", "  ")
	if err != nil {
		return err
	}

	filename := fmt.Sprintf("%s/%s.json", dir, fixture.Name)
	return os.WriteFile(filename, data, 0o600)
}
