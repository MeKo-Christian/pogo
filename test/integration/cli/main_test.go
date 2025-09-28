package cli_test

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MeKo-Tech/pogo/cmd/ocr/cmd"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/MeKo-Tech/pogo/test/integration/cli/support"
	"github.com/cucumber/godog"
)

// testContext holds the global test context.
var testContext *support.TestContext

// InitializeScenario sets up the test context for each scenario.
func InitializeScenario(sc *godog.ScenarioContext) {
	var err error
	testContext, err = support.NewTestContext()
	if err != nil {
		panic(fmt.Sprintf("Failed to create test context: %v", err))
	}

	// Reset viper configuration to prevent state leakage between scenarios
	// This is necessary because viper retains flag values from previous scenario executions
	viper := cmd.GetConfigLoader().GetViper()
	viper.Set("pipeline.recognizer.dict_path", "")
	viper.Set("pipeline.recognizer.dict_langs", "")
	viper.Set("pipeline.detector.model_path", "")
	viper.Set("pipeline.recognizer.model_path", "")

	// Register step definitions
	testContext.RegisterCommonSteps(sc)
	testContext.RegisterImageSteps(sc)
	testContext.RegisterServerSteps(sc)
	testContext.RegisterPDFSteps(sc)
	testContext.RegisterErrorSteps(sc)

	// Setup scenario cleanup
	sc.After(func(ctx context.Context, sc *godog.Scenario, err error) (context.Context, error) {
		if cleanupErr := testContext.Cleanup(); cleanupErr != nil {
			fmt.Printf("Warning: Failed to cleanup test context: %v\n", cleanupErr)
		}
		return ctx, nil
	})
}

// TestFeatures runs the Godog test suite.
func TestFeatures(t *testing.T) {
	// Discover all feature files under the local features directory.
	entries, err := os.ReadDir("features")
	if err != nil {
		t.Fatalf("failed to read features directory: %v", err)
	}

	format := os.Getenv("GODOG_FORMAT")
	if format == "" {
		format = "pretty"
	}
	tags := os.Getenv("GODOG_TAGS")

	found := false
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".feature") {
			continue
		}
		found = true
		featurePath := filepath.Join("features", e.Name())

		t.Run(e.Name(), func(t *testing.T) {
			suite := godog.TestSuite{
				ScenarioInitializer: InitializeScenario,
				Options: &godog.Options{
					Format:   format,
					Tags:     tags,
					Paths:    []string{featurePath},
					TestingT: t,
				},
			}

			if suite.Run() != 0 {
				t.Fatalf("non-zero status returned for %s", featurePath)
			}
		})
	}

	if !found {
		t.Fatalf("no .feature files found in features/")
	}
}

// TestMain ensures the CLI binary exists at project_root/bin/pogo
// before any feature tests run. If it cannot be built, the suite fails early.
func TestMain(m *testing.M) {
	// Locate project root
	root, err := testutil.GetProjectRootValidated()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to locate project root: %v\n", err)
		os.Exit(1)
	}

	binDir := filepath.Join(root, "bin")
	binPath := filepath.Join(binDir, "pogo")

	if _, err := os.Stat(binPath); os.IsNotExist(err) {
		if mkErr := os.MkdirAll(binDir, 0o755); mkErr != nil {
			fmt.Fprintf(os.Stderr, "failed to create bin dir: %v\n", mkErr)
			os.Exit(1)
		}
		// Build the CLI binary
		cmd := exec.CommandContext(context.Background(), "go", "build", "-o", binPath, "./cmd/ocr")
		cmd.Dir = root
		cmd.Env = os.Environ()
		if out, buildErr := cmd.CombinedOutput(); buildErr != nil {
			fmt.Fprintf(os.Stderr, "failed to build CLI binary: %v\n%s\n", buildErr, string(out))
			os.Exit(1)
		}
	}

	// Prepend project bin dir to PATH so plain "pogo" resolves consistently.
	_ = os.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	_ = os.Setenv("GO_OAR_OCR_BIN", binPath)
	_ = os.Setenv("GO_OAR_OCR_MODELS_DIR", filepath.Join(root, "models"))

	os.Exit(m.Run())
}

// Note: Main function removed since this is now a test package.
// Run tests with: go test -v
