# CLI Integration Tests

This directory contains behavior-driven integration tests for the pogo CLI using Cucumber/Godog.

## Overview

The tests are written in Gherkin format and use [Godog](https://github.com/cucumber/godog) to execute them. They test the CLI commands end-to-end, including:

- Image processing (`pogo image`)
- PDF processing (`pogo pdf`)
- Server mode (`pogo serve`)
- Error handling and edge cases
- Configuration options

## Structure

```
test/integration/cli/
├── features/           # Gherkin feature files
│   ├── image_processing.feature
│   ├── pdf_processing.feature
│   ├── server_mode.feature
│   ├── error_handling.feature
│   └── configuration.feature
├── steps/              # Step definitions
│   ├── common_steps.go
│   ├── image_steps.go
│   ├── server_steps.go
│   └── ...
├── support/            # Test utilities
│   ├── context.go
│   ├── server.go
│   └── ...
├── main_test.go        # Test runner
└── README.md           # This file
```

## Running Tests

### Prerequisites

1. **OCR Models**: Ensure OCR models are available in the `models/` directory or set `GO_OAR_OCR_MODELS_DIR` environment variable.

2. **Test Data**: Ensure test images and PDFs are available in `testdata/`:

   ```bash
   just test-data-setup
   ```

3. **Binary**: Build the pogo binary:
   ```bash
   just build
   ```

### Run All CLI Integration Tests

```bash
# Using justfile
just test-integration-cli

# Or directly
cd test/integration/cli && go test -v
```

### Run Specific Feature

```bash
# Test only image processing
just test-integration-cli-feature image_processing

# Test only server mode
just test-integration-cli-feature server_mode
```

### Run with Verbose Output

```bash
just test-integration-cli-verbose
```

### Run Individual Scenarios

```bash
cd test/integration/cli
godog run features/image_processing.feature:10  # Run scenario at line 10
```

## Test Features

### Image Processing (`image_processing.feature`)

- Single and multiple image processing
- Various output formats (text, JSON, CSV)
- Confidence filtering
- Orientation detection
- Overlay generation
- Language selection

### PDF Processing (`pdf_processing.feature`)

- PDF file processing
- Page range selection
- Multiple output formats
- Error handling for invalid PDFs

### Server Mode (`server_mode.feature`)

- Server startup and shutdown
- API endpoint testing
- Image upload and processing
- Health checks
- Graceful shutdown

### Error Handling (`error_handling.feature`)

- Invalid input files
- Missing models
- Configuration errors
- Network issues

### Configuration (`configuration.feature`)

- Command-line flags
- Environment variables
- Model path configuration
- Pipeline options

## Environment Variables

- `GO_OAR_OCR_MODELS_DIR`: Override default models directory
- `GODOG_FORMAT`: Set output format (pretty, progress, json)
- `GODOG_TAGS`: Run only scenarios with specific tags

## Debugging Tests

### View Test Output

```bash
cd test/integration/cli
godog --format=pretty --no-colors=false features/
```

### Debug Failed Tests

1. Check test output for detailed error messages
2. Verify prerequisites (models, test data, binary)
3. Run individual scenarios to isolate issues
4. Check server logs for server mode tests

### Common Issues

1. **Models not found**: Ensure OCR models are properly installed
2. **Test data missing**: Run `just test-data-setup`
3. **Binary not found**: Run `just build`
4. **Port conflicts**: Ensure ports 8080-8090 are available for server tests
5. **Timeout errors**: Increase timeout for slow systems

## Writing New Tests

### Adding New Scenarios

1. Add scenarios to existing feature files or create new ones
2. Follow Gherkin syntax: Given/When/Then
3. Use existing step definitions when possible
4. Add new step definitions in appropriate `steps/*.go` files

### Example Scenario

```gherkin
Scenario: Process image with custom settings
  Given the OCR models are available
  When I run "pogo image test.png --confidence 0.8 --format json"
  Then the command should succeed
  And the output should be valid JSON
  And all detected regions should have confidence >= 0.8
```

### Step Definition Example

```go
func (tc *TestContext) theOutputShouldContainText(expectedText string) error {
    if !strings.Contains(tc.LastOutput, expectedText) {
        return fmt.Errorf("output does not contain '%s'", expectedText)
    }
    return nil
}
```

## Integration with CI/CD

These tests are designed to run in CI/CD pipelines:

1. **GitHub Actions**: Tests run automatically on pull requests
2. **Local Development**: Run before committing changes
3. **Release Testing**: Validate releases before deployment

The tests require:

- Go runtime
- OCR models (can be downloaded in CI)
- Test data (generated or downloaded)
- Network access for server tests
