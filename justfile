# go-oar-ocr justfile

set shell := ["bash", "-c"]

# Default target
default: build

# Format code using treefmt
fmt:
    treefmt --allow-missing-formatter

# Generate Go code
gen:
    go generate ./...

# Run golangci-lint
lint:
    golangci-lint run --config ./.golangci.toml --timeout 5m

# Run golangci-lint with fixes
lint-fix:
    golangci-lint run --config ./.golangci.toml --timeout 5m --fix

# Run all checks
check: check-formatted lint test check-tidy

# Check if code is formatted
check-formatted:
    @echo "Checking if code is formatted..."
    @if ! treefmt --allow-missing-formatter --fail-on-change; then \
        echo "Code is not formatted. Run 'just fmt' to fix."; \
        exit 1; \
    fi

# Check if go.mod is tidy
check-tidy:
    @echo "Checking if go.mod is tidy..."
    @go mod tidy
    @if ! git diff --exit-code go.mod go.sum; then \
        echo "go.mod/go.sum not tidy. Run 'go mod tidy'."; \
        exit 1; \
    fi

# Run tests
test:
    go test -v ./...

# Run tests with coverage
test-coverage:
    go test -v -coverprofile=coverage.out ./...
    go tool cover -html=coverage.out -o coverage.html

# Test that the project can build successfully
test-can-build:
    @echo "Testing that the project can build..."
    @just build

# Build the binary with version info
build:
    #!/usr/bin/env bash
    VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "dev")
    COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    mkdir -p bin/
    go build \
        -ldflags "-X github.com/MeKo-Tech/go-oar-ocr/internal/version.Version=${VERSION} -X github.com/MeKo-Tech/go-oar-ocr/internal/version.GitCommit=${COMMIT} -X github.com/MeKo-Tech/go-oar-ocr/internal/version.BuildDate=${BUILD_DATE}" \
        -o bin/go-oar-ocr ./cmd/ocr

# Build the binary without version info (faster for development)
build-dev:
    mkdir -p bin/
    go build -o bin/go-oar-ocr ./cmd/ocr

# Clean build artifacts
clean:
    rm -rf bin/
    rm -f coverage.out coverage.html

# Setup development dependencies and tools (assumes just is already installed)
setup-deps:
    @echo "Installing development tools..."
    command -v go >/dev/null 2>&1 || { echo "Go is required but not installed. Please install Go first."; exit 1; }
    @echo "Installing Go tools..."
    command -v golangci-lint >/dev/null 2>&1 || { echo "Installing golangci-lint..."; curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin v1.60.3; }
    command -v gofumpt >/dev/null 2>&1 || { echo "Installing gofumpt..."; go install mvdan.cc/gofumpt@latest; }
    command -v gci >/dev/null 2>&1 || { echo "Installing gci..."; go install github.com/daixiang0/gci@latest; }
    @echo "Installing formatters..."
    command -v treefmt >/dev/null 2>&1 || { echo "Installing treefmt..."; cargo install treefmt || echo "treefmt installation failed - cargo not found. Please install Rust/Cargo or treefmt manually."; }
    command -v prettier >/dev/null 2>&1 || { echo "Installing prettier..."; npm install -g prettier || echo "prettier installation failed - npm not found. Please install Node.js/npm or prettier manually."; }
    command -v shfmt >/dev/null 2>&1 || { echo "Installing shfmt..."; go install mvdan.cc/sh/v3/cmd/shfmt@latest; }
    command -v shellcheck >/dev/null 2>&1 || { echo "Installing shellcheck..."; echo "Please install shellcheck manually: https://github.com/koalaman/shellcheck#installing"; }
    command -v taplo >/dev/null 2>&1 || { echo "Installing taplo..."; cargo install taplo-cli --locked || echo "taplo installation failed - cargo not found. Please install Rust/Cargo or taplo manually."; }
    command -v yamlfmt >/dev/null 2>&1 || { echo "Installing yamlfmt..."; go install github.com/google/yamlfmt/cmd/yamlfmt@latest; }
    command -v dockerfmt >/dev/null 2>&1 || { echo "Installing dockerfmt..."; go install github.com/reteps/dockerfmt@latest; }
    @echo "Development tools setup complete!"
    @just deps

# Install dependencies
deps:
    go mod download
    go mod tidy

# Run the application with image input
run *args:
    go run ./cmd/ocr {{ args }}

# Run OCR on a test image
run-test-image image="testdata/sample.jpg":
    @echo "Running OCR on {{ image }}..."
    @just run --image {{ image }} --format json

# Build Docker image
docker-build tag="latest":
    docker build -t go-oar-ocr:{{ tag }} .

# Push Docker image
docker-push tag="latest":
    docker push go-oar-ocr:{{ tag }}

# Download sample models for testing
download-models:
    @echo "Downloading sample models..."
    mkdir -p models/
    @echo "Note: Model download URLs need to be added when available"
    @echo "Please manually download PP-OCR models to models/ directory"

# Show help
help:
    @just --list