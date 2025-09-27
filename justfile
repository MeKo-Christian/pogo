# pogo justfile

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

# Run go command with ONNX Runtime environment (fallback if direnv not available)
go-onnx *ARGS:
    #!/usr/bin/env bash
    # Check if direnv variables are already set
    if [[ -n "$CGO_CFLAGS" && "$CGO_CFLAGS" == *"onnxruntime"* ]]; then
        # direnv is active, use it
        go {{ ARGS }}
    else
        # direnv not active, set environment manually
        export CGO_CFLAGS="-I{{justfile_directory()}}/onnxruntime/include"
        export CGO_LDFLAGS="-L{{justfile_directory()}}/onnxruntime/lib -lonnxruntime"
        export LD_LIBRARY_PATH="{{justfile_directory()}}/onnxruntime/lib:$LD_LIBRARY_PATH"
        go {{ ARGS }}
    fi

# Run tests
test:
    just go-onnx test -v ./...

# Run tests with coverage
test-coverage:
    just go-onnx test -v -coverprofile=coverage.out ./...
    go tool cover -html=coverage.out -o coverage.html

# Run unit tests only
test-unit:
    just go-onnx test -v -short ./...

# Run integration tests only
test-integration:
    just go-onnx test -v -run "Integration" ./...

# Run CLI integration tests using Cucumber/Godog
test-integration-cli:
    go test -v ./test/integration/cli

# Run CLI integration tests with specific features
test-integration-cli-feature feature:
    cd test/integration/cli && godog run features/{{ feature }}.feature

# Run CLI integration tests in verbose mode
test-integration-cli-verbose:
    cd test/integration/cli && godog --format=pretty --no-colors=false features/

# Run benchmark tests
test-benchmark:
    just go-onnx test -v -run="^$$" -bench=. ./...

# Run ONNX Runtime smoke tests
test-onnx:
    just go-onnx test -v ./internal/onnx

# Test specific package with optional test pattern
test-package package *pattern:
    #!/usr/bin/env bash
    if [ -n "{{ pattern }}" ]; then
        just go-onnx test -v ./{{ package }} -run "{{ pattern }}"
    else
        just go-onnx test -v ./{{ package }}
    fi

# Test parallel functionality specifically
test-parallel:
    just test-package internal/pipeline ".*Parallel.*"

# Generate test data (using Go program)
test-data-generate:
    go run ./cmd/generate-test-data

# Download and setup test data (using shell script)
test-data-setup:
    ./scripts/download-test-data.sh

# Setup all test data (generate + download)
test-data: test-data-generate test-data-setup

# Clean test data
test-data-clean:
    rm -rf testdata/images/*
    rm -rf testdata/fixtures/*
    rm -rf testdata/synthetic/*

# Run tests with race detection
test-race:
    just go-onnx test -v -race ./...

# Run all test variants
test-all: test test-race test-benchmark

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
    # Check if direnv variables are already set
    if [[ -n "$CGO_CFLAGS" && "$CGO_CFLAGS" == *"onnxruntime"* ]]; then
        # direnv is active, use it
        go build \
            -ldflags "-X github.com/MeKo-Tech/pogo/internal/version.Version=$VERSION -X github.com/MeKo-Tech/pogo/internal/version.GitCommit=$COMMIT -X github.com/MeKo-Tech/pogo/internal/version.BuildDate=$BUILD_DATE" \
            -o bin/pogo ./cmd/ocr
    else
        # direnv not active, set environment manually
        export CGO_CFLAGS="-I{{justfile_directory()}}/onnxruntime/include"
        export CGO_LDFLAGS="-L{{justfile_directory()}}/onnxruntime/lib -lonnxruntime"
        export LD_LIBRARY_PATH="{{justfile_directory()}}/onnxruntime/lib:$LD_LIBRARY_PATH"
        go build \
            -ldflags "-X github.com/MeKo-Tech/pogo/internal/version.Version=$VERSION -X github.com/MeKo-Tech/pogo/internal/version.GitCommit=$COMMIT -X github.com/MeKo-Tech/pogo/internal/version.BuildDate=$BUILD_DATE" \
            -o bin/pogo ./cmd/ocr
    fi

# Build the binary without version info (faster for development)
build-dev:
    mkdir -p bin/
    just go-onnx build -o bin/pogo ./cmd/ocr

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
    just go-onnx run ./cmd/ocr {{ args }}

# Run OCR on a test image
run-test-image image="testdata/sample.jpg":
    @echo "Running OCR on {{ image }}..."
    @just run --image {{ image }} --format json

# Build Docker image
docker-build tag="latest":
    docker build -t pogo:{{ tag }} .

# Push Docker image
docker-push tag="latest":
    docker push pogo:{{ tag }}

# Install ONNX Runtime (both CPU and GPU variants)
install-onnxruntime:
    ./scripts/install-onnxruntime.sh

# Clean up ONNX Runtime installation
cleanup-onnxruntime:
    ./scripts/cleanup-onnxruntime.sh

# Setup local ONNX Runtime (project-specific, no root required)
setup-onnxruntime:
    ./scripts/setup-onnxruntime.sh

# Setup environment for ONNX Runtime
setup-env:
    @echo "Run: source scripts/setup-env.sh"

# Download sample models for testing
download-models:
    @echo "Downloading sample models..."
    mkdir -p models/
    @echo "Note: Model download URLs need to be added when available"
    @echo "Please manually download PP-OCR models to models/ directory"

# Show help
help:
    @just --list