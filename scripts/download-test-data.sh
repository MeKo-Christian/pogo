#!/bin/bash

# Download test data for pogo testing
# This script downloads sample images for testing OCR functionality

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TESTDATA_DIR="$PROJECT_ROOT/testdata"

# Logging functions
log_info() {
	echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
	echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
	echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
	echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are available
check_dependencies() {
	local deps=("curl" "wget")
	local missing=()

	for dep in "${deps[@]}"; do
		if ! command -v "$dep" &>/dev/null; then
			missing+=("$dep")
		fi
	done

	if [ ${#missing[@]} -eq 2 ]; then
		log_error "Neither curl nor wget is available. Please install one of them."
		exit 1
	fi

	if [ ${#missing[@]} -eq 1 ]; then
		log_warning "${missing[0]} is not available, will use the other tool."
	fi
}

# Download function that tries curl first, then wget
download_file() {
	local url="$1"
	local output="$2"
	local description="$3"

	log_info "Downloading $description..."

	# Create output directory if it doesn't exist
	mkdir -p "$(dirname "$output")"

	if command -v curl &>/dev/null; then
		if curl -fsSL -o "$output" "$url"; then
			log_success "Downloaded $description"
			return 0
		else
			log_error "Failed to download $description with curl"
			return 1
		fi
	elif command -v wget &>/dev/null; then
		if wget -q -O "$output" "$url"; then
			log_success "Downloaded $description"
			return 0
		else
			log_error "Failed to download $description with wget"
			return 1
		fi
	else
		log_error "No download tool available"
		return 1
	fi
}

# Generate synthetic test images using Go test
generate_synthetic_images() {
	log_info "Generating synthetic test images..."

	cd "$PROJECT_ROOT"

	if go test ./internal/testutil -run TestGenerateTestImages -v; then
		log_success "Generated synthetic test images"
	else
		log_error "Failed to generate synthetic test images"
		return 1
	fi
}

# Download sample OCR test images from public sources
download_sample_images() {
	log_info "Downloading sample test images..."

	# Create directories
	mkdir -p "$TESTDATA_DIR/images/samples"

	# Sample text images from publicly available sources
	# These are placeholder URLs - in a real implementation, you would use actual test image URLs
	local samples=(
		"https://via.placeholder.com/640x480/FFFFFF/000000?text=Sample+Text+1"
		"https://via.placeholder.com/800x600/FFFFFF/000000?text=Multi+Line%0AText+Sample"
		"https://via.placeholder.com/512x384/FFFFFF/000000?text=OCR+Test+Image"
	)

	local count=1
	for url in "${samples[@]}"; do
		local filename="sample_${count}.png"
		local output="$TESTDATA_DIR/images/samples/$filename"

		if download_file "$url" "$output" "sample image $count"; then
			((count++))
		else
			log_warning "Skipping sample image $count due to download failure"
		fi
	done

	log_info "Downloaded sample images to $TESTDATA_DIR/images/samples/"
}

# Download test documents (if available)
download_test_documents() {
	log_info "Setting up test documents directory..."

	mkdir -p "$TESTDATA_DIR/documents"

	# For now, just create a placeholder
	cat >"$TESTDATA_DIR/documents/README.md" <<'EOF'
# Test Documents

This directory contains test documents for PDF OCR testing.

## Adding Test Documents

To add test documents:

1. Place PDF files in this directory
2. Ensure they contain text suitable for OCR testing
3. Update test fixtures accordingly

## Sample Document Types

- Simple text documents
- Multi-page documents
- Documents with images and text
- Scanned documents
- Documents with various fonts and layouts
EOF

	log_success "Created test documents directory structure"
}

# Create test data manifest
create_manifest() {
	log_info "Creating test data manifest..."

	cat >"$TESTDATA_DIR/manifest.json" <<EOF
{
  "name": "pogo test data",
  "version": "1.0.0",
  "description": "Test data for pogo OCR pipeline testing",
  "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "directories": {
    "images": {
      "simple": "Simple single-word test images",
      "multiline": "Multi-line text document images",
      "rotated": "Rotated text samples",
      "scanned": "Simulated scanned document images",
      "samples": "Downloaded sample images"
    },
    "synthetic": "Generated synthetic test images",
    "fixtures": "Test fixtures with expected results",
    "documents": "Test PDF documents"
  },
  "generation_method": "Combination of synthetic generation and downloads",
  "tools_used": [
    "Go test framework",
    "golang.org/x/image libraries",
    "github.com/disintegration/imaging"
  ]
}
EOF

	log_success "Created test data manifest"
}

# Print usage information
usage() {
	cat <<EOF
Usage: $0 [OPTIONS]

Download and generate test data for pogo testing.

OPTIONS:
    -h, --help          Show this help message
    -s, --synthetic     Generate only synthetic images
    -d, --download      Download only sample images
    -a, --all           Generate and download all test data (default)
    --no-synthetic      Skip synthetic image generation
    --no-download       Skip sample image download

EXAMPLES:
    $0                  # Generate and download all test data
    $0 --synthetic      # Generate only synthetic images
    $0 --download       # Download only sample images
    $0 --no-download    # Generate synthetic images only

EOF
}

# Main function
main() {
	local generate_synthetic=true
	local download_samples=true

	# Parse command line arguments
	while [[ $# -gt 0 ]]; do
		case $1 in
		-h | --help)
			usage
			exit 0
			;;
		-s | --synthetic)
			generate_synthetic=true
			download_samples=false
			shift
			;;
		-d | --download)
			generate_synthetic=false
			download_samples=true
			shift
			;;
		-a | --all)
			generate_synthetic=true
			download_samples=true
			shift
			;;
		--no-synthetic)
			generate_synthetic=false
			shift
			;;
		--no-download)
			download_samples=false
			shift
			;;
		*)
			log_error "Unknown option: $1"
			usage
			exit 1
			;;
		esac
	done

	log_info "Starting test data setup for pogo"
	log_info "Project root: $PROJECT_ROOT"
	log_info "Test data directory: $TESTDATA_DIR"

	# Check dependencies
	if [ "$download_samples" = true ]; then
		check_dependencies
	fi

	# Ensure testdata directory exists
	mkdir -p "$TESTDATA_DIR"

	# Generate synthetic images
	if [ "$generate_synthetic" = true ]; then
		if ! generate_synthetic_images; then
			log_error "Failed to generate synthetic images"
			exit 1
		fi
	fi

	# Download sample images
	if [ "$download_samples" = true ]; then
		download_sample_images
	fi

	# Always set up document structure
	download_test_documents

	# Create manifest
	create_manifest

	log_success "Test data setup completed successfully!"
	log_info "Test data is available in: $TESTDATA_DIR"

	# Show summary
	echo
	log_info "Summary of available test data:"
	if [ -d "$TESTDATA_DIR/images" ]; then
		echo "  - Images: $(find "$TESTDATA_DIR/images" -name "*.png" | wc -l) files"
	fi
	if [ -d "$TESTDATA_DIR/fixtures" ]; then
		echo "  - Fixtures: $(find "$TESTDATA_DIR/fixtures" -name "*.json" | wc -l) files"
	fi
	echo "  - Manifest: $TESTDATA_DIR/manifest.json"

	echo
	log_info "To run tests with this data:"
	echo "  go test ./..."
	echo "  just test"
	echo "  just test-coverage"
}

# Run main function
main "$@"
