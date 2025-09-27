#!/bin/bash
set -euo pipefail

# ONNX Runtime cleanup script
# Removes all ONNX Runtime installations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly PROJECT_ROOT
readonly ONNX_VERSION="1.23.0"
readonly INSTALL_PREFIX="/usr/local"
readonly OPT_DIR="/opt/onnxruntime"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

log_info() {
	echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
	echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
	echo -e "${RED}[ERROR]${NC} $*"
}

cleanup_downloads() {
	log_info "Removing downloaded archives..."
	cd "$PROJECT_ROOT"
	rm -f "onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
	rm -f "onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz"
}

cleanup_installations() {
	log_info "Removing installed files..."

	# Remove installation directories
	if [[ -d $OPT_DIR ]]; then
		sudo rm -rf "$OPT_DIR"
		log_info "Removed $OPT_DIR"
	fi

	# Remove headers
	if [[ -d "${INSTALL_PREFIX}/include/onnxruntime" ]]; then
		sudo rm -rf "${INSTALL_PREFIX}/include/onnxruntime"
		log_info "Removed headers"
	fi

	# Remove library symlinks
	sudo find "${INSTALL_PREFIX}/lib" -name "libonnxruntime.so*" -type l -delete 2>/dev/null || true
	log_info "Removed library symlinks"

	# Update library cache
	sudo ldconfig
}

cleanup_scripts() {
	log_info "Removing helper scripts..."
	rm -f "${PROJECT_ROOT}/scripts/enable-gpu.sh"
	rm -f "/tmp/enable-ort-gpu.sh" # Legacy script
}

main() {
	log_info "Cleaning up ONNX Runtime installation..."

	if [[ $EUID -eq 0 ]]; then
		log_error "This script should not be run as root"
		exit 1
	fi

	cleanup_downloads
	cleanup_installations
	cleanup_scripts

	log_info "Cleanup complete!"
}

main "$@"
