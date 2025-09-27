#!/bin/bash
set -euo pipefail

# ONNX Runtime installer for Linux x64
# Installs both CPU and GPU variants with CPU as default

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

check_requirements() {
	if [[ $EUID -eq 0 ]]; then
		log_error "This script should not be run as root"
		exit 1
	fi

	command -v curl >/dev/null 2>&1 || {
		log_error "curl is required but not installed"
		exit 1
	}

	command -v sudo >/dev/null 2>&1 || {
		log_error "sudo is required but not installed"
		exit 1
	}
}

download_archives() {
	log_info "Downloading ONNX Runtime v${ONNX_VERSION}..."

	local cpu_archive="onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
	local gpu_archive="onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz"
	local base_url="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}"

	cd "$PROJECT_ROOT"

	if [[ ! -f $cpu_archive ]]; then
		log_info "Downloading CPU version..."
		curl -fL -o "$cpu_archive" "${base_url}/${cpu_archive}"
	else
		log_info "CPU archive already exists"
	fi

	if [[ ! -f $gpu_archive ]]; then
		log_info "Downloading GPU version..."
		curl -fL -o "$gpu_archive" "${base_url}/${gpu_archive}"
	else
		log_info "GPU archive already exists"
	fi

	log_info "Download complete"
	ls -lh "$cpu_archive" "$gpu_archive"
}

create_directories() {
	log_info "Creating installation directories..."
	sudo mkdir -p "${OPT_DIR}/cpu" "${OPT_DIR}/gpu"
	sudo mkdir -p "${INSTALL_PREFIX}/include/onnxruntime"
}

extract_archives() {
	log_info "Extracting archives..."

	cd "$PROJECT_ROOT"

	local cpu_archive="onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
	local gpu_archive="onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz"

	sudo tar -xzf "$cpu_archive" -C "${OPT_DIR}/cpu" --strip-components=1
	sudo tar -xzf "$gpu_archive" -C "${OPT_DIR}/gpu" --strip-components=1
}

install_headers() {
	log_info "Installing headers..."
	sudo cp -r "${OPT_DIR}/cpu/include/"* "${INSTALL_PREFIX}/include/onnxruntime/"
}

setup_cpu_default() {
	log_info "Setting up CPU version as default..."
	sudo ln -sf "${OPT_DIR}/cpu/lib/libonnxruntime.so"* "${INSTALL_PREFIX}/lib/"
	sudo ldconfig
}

create_gpu_helper() {
	log_info "Creating GPU enabler script..."

	cat >"${PROJECT_ROOT}/scripts/enable-gpu.sh" <<'EOF'
#!/bin/bash
# Enable ONNX Runtime GPU for current shell session

export LD_LIBRARY_PATH="/opt/onnxruntime/gpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Uncomment if CUDA is not in system PATH
# export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

echo "ONNX Runtime GPU enabled for this shell session"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
EOF

	chmod +x "${PROJECT_ROOT}/scripts/enable-gpu.sh"
}

verify_installation() {
	log_info "Verifying installation..."

	if [[ -f "${INSTALL_PREFIX}/lib/libonnxruntime.so" ]]; then
		log_info "✅ CPU version installed successfully"
	else
		log_error "❌ CPU installation failed"
		exit 1
	fi

	if [[ -f "${OPT_DIR}/gpu/lib/libonnxruntime.so" ]]; then
		log_info "✅ GPU version available"

		# Check CUDA dependencies
		if ldd "${OPT_DIR}/gpu/lib/libonnxruntime.so" | grep -E 'cuda|cublas|cudnn' >/dev/null; then
			log_info "✅ GPU version has CUDA dependencies"
		else
			log_warn "⚠️  CUDA libraries not found in system path"
			log_warn "   Use scripts/enable-gpu.sh to enable GPU runtime"
		fi
	else
		log_error "❌ GPU installation failed"
	fi
}

main() {
	log_info "Installing ONNX Runtime v${ONNX_VERSION} for Linux x64"

	check_requirements
	download_archives
	create_directories
	extract_archives
	install_headers
	setup_cpu_default
	create_gpu_helper
	verify_installation

	log_info "Installation complete!"
	log_info "CPU version is active by default"
	log_info "To use GPU: source scripts/enable-gpu.sh"
}

main "$@"
