#!/bin/bash
set -e

# ONNX Runtime setup script for pogo
# This script downloads and sets up ONNX Runtime for use with onnxruntime-go

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ONNX_DIR="$PROJECT_ROOT/onnxruntime"

# Detect platform
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

# Map architecture names
case $ARCH in
x86_64)
	ARCH="x64"
	;;
aarch64 | arm64)
	ARCH="arm64"
	;;
*)
	echo "Unsupported architecture: $ARCH"
	exit 1
	;;
esac

# ONNX Runtime version
ONNX_VERSION="1.23.0"

# Determine download URL based on platform
case $OS in
linux)
	if [ "$ARCH" = "x64" ]; then
		DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
		FILENAME="onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
	elif [ "$ARCH" = "arm64" ]; then
		DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz"
		FILENAME="onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz"
	fi
	;;
darwin)
	if [ "$ARCH" = "x64" ]; then
		DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-osx-x86_64-${ONNX_VERSION}.tgz"
		FILENAME="onnxruntime-osx-x86_64-${ONNX_VERSION}.tgz"
	elif [ "$ARCH" = "arm64" ]; then
		DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-osx-arm64-${ONNX_VERSION}.tgz"
		FILENAME="onnxruntime-osx-arm64-${ONNX_VERSION}.tgz"
	fi
	;;
*)
	echo "Unsupported operating system: $OS"
	echo "Please download ONNX Runtime manually from:"
	echo "https://github.com/microsoft/onnxruntime/releases/tag/v${ONNX_VERSION}"
	exit 1
	;;
esac

echo "Setting up ONNX Runtime v${ONNX_VERSION} for ${OS}-${ARCH}..."

# Create onnxruntime directory
mkdir -p "$ONNX_DIR"
cd "$ONNX_DIR"

# Check if already downloaded
if [ -f "$FILENAME" ]; then
	echo "ONNX Runtime archive already exists, skipping download..."
else
	echo "Downloading ONNX Runtime from $DOWNLOAD_URL..."
	curl -L -o "$FILENAME" "$DOWNLOAD_URL"
fi

# Extract if not already extracted
EXTRACTED_DIR="onnxruntime-${OS}-*-${ONNX_VERSION}"
if [ -d "$EXTRACTED_DIR" ]; then
	echo "ONNX Runtime already extracted, skipping extraction..."
else
	echo "Extracting ONNX Runtime..."
	tar -xzf "$FILENAME"
fi

# Find the extracted directory (handle different naming conventions)
ONNX_EXTRACTED=$(find . -maxdepth 1 -type d -name "onnxruntime-*-${ONNX_VERSION}" | head -n 1)

if [ -z "$ONNX_EXTRACTED" ]; then
	echo "Error: Could not find extracted ONNX Runtime directory"
	exit 1
fi

# Create symlinks for easier access
rm -f lib include
ln -sf "$ONNX_EXTRACTED/lib" lib
ln -sf "$ONNX_EXTRACTED/include" include

echo "ONNX Runtime setup complete!"
echo ""
echo "Library path: $ONNX_DIR/lib"
echo "Include path: $ONNX_DIR/include"
echo ""
echo "To use with Go, set the following environment variables:"
echo "export CGO_CFLAGS=\"-I$ONNX_DIR/include\""
echo "export CGO_LDFLAGS=\"-L$ONNX_DIR/lib -lonnxruntime\""

# On Linux/macOS, add library path
if [ "$OS" = "linux" ]; then
	echo "export LD_LIBRARY_PATH=\"$ONNX_DIR/lib:\$LD_LIBRARY_PATH\""
elif [ "$OS" = "darwin" ]; then
	echo "export DYLD_LIBRARY_PATH=\"$ONNX_DIR/lib:\$DYLD_LIBRARY_PATH\""
fi

echo ""
echo "Or run: source scripts/setup-env.sh"
