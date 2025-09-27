#!/bin/bash

# Environment setup script for pogo
# Sets up environment variables for ONNX Runtime

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ONNX_DIR="$PROJECT_ROOT/onnxruntime"

# Check if ONNX Runtime is installed
if [ ! -d "$ONNX_DIR/lib" ]; then
	echo "ONNX Runtime not found. Please run: ./scripts/setup-onnxruntime.sh"
	# shellcheck disable=SC2317
	{ return 1 2>/dev/null; } || exit 1
fi

# Set CGO flags for compilation
export CGO_CFLAGS="-I$ONNX_DIR/include"
export CGO_LDFLAGS="-L$ONNX_DIR/lib -lonnxruntime"

# Set library path for runtime
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
case $OS in
linux)
	export LD_LIBRARY_PATH="$ONNX_DIR/lib:$LD_LIBRARY_PATH"
	;;
darwin)
	export DYLD_LIBRARY_PATH="$ONNX_DIR/lib:$DYLD_LIBRARY_PATH"
	;;
esac

echo "Environment configured for ONNX Runtime:"
echo "  CGO_CFLAGS: $CGO_CFLAGS"
echo "  CGO_LDFLAGS: $CGO_LDFLAGS"
if [ "$OS" = "linux" ]; then
	echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
elif [ "$OS" = "darwin" ]; then
	echo "  DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
fi
