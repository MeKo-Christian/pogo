#!/bin/bash
# Enable ONNX Runtime GPU for current shell session

# Set GPU library path for runtime
export LD_LIBRARY_PATH="/opt/onnxruntime/gpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Add CUDA libraries if not already in path
if [[ -d "/usr/local/cuda/lib64" ]]; then
	export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

echo "ONNX Runtime GPU enabled for this shell session"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Verify GPU libraries are available
if [[ -f "/opt/onnxruntime/gpu/lib/libonnxruntime.so" ]]; then
	echo "✅ GPU ONNX Runtime library found"
else
	echo "❌ GPU ONNX Runtime library not found"
	echo "   Install GPU libraries with: sudo scripts/install-onnxruntime.sh"
fi

# Check CUDA availability
if command -v nvidia-smi >/dev/null 2>&1; then
	echo "✅ NVIDIA GPU driver available"
	nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
else
	echo "❌ NVIDIA GPU driver not found"
fi

# Test CUDA runtime
if ldd /opt/onnxruntime/gpu/lib/libonnxruntime.so 2>/dev/null | grep -E 'cuda|cublas' >/dev/null; then
	echo "✅ CUDA dependencies linked"
else
	echo "⚠️  CUDA dependencies may not be available"
fi
