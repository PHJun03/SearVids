#!/bin/bash
set -e

echo "🚀 Initializing submodules..."
git submodule update --init --recursive

# -----------------------------
# Check / Install Ninja
# -----------------------------
if command -v ninja >/dev/null 2>&1; then
    echo "✅ Ninja is already installed: $(ninja --version)"
else
    echo "⬇️ Ninja not found. Installing..."
    if command -v apt >/dev/null 2>&1; then
        sudo apt update
        sudo apt install -y ninja-build
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y ninja-build
    elif command -v brew >/dev/null 2>&1; then
        brew install ninja
    else
        echo "⚠️ Could not install Ninja automatically. Please install manually."
        exit 1
    fi
    echo "✅ Ninja installed: $(ninja --version)"
fi

# -----------------------------
# Check / Install system dependencies
# -----------------------------
echo "🚀 Checking other system dependencies..."
if command -v apt >/dev/null 2>&1; then
    sudo apt install -y git cmake build-essential pkg-config wget unzip nasm yasm
elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y git cmake make gcc-c++ pkgconfig wget unzip nasm yasm
elif command -v brew >/dev/null 2>&1; then
    brew install git cmake nasm wget unzip
fi

# -----------------------------
# Create build directory
# -----------------------------
echo "🚀 Creating build directory..."
mkdir -p build
cd build

# -----------------------------
# Configure CMake
# -----------------------------
echo "🚀 Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_WHISPERCPP=ON \
    -DBUILD_FFMPEG=ON \
    -DDOWNLOAD_ONNX=ON \
    -G Ninja

# -----------------------------
# Build
# -----------------------------
echo "🚀 Building SearvidsServer..."
cmake --build .

echo "✅ Build complete. Run server with:"
echo "./bin/SearvidsServer"