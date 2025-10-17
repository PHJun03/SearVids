#!/bin/bash
set -e

echo "🚀 Initializing submodules..."
git submodule update --init --recursive

echo "🚀 Checking system dependencies..."
if command -v apt >/dev/null; then
    sudo apt update && sudo apt install -y git cmake build-essential pkg-config wget unzip nasm yasm
elif command -v dnf >/dev/null; then
    sudo dnf install -y git cmake make gcc-c++ pkgconfig wget unzip nasm yasm
elif command -v brew >/dev/null; then
    brew install git cmake nasm wget unzip
else
    echo "⚠️ Unsupported package manager, please install dependencies manually."
fi

echo "🚀 Creating build directory..."
mkdir -p build
cd build

echo "🚀 Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_WHISPERCPP=ON -DBUILD_FFMPEG=ON -DDOWNLOAD_ONNX=ON

echo "🚀 Building SearvidsServer..."
cmake --build . -j$(nproc || sysctl -n hw.ncpu)

echo "✅ Build complete. Run server with:"
echo "./bin/SearvidsServer"