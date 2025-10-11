#!/bin/bash
set -e

echo "🚀 Initializing submodules..."
git submodule update --init --recursive

echo "🚀 Installing system build tools..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt update
    sudo apt install -y git cmake build-essential pkg-config wget unzip nasm yasm
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install git cmake nasm wget unzip
fi

echo "🚀 Creating build directory..."
mkdir -p build
cd build

echo "🚀 Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_WHISPERCPP=ON -DBUILD_FFMPEG=ON -DDOWNLOAD_ONNX=ON

echo "🚀 Building SearvidsServer..."
cmake --build . -j$(nproc)

echo "🚀 Server built at ./bin/SearvidsServer"