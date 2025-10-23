#!/usr/bin/env bash
set -e

echo "[INFO] Starting Searvids Server bootstrap (Linux/macOS)"

SCRIPT_PATH="$(realpath "$0")"
RESTART_FLAG=0

# --- Parse arguments ---
CLEAN=0
REBUILD=0
for arg in "$@"; do
    case "$arg" in
        --clean)
            CLEAN=1 ;;
        --rebuild)
            REBUILD=1 ;;
    esac
done

# --- Clean build directories if requested ---
if [ $CLEAN -eq 1 ] || [ $REBUILD -eq 1 ]; then
    echo "[INFO] Performing clean build..."
    rm -rf build
    echo "[INFO] Clean complete."
    if [ $CLEAN -eq 1 ]; then
        echo "[INFO] Clean-only mode complete. Exiting."
        exit 0
    fi
fi

check_command() {
    command -v "$1" >/dev/null 2>&1
}

install_package() {
    local pkg="$1"
    echo "[INFO] Installing missing dependency: $pkg"

    if command -v apt >/dev/null 2>&1; then
        sudo apt update && sudo apt install -y "$pkg"
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y "$pkg"
    elif command -v brew >/dev/null 2>&1; then
        brew install "$pkg"
    else
        echo "[ERROR] Unknown package manager. Please install $pkg manually."
        exit 1
    fi
}

# --- Check and install dependencies ---
for pkg in git cmake ninja; do
    if ! check_command "$pkg"; then
        install_package "$pkg"
        RESTART_FLAG=1
    fi
    echo "[INFO] Dependency '$pkg' is installed: $($pkg --version | head -n 1)"
done

# --- Restart script if new installs occurred ---
if [ $RESTART_FLAG -eq 1 ]; then
    echo "[INFO] Some tools were just installed. Refreshing shell PATH..."
    hash -r
    echo "[INFO] Restarting bootstrap..."
    exec "$SCRIPT_PATH" "$@"
fi

# --- Proceed with build ---
echo "[INFO] Updating git submodules..."
git submodule update --init --recursive

mkdir -p build
cd build

echo "[INFO] Configuring project with CMake..."
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_WHISPERCPP=ON -DBUILD_FFMPEG=ON -DDOWNLOAD_ONNX=ON

echo "[INFO] Building project..."
cmake --build . --config Release --parallel

echo "[SUCCESS] Build complete. Executable available in ./build/bin/"