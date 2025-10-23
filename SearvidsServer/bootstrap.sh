#!/usr/bin/env bash
set -e

echo "[INFO] Starting Searvids Server bootstrap (Linux/macOS)"

SCRIPT_PATH="$(realpath "$0")"
PROJECT_ROOT="$(dirname "$SCRIPT_PATH")"
RESTART_FLAG=0

cd "$PROJECT_ROOT"  # 프로젝트 루트 기준으로 작업

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
        sudo apt update && sudo apt install -y "$pkg" || {
            echo "[ERROR] Failed to install $pkg. Please run this script with sudo or install manually."
            exit 1
        }
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y "$pkg" || {
            echo "[ERROR] Failed to install $pkg. Please run this script with sudo or install manually."
            exit 1
        }
    elif command -v brew >/dev/null 2>&1; then
        brew install "$pkg" || {
            echo "[ERROR] Failed to install $pkg. Please install manually via Homebrew."
            exit 1
        }
    else
        echo "[ERROR] Unknown package manager. Please install $pkg manually."
        exit 1
    fi
}

# --- Dependencies to check/install ---
DEPS=(git cmake ninja)
for pkg in "${DEPS[@]}"; do
    if ! check_command "$pkg"; then
        install_package "$pkg"
        RESTART_FLAG=1
    fi
    echo "[INFO] Dependency '$pkg' is installed: $($pkg --version | head -n 1)"
done

# --- ASIO (Crow dependency) ---
ASIO_FOUND=0
for dir in "/usr/include/asio" "/usr/local/include/asio"; do
    if [ -d "$dir" ]; then
        ASIO_FOUND=1
        break
    fi
done

if [ $ASIO_FOUND -eq 0 ]; then
    echo "[INFO] ASIO library not found. Installing..."
    if command -v apt >/dev/null 2>&1; then
        sudo apt install -y libasio-dev || {
            echo "[ERROR] Could not install ASIO. Please install manually: sudo apt install libasio-dev"
            exit 1
        }
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y asio-devel || {
            echo "[ERROR] Could not install ASIO. Please install manually: sudo dnf install asio-devel"
            exit 1
        }
    elif command -v brew >/dev/null 2>&1; then
        brew install asio || {
            echo "[ERROR] Could not install ASIO. Please install manually via brew."
            exit 1
        }
    else
        echo "[ERROR] Could not detect package manager. Please install ASIO manually."
        exit 1
    fi
    RESTART_FLAG=1
fi

# --- Restart script if new installs occurred ---
if [ $RESTART_FLAG -eq 1 ]; then
    echo "[INFO] Some tools were just installed. Refreshing shell PATH..."
    hash -r
    echo "[INFO] Restarting bootstrap...(But Restarting computer recommended)"
    exec "$SCRIPT_PATH" "$@"
fi

# --- Proceed with build ---
echo "[INFO] Updating git submodules..."
git submodule sync
git submodule update --init --recursive

mkdir -p build
cd build

echo "[INFO] Configuring project with CMake..."
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_WHISPERCPP=ON -DBUILD_FFMPEG=ON -DDOWNLOAD_ONNX=ON

echo "[INFO] Building project..."
if cmake --build . --config Release --parallel; then
    echo "[SUCCESS] Build complete. Executable available in ./build/bin/"
else
    echo "[ERROR] Build failed!"
    exit 1
fi