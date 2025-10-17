Write-Host "🚀 Initializing submodules..."
git submodule update --init --recursive

$VcpkgRoot = "$env:USERPROFILE\vcpkg"
if (-Not (Test-Path $VcpkgRoot)) {
    Write-Host "⬇️ Installing vcpkg..."
    git clone https://github.com/microsoft/vcpkg.git $VcpkgRoot
    & $VcpkgRoot\bootstrap-vcpkg.bat
}

Write-Host "🚀 Installing FFmpeg via vcpkg..."
& $VcpkgRoot\vcpkg install ffmpeg:x64-windows

$BuildDir = "build"
if (-Not (Test-Path $BuildDir)) { New-Item -ItemType Directory -Path $BuildDir }
Set-Location $BuildDir

Write-Host "🚀 Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_WHISPERCPP=ON `
    -DCMAKE_TOOLCHAIN_FILE="$VcpkgRoot\scripts\buildsystems\vcpkg.cmake" `
    -DDOWNLOAD_ONNX=ON

Write-Host "🚀 Building SearvidsServer..."
cmake --build . --config Release

Write-Host "✅ Server built at .\bin\SearvidsServer.exe"
Write-Host "Run with: .\bin\SearvidsServer.exe"