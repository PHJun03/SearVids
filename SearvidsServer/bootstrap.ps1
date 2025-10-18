# PowerShell bootstrap script
$ErrorActionPreference = "Stop"

Write-Host "[INFO] Starting Searvids Server bootstrap (Windows)"
$ScriptPath = $MyInvocation.MyCommand.Path
$RestartNeeded = $false

function Test-Command {
    param([string]$cmd)
    return (Get-Command $cmd -ErrorAction SilentlyContinue) -ne $null
}

function Install-PackageIfMissing {
    param([string]$cmd, [string]$wingetId)
    if (-not (Test-Command $cmd)) {
        Write-Host "[INFO] Installing missing dependency: $cmd"
        try {
            winget install $wingetId -e --accept-source-agreements --accept-package-agreements
            $global:RestartNeeded = $true
        }
        catch {
            Write-Host "[ERROR] Failed to install $cmd automatically."
            exit 1
        }
    }
}

# --- Install dependencies if missing ---
Install-PackageIfMissing -cmd "git" -wingetId "Git.Git"
Install-PackageIfMissing -cmd "cmake" -wingetId "Kitware.CMake"
Install-PackageIfMissing -cmd "ninja" -wingetId "Ninja-build.Ninja"

# --- Restart script if any dependency was installed ---
if ($RestartNeeded) {
    Write-Host "[INFO] Dependencies were installed. Restarting bootstrap script..."
    & powershell -ExecutionPolicy Bypass -File $ScriptPath
    exit 0
}

# --- Proceed with build ---
Write-Host "[INFO] Updating git submodules..."
git submodule update --init --recursive

if (!(Test-Path "build")) { New-Item -ItemType Directory -Path "build" | Out-Null }
Set-Location build

Write-Host "[INFO] Configuring project with CMake..."
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DBUILD_WHISPERCPP=ON -DBUILD_FFMPEG=ON -DDOWNLOAD_ONNX=ON

Write-Host "[INFO] Building project..."
cmake --build . --config Release

Write-Host "[SUCCESS] Build complete. Executable in ./build/bin/"