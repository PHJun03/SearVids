param(
    [switch]$Clean,
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"

Write-Host "[INFO] Starting Searvids Server bootstrap (Windows)"
$ScriptPath = $MyInvocation.MyCommand.Path
$RestartNeeded = $false

# --- Clean build directories if requested ---
if ($Clean -or $Rebuild) {
    Write-Host "[INFO] Performing clean build..."
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    Write-Host "[INFO] Clean complete."
    if ($Clean) {
        Write-Host "[INFO] Clean-only mode complete. Exiting."
        exit 0
    }
}

function Test-Command {
    param([string]$cmd)
    return (Get-Command $cmd -ErrorAction SilentlyContinue) -ne $null
}

function Install-PackageIfMissing {
    param([string]$cmd, [string]$wingetId)

    if (-not (Test-Command $cmd)) {
        Write-Host "[INFO] Installing missing dependency: $cmd"
        try {
            winget install $wingetId -e --accept-source-agreements --accept-package-agreements 2>$null
        } catch {
            Write-Host "[WARN] winget returned an error, continuing..."
        }
        if (-not (Test-Command $cmd)) {
            Write-Host "[ERROR] $cmd still not found after installation attempt."
            exit 1
        } else {
            Write-Host "[INFO] Successfully installed or already present: $cmd ($(( & $cmd --version | Select-Object -First 1 )))"
        }
        $global:RestartNeeded = $true
    } else {
        Write-Host "[INFO] Dependency '$cmd' is installed: $(( & $cmd --version | Select-Object -First 1 ))"
    }
}

# --- Install dependencies if missing ---
Install-PackageIfMissing -cmd "git" -wingetId "Git.Git"
Install-PackageIfMissing -cmd "cmake" -wingetId "Kitware.CMake"
Install-PackageIfMissing -cmd "ninja" -wingetId "Ninja-build.Ninja"

# --- Restart script if any dependency was installed ---
if ($RestartNeeded) {
    Write-Host "[INFO] Some tools were just installed. Restarting bootstrap script..."
    Start-Process -FilePath "powershell" -ArgumentList "-ExecutionPolicy Bypass -File `"$ScriptPath`"" -Wait
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
cmake --build . --config Release --parallel

Write-Host "[SUCCESS] Build complete. Executable in ./build/bin/"
