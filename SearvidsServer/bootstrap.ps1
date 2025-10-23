param(
    [switch]$Clean,
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"

try {
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
        return ($null -ne (Get-Command $cmd -ErrorAction SilentlyContinue))
    }

    function Install-PackageIfMissing {
        param([string]$cmd, [string]$wingetId)
        if (-not (Test-Command $cmd)) {
            Write-Host "[INFO] Installing missing dependency: $cmd"
            try {
                winget install $wingetId -e --accept-source-agreements --accept-package-agreements
                $global:RestartNeeded = $true
                Write-Host "[INFO] Successfully installed: $(( & $cmd --version | Select-Object -First 1 ))"
            }
            catch {
                Write-Host "[ERROR] Failed to install $cmd automatically."
                exit 1
            }
        }
    }

    # --- Dependencies to check/install ---
    Install-PackageIfMissing -cmd "git" -wingetId "Git.Git"
    Install-PackageIfMissing -cmd "cmake" -wingetId "Kitware.CMake"
    Install-PackageIfMissing -cmd "ninja" -wingetId "Ninja-build.Ninja"

    # --- vcpkg + ASIO ---
    if (-not $env:VCPKG_ROOT) {
        $VcpkgRoot = Join-Path $env:USERPROFILE "vcpkg"
        Write-Host "[INFO] VCPKG_ROOT not set. Installing vcpkg at $VcpkgRoot..."
        git clone https://github.com/microsoft/vcpkg.git $VcpkgRoot
        Set-Location $VcpkgRoot
        .\bootstrap-vcpkg.bat
        $env:VCPKG_ROOT = $VcpkgRoot
        $global:RestartNeeded = $true
    }

    $ASIOPaths = @(
        "$env:VCPKG_ROOT\installed\x64-windows\include\asio",
        "$env:ProgramFiles\asio"
    )
    $ASIOFound = $false
    foreach ($path in $ASIOPaths) {
        if (Test-Path $path) { $ASIOFound = $true; break }
    }
    if (-not $ASIOFound) {
        Write-Host "[INFO] Installing ASIO via vcpkg..."
        & "$env:VCPKG_ROOT\vcpkg.exe" install asio:x64-windows
        $global:RestartNeeded = $true
    }

    # --- Restart script if any dependency was installed ---
    if ($RestartNeeded) {
        Write-Host "[INFO] Some tools were just installed. Restarting bootstrap script...(But Restarting computer recommended)"
        Start-Process -FilePath "powershell" -ArgumentList "-ExecutionPolicy Bypass -File `"$ScriptPath`"" -Wait
        exit 0
    }

    # --- Proceed with build ---
    Write-Host "[INFO] Updating git submodules..."
    git submodule sync
    git submodule update --init --recursive

    if (!(Test-Path "build")) { New-Item -ItemType Directory -Path "build" | Out-Null }
    Set-Location build

    Write-Host "[INFO] Configuring project with CMake..."
    cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DBUILD_WHISPERCPP=ON -DBUILD_FFMPEG=ON -DDOWNLOAD_ONNX=ON

    Write-Host "[INFO] Building project..."
    cmake --build . --config Release --parallel

    Write-Host "[SUCCESS] Build complete. Executable in ./build/bin/"

} catch [System.Management.Automation.PSSecurityException] {
    Write-Host "[ERROR] PowerShell script execution is blocked by policy."
    Write-Host "To allow running this script temporarily, execute the following command in this terminal:"
    Write-Host "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass"
    exit 1
}