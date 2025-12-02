param(
  [string]$ServerExe = "$PSScriptRoot\..\build\Release\SearvidsServer.exe",
  [int]$Port = 8080,
  [bool]$StartServer = $true,
  [bool]$KillAfter = $true,
  [int]$TimeoutSec = 25
)

$ErrorActionPreference = "SilentlyContinue"

function New-Result($Name, $Passed, $StatusCode, $Message) {
  [PSCustomObject]@{ Test=$Name; Passed=$Passed; Status=$StatusCode; Message=$Message }
}

function Get-StatusCodeFromError($err) {
  try { return $err.Exception.Response.StatusCode.value__ } catch { return -1 }
}

# Wait until /health returns status: ok or timeout
function Wait-ForHealth($BaseUrl, $TimeoutSec) {
  $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSec)
  while ([DateTime]::UtcNow -lt $deadline) {
    try {
      $resp = Invoke-RestMethod -Method GET -Uri "$BaseUrl/health" -TimeoutSec 5
      if ($resp.status -eq "ok") { return $true }
    } catch {}
    Start-Sleep -Milliseconds 500
  }
  return $false
}

# Simple GET test
function Test-GET($Name, $Url) {
  try {
    $r = Invoke-WebRequest -Method GET -Uri $Url -TimeoutSec 10
    return New-Result $Name ($r.StatusCode -eq 200) $r.StatusCode "OK"
  } catch {
    return New-Result $Name $false (Get-StatusCodeFromError $_) ($_.Exception.Message)
  }
}

# Simple POST JSON test
function Test-POST-JSON($Name, $Url, $Body) {
  try {
    $json = $Body | ConvertTo-Json -Depth 10
    $r = Invoke-WebRequest -Method POST -Uri $Url -TimeoutSec 20 -ContentType "application/json" -Body $json
    return New-Result $Name ($r.StatusCode -eq 200) $r.StatusCode "OK"
  } catch {
    return New-Result $Name $false (Get-StatusCodeFromError $_) ($_.Exception.Message)
  }
}

# 1) Optionally start the server
$proc = $null
$base = "http://localhost:$Port"

if ($StartServer) {
  if (-not (Test-Path $ServerExe)) {
    Write-Host "Server executable not found: $ServerExe" -ForegroundColor Red
    exit 1
  }
  Write-Host "Starting server: $ServerExe" -ForegroundColor Cyan
  $proc = Start-Process -FilePath $ServerExe -WindowStyle Minimized -PassThru
}

# 2) Wait for /health
Write-Host "Waiting for health at $base/health ..." -ForegroundColor Cyan
if (-not (Wait-ForHealth $base $TimeoutSec)) {
  Write-Host "Health check failed (timeout)" -ForegroundColor Red
  if ($proc -and $KillAfter) { Stop-Process -Id $proc.Id -Force }
  exit 1
}

# 3) Run tests
$results = @()

# 3-1) Health
$results += Test-GET "GET /health" "$base/health"

# 3-2) Index info (treat 404 as SKIP)
$idx = Test-GET "GET /index/info" "$base/index/info"
if (-not $idx.Passed -and $idx.Status -eq 404) {
  $idx.Message = "SKIP (not implemented)"
  $idx.Passed = $true
}
$results += $idx

# 3-3) Search: try query_text first, then fallback to embedding
$search1 = Test-POST-JSON "POST /search (query_text)" "$base/search" @{ query_text = "test"; topk = 3 }
if (-not $search1.Passed) {
  $results += (Test-POST-JSON "POST /search (embedding)" "$base/search" @{ embedding = @(0.1,0.1,0.1,0.1,0.1); topk = 3 })
} else {
  $results += $search1
}

# 4) Summary
Write-Host ""
Write-Host "===== SearvidsServer Smoke Test =====" -ForegroundColor Yellow
$passCount = 0
foreach ($r in $results) {
  $status = if ($r.Passed) { "PASS" } else { "FAIL" }
  if ($r.Passed) { $passCount++ }
  $color = if ($r.Passed) { "Green" } else { "Red" }
  Write-Host ("{0,-30} {1,-6} ({2}) {3}" -f $r.Test, $status, $r.Status, $r.Message) -ForegroundColor $color
}
Write-Host ("=====================================")

$allPass = ($results | Where-Object { -not $_.Passed }).Count -eq 0

# 5) Optionally stop the server
if ($proc -and $KillAfter) {
  Write-Host "Stopping server (PID=$($proc.Id))" -ForegroundColor Cyan
  Stop-Process -Id $proc.Id -Force
}

if ($allPass) { exit 0 } else { exit 1 }