param(
    [ValidateSet("smoke", "open-small", "medical-small", "research-2d")]
    [string]$Profile = "smoke",
    [switch]$IncludeGated,
    [switch]$SkipDownload,
    [switch]$SkipRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    python -m venv (Join-Path $RepoRoot ".venv")
}

& $PythonExe -m pip install --upgrade pip
if ($Profile -eq "smoke") {
    & $PythonExe -m pip install -e "${RepoRoot}[dev]"
} else {
    & $PythonExe -m pip install -e "${RepoRoot}[research,dev]"
}

if (-not $SkipDownload) {
    $DownloadArgs = @(
        "-m", "merit_feddg.cli", "download",
        "--profile", $Profile,
        "--root", (Join-Path $RepoRoot "artifacts")
    )
    if ($IncludeGated) {
        $DownloadArgs += "--include-gated"
    }
    & $PythonExe @DownloadArgs
}

& $PythonExe -m pytest
if (-not $SkipRun) {
    & $PythonExe -m merit_feddg.cli simulate --config (Join-Path $RepoRoot "configs\smoke.yaml") --output (Join-Path $RepoRoot "runs\smoke")
}

Write-Host "MERIT-FedDG is ready. Profile: $Profile"
