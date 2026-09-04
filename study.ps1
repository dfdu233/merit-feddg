param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,
    [ValidateSet("medical-small", "research-2d")]
    [string]$ModelProfile = "medical-small",
    [string]$EvidenceConfig = "configs\medical_small.yaml",
    [string]$ComparisonConfig = "configs\real_compare.example.yaml",
    [string]$RunName = "real-lodo",
    [switch]$OracleRouter,
    [switch]$IncludeGated,
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Artifacts = Join-Path $RepoRoot "artifacts"
$Evidence = Join-Path $RepoRoot "cache\$RunName.jsonl"
$RunDirectory = Join-Path $RepoRoot "runs\$RunName"

& (Join-Path $RepoRoot "bootstrap.ps1") -Profile $ModelProfile -IncludeGated:$IncludeGated -SkipDownload:$SkipDownload -SkipRun

$ExtractArgs = @(
    "-m", "merit_feddg.cli", "extract",
    "--manifest", $Manifest,
    "--config", (Join-Path $RepoRoot $EvidenceConfig),
    "--artifacts", $Artifacts,
    "--output", $Evidence
)
if ($OracleRouter) {
    $ExtractArgs += "--oracle-router"
}
& $PythonExe @ExtractArgs
& $PythonExe -m merit_feddg.cli compare --input $Evidence --config (Join-Path $RepoRoot $ComparisonConfig) --output $RunDirectory
& $PythonExe -m merit_feddg.cli med-defer-compare --input $Evidence --config (Join-Path $RepoRoot $ComparisonConfig) --output (Join-Path $RunDirectory "med-defer\result.json")

Write-Host "Completed real-model study: $RunDirectory"
