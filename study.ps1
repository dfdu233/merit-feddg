param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,
    [string]$EvidenceConfig = "configs\real_2d.yaml",
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

& (Join-Path $RepoRoot "bootstrap.ps1") -Profile research-2d -IncludeGated:$IncludeGated -SkipDownload:$SkipDownload -SkipRun

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

Write-Host "Completed real-model study: $RunDirectory"
