param(
    [string]$Python = '',
    [int]$Port = 8511,
    [string]$RunDirectory = 'runs/encoder-flower-balanced-v2',
    [string]$PolicyFile = ''
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $env:USERPROFILE '.venvs\loflwr-cpu\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Choose a prepared Python environment with -Python.' }
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Choose a port between 1024 and 65535.' }
Push-Location -LiteralPath $projectDirectory
try {
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    $env:TOKENIZERS_PARALLELISM = 'false'
    Write-Host "Banking demo: http://127.0.0.1:$Port. Press Ctrl+C to stop."
    $demoArguments = @('-m', 'streamlit', 'run', 'fedlora_edge/encoder_dashboard.py', '--server.port', $Port, '--server.address', '127.0.0.1', '--server.headless', 'true', '--browser.gatherUsageStats', 'false', '--', '--run-dir', $RunDirectory)
    if ($PolicyFile) { $demoArguments += @('--policy-file', $PolicyFile) }
    & $Python @demoArguments
    if ($LASTEXITCODE -ne 0) { throw "Demo exited with code $LASTEXITCODE." }
}
finally { Pop-Location }
