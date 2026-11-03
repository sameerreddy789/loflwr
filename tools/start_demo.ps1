param(
    [string]$Python = '',
    [int]$Port = 8510,
    [string]$RunsDirectory = 'runs'
)

$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $candidates = @(
        (Join-Path $env:USERPROFILE '.venvs\loflwr-cpu\Scripts\python.exe'),
        (Join-Path $projectDirectory '.venv-cpu\Scripts\python.exe'),
        (Join-Path $projectDirectory '.venv\Scripts\python.exe')
    )
    $Python = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
}
if (-not $Python -or -not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Pass -Python with a Python environment containing the tested training and dashboard dependencies.'
}
if ($Port -lt 1024 -or $Port -gt 65535) {
    throw 'Choose a local port between 1024 and 65535.'
}

Push-Location -LiteralPath $projectDirectory
try {
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    $env:TOKENIZERS_PARALLELISM = 'false'
    Write-Host "FedLoRA-Edge: http://127.0.0.1:$Port"
    Write-Host 'Keep this terminal open. Press Ctrl+C to stop the dashboard.'
    & $Python -m streamlit run fedlora_edge/dashboard.py --server.port $Port --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false -- --runs-dir $RunsDirectory
    if ($LASTEXITCODE -ne 0) { throw "Dashboard exited with code $LASTEXITCODE." }
}
finally {
    Pop-Location
}
