param(
    [string]$Python = '',
    [int]$Port = 8511,
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $env:USERPROFILE '.venvs\loflwr-cpu\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Choose a prepared Python environment with -Python.' }
Push-Location -LiteralPath $projectDirectory
try {
    $preflightArguments = @('tools/demo_preflight.py', '--port', $Port)
    if ($Output) { $preflightArguments += @('--output', $Output) }
    & $Python @preflightArguments
    if ($LASTEXITCODE -ne 0) { throw 'Demo preflight failed. Read the checks above and the saved report.' }
}
finally { Pop-Location }
