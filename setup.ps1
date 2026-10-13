param([string]$TaskBasePython = 'D:\ProgramData\anaconda3\envs\pytorch\python.exe')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $TaskBasePython)) { throw 'Specify -TaskBasePython pointing to your existing CUDA PyTorch Python.' }
& $TaskBasePython -m venv --system-site-packages (Join-Path $PSScriptRoot '.venv')
if ($LASTEXITCODE -ne 0) { throw 'Failed to create independent environment' }
$TaskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $TaskPython -m pip install -r requirements-local.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $TaskPython -m pip install -e . --no-deps
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed' }
