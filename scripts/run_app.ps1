<#
.SYNOPSIS
    Launch LinguaBridge AI from its Python 3.11 project environment.

.DESCRIPTION
    1. Locates the project root (this script's parent).
    2. Verifies the Python version and required project dependencies.
    4. Launches Streamlit: python -m streamlit run app.py

    Never falls back to the global Python interpreter. If the virtual environment
    is missing or incomplete, it shows a setup message and exits. Administrator
    privileges are not required.
#>
param(
    [string]$VenvRoot = "D:\LinguaBridge_Environment\venv311",
    [string]$AppPort  = 8501
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")

if (-not (Test-Path (Join-Path $VenvRoot "Scripts\python.exe"))) {
    Write-Error "Virtual environment not found at $VenvRoot. Run setup first (docs/installation_guide.md)."
    exit 1
}

Write-Host "Project root: $projectRoot"
Write-Host "Virtual env : $VenvRoot"

$py = Join-Path $VenvRoot "Scripts\python.exe"
Write-Host "Interpreter : $py"
Set-Location $projectRoot

$pythonVersion = & $py -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to run the project Python interpreter at $py."
}
if ($pythonVersion.Trim() -ne "3.11") {
    throw "LinguaBridge AI was validated with Python 3.11; found Python $pythonVersion. See docs/installation_guide.md."
}

Write-Host "Python       : $pythonVersion"
Write-Host "Checking dependencies ..."
& $py (Join-Path $projectRoot "scripts\check_environment.py")
if ($LASTEXITCODE -ne 0) {
    throw "Required dependencies are missing. See docs/installation_guide.md."
}

Write-Host "Launching Streamlit on port $AppPort ..." -ForegroundColor Cyan
& $py -m streamlit run app.py --server.port $AppPort --browser.gatherUsageStats false
exit $LASTEXITCODE
