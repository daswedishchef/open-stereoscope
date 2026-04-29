param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt
& $Python -m PyInstaller packaging/open-stereoscope.spec --clean --noconfirm

Write-Host "Build complete: dist\open-stereoscope\open-stereoscope.exe"
