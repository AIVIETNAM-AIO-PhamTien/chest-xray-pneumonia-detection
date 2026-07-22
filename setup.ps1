# Tao virtual environment (.venv) va cai dat dependencies cho dev/test local.
# Khong dung cho Colab/Kaggle - o do da la moi truong cach ly san.
#
# Chay: powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"

$venvDir = ".venv"

if (-not (Test-Path $venvDir)) {
    Write-Host "Tao virtual environment tai $venvDir ..."
    python -m venv $venvDir
} else {
    Write-Host "Virtual environment da ton tai tai $venvDir, bo qua buoc tao."
}

$python = Join-Path $venvDir "Scripts\python.exe"

Write-Host "Cai dat dependencies tu requirements.txt ..."
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Xong. De kich hoat moi truong trong phien PowerShell hien tai, chay:"
Write-Host "  .\$venvDir\Scripts\Activate.ps1"
