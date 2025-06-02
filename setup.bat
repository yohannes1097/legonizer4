@echo off
REM Script untuk setup project Legonizer4 di Windows

echo Memeriksa versi Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python tidak ditemukan. Pastikan Python 3.8+ sudah terinstall.
    pause
    exit /b 1
)

echo Membuat virtual environment...
python -m venv venv

echo Mengaktifkan virtual environment...
call venv\Scripts\activate.bat

echo Menginstall dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Menginstall package dalam mode development...
pip install -e ".[dev]"

echo Membuat direktori yang dibutuhkan...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "data\models" mkdir "data\models"
if not exist "data\metrics\reports" mkdir "data\metrics\reports"

echo.
echo Setup selesai!
echo Untuk mengaktifkan virtual environment, jalankan: venv\Scripts\activate.bat
echo Untuk menjalankan API server, jalankan: python -m src.api.main
pause
