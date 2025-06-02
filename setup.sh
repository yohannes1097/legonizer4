#!/bin/bash

# Script untuk setup project Legonizer4

# Pastikan Python 3.8+ terinstall
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$python_version 3.8" | awk '{print ($1 < $2)}') )); then
    echo "Error: Python 3.8 atau lebih baru dibutuhkan (terinstall: $python_version)"
    exit 1
fi

# Buat virtual environment
echo "Membuat virtual environment..."
python3 -m venv venv

# Aktifkan virtual environment
echo "Mengaktifkan virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Menginstall dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install package dalam mode development
echo "Menginstall package dalam mode development..."
pip install -e ".[dev]"

# Buat direktori yang dibutuhkan
echo "Membuat direktori yang dibutuhkan..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/models
mkdir -p data/metrics/reports

echo "Setup selesai!"
echo "Untuk mengaktifkan virtual environment, jalankan: source venv/bin/activate"
echo "Untuk menjalankan API server, jalankan: python -m src.api.main"
