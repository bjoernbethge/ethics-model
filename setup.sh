#!/bin/bash
# Setup-Skript für EthicsModel mit CUDA (Linux)
# Führt uv sync aus und installiert das passende CUDA-Wheel für torch

set -e

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "uv not found. Install uv first."
    exit 1
fi

# 2. Abhängigkeiten synchronisieren
uv sync --extra full

# 3. CUDA-fähiges PyTorch installieren (hier: CUDA 12.8, Torch 2.7.0)
echo "Installiere torch==2.7.0+cu128 ..."
uv pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

echo "Done." 