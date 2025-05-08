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

# 3. CUDA-fähiges PyTorch installieren (hier: CUDA 12.6, Torch 2.6.0)
echo "Installiere torch==2.6.0+cu126 ..."
uv pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

# 4. bitsandbytes installieren (Multi-Backend, CUDA, ROCm, Intel, Apple Silicon)
echo "Installiere bitsandbytes ..."
uv pip install --force-reinstall 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl'

echo "Done." 