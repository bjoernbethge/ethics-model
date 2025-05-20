# Setup-Skript für EthicsModel mit CUDA (Windows)
# Führt uv sync aus und installiert das passende CUDA-Wheel für torch

# Check for uv
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Install uv first."
    exit
}

# 2. Abhängigkeiten synchronisieren
uv sync --extra full

# 3. CUDA-fähiges PyTorch installieren (hier: CUDA 12.8, Torch 2.7.0)
Write-Host "Installiere torch==2.7.0+cu128 ..."
uv pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

Write-Host "Done." 