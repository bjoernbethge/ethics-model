# Setup-Skript für EthicsModel mit CUDA (Windows)
# Führt uv sync aus und installiert das passende CUDA-Wheel für torch

# Check for uv
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Install uv first."
    exit
}

# 2. Abhängigkeiten synchronisieren
uv sync --extra full

# 3. CUDA-fähiges PyTorch installieren (hier: CUDA 12.6, Torch 2.6.0)
Write-Host "Installiere torch==2.6.0+cu126 ..."
uv pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

# 4. bitsandbytes installieren (Windows Wheel)
Write-Host "Installiere bitsandbytes ..."
uv pip install --force-reinstall "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-win_amd64.whl"
# Note, if you don't want to reinstall BNBs dependencies, append the `--no-deps` flag!
pip install --force-reinstall 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-win_amd64.whl'
Write-Host "Done." 