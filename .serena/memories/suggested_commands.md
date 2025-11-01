# Suggested Commands for EthicsModel Development

## Package Management (REQUIRED: use uv only!)
```powershell
# Sync all dependencies (core + extras)
uv sync --extra full

# Add new dependency
uv add <package-name>

# Add dev dependency
uv add --dev <package-name>

# Add training dependency
uv add --extra train <package-name>

# Remove dependency
uv remove <package-name>
```

## CUDA Setup (Windows)
```powershell
# Initial CUDA setup (run once after cloning)
./setup_cuda_win.ps1

# Verify CUDA is available
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Testing
```powershell
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_activation.py

# Run with coverage
pytest --cov=ethics_model tests/

# Run in parallel (faster)
pytest -n auto tests/

# Run fast LLM training test
pytest tests/test_llm_training.py
```

## TensorBoard (Profiling & Logging)
```powershell
# Start TensorBoard (view test profiling data)
tensorboard --logdir=runs

# Tests automatically write profiling data to TensorBoard
```

## Development Workflow
```powershell
# 1. Make code changes
# 2. Run tests
pytest tests/

# 3. Check CUDA functionality (if using GPU features)
pytest tests/test_cuda.py

# 4. View profiling results
tensorboard --logdir=runs
```

## Git Commands (Windows PowerShell)
```powershell
# Status
git status

# Add files
git add .

# Commit
git commit -m "message"

# Push
git push

# Pull
git pull
```

## File Operations (PowerShell)
```powershell
# List directory
ls
dir

# Navigate
cd path\to\directory

# Find files
Get-ChildItem -Recurse -Filter "*.py"

# Search in files (grep equivalent)
Select-String -Path "*.py" -Pattern "search_term"
```

## Python Execution
```powershell
# Run Python script
python script.py

# Run module
python -m ethics_model.training

# Interactive Python
python
```

## Build & Publish
```powershell
# Build package
uv build

# The project uses GitHub Actions for publishing to PyPI
# See .github/workflows/python-publish.yml
```
