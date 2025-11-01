# Task Completion Checklist

When a development task is completed, follow these steps:

## 1. Run Tests
```powershell
# Run all tests to ensure nothing broke
pytest tests/

# Or run specific test file if only specific module changed
pytest tests/test_<module>.py
```

## 2. Verify CUDA (if GPU code was modified)
```powershell
pytest tests/test_cuda.py
```

## 3. Check Type Annotations
- Ensure all new functions have proper type hints
- Verify return types are specified

## 4. Code Review Self-Check
- [ ] All imports at top level (no try/except imports)
- [ ] No version checks or compatibility wrappers
- [ ] Type hints present on all functions
- [ ] Error messages are descriptive
- [ ] Tests write profiling data to TensorBoard

## 5. Test Coverage (Optional but Recommended)
```powershell
pytest --cov=ethics_model tests/
```

## 6. TensorBoard Verification (Optional)
```powershell
# Start TensorBoard to view profiling results
tensorboard --logdir=runs
```

## 7. Commit Changes
```powershell
git add .
git commit -m "Descriptive commit message"
```

## Notes
- **NEVER use pip** - always use `uv` for package management
- Tests automatically include profiling via fixtures in `conftest.py`
- All test functions receive `summary_writer` and `cpu_or_cuda_profiler` fixtures
- CUDA setup requires running `setup_cuda_win.ps1` on Windows
