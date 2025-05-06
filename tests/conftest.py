import os
# CUDA-Umgebungsvariablen für bitsandbytes und CUDA-Toolkit
os.environ["BNB_CUDA_VERSION"] = "121"
os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME", "D:/dev/nvidia")
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

import pytest
import tempfile
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.profiler

@pytest.fixture
def summary_writer():
    with tempfile.TemporaryDirectory() as logdir:
        writer = SummaryWriter(logdir)
        yield writer
        writer.close()

@pytest.fixture
def cpu_or_cuda_profiler():
    # Prüfe, ob CUDA verfügbar ist
    if torch.cuda.is_available():
        # Setze BNB_CUDA_VERSION passend zur installierten CUDA-Version
        cuda_version = torch.version.cuda
        if cuda_version:
            os.environ["BNB_CUDA_VERSION"] = cuda_version.replace(".", "")
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    else:
        activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        yield prof 