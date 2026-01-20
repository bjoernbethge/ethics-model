FROM python:3.14-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install UV for package management (as used in the project)
RUN curl -sSf https://sh.uv.dev | sh

# Add UV to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy the project files
COPY . .

# Install dependencies using UV
# The --extra full flag installs all dependencies including dev dependencies
RUN uv sync --extra full

# Install CUDA-enabled PyTorch (as per README instructions)
RUN uv pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

# Install bitsandbytes for Windows (as specified in pyproject.toml)
RUN uv pip install --force-reinstall "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl"

# Install SpaCy language model for GraphBrain
RUN python -m spacy download en_core_web_sm

# Test CUDA availability (will output to build logs)
RUN python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'CUDA Available:', torch.cuda.is_available())"

# Set up working directory for development
WORKDIR /app

# Command to keep the container running
CMD ["bash"]
