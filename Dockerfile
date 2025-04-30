FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies with minimal layer size
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set environment variables for CUDA
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Copy and install requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Celery, Flower, and Redis with their own layer
RUN pip install --no-cache-dir celery flower redis

# Install GPU-accelerated packages
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    tensorflow==2.15.0 \
    numba

# Create directories for data persistence
RUN mkdir -p /app/data /app/models /app/output /app/logs

# Copy application code (after dependencies to leverage caching)
COPY . .

# Expose ports for API, Flower monitoring
EXPOSE 8000 5555

# Default command
# This will be overridden in docker-compose.yml for different services
CMD ["python", "-m", "src.api.main"] 