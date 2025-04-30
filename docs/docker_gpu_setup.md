# Docker GPU Setup Guide

This guide explains how to set up and verify GPU support in our Docker containers.

## Prerequisites

Before running our Docker containers with GPU support, you need to have:

1. **NVIDIA GPU** - We've optimized for the NVIDIA 4080 GPU
2. **NVIDIA Drivers** - Latest version recommended (minimum driver version for CUDA 11.8: 450.80.02)
3. **Docker Desktop** - Latest version with WSL2 backend on Windows
4. **NVIDIA Container Toolkit** - Required for GPU access in Docker containers

## Installation Steps

### 1. Install NVIDIA Drivers

Ensure you have the latest NVIDIA drivers installed for your GPU:

- **Windows**: Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- **Linux**: Use your distribution's package manager or NVIDIA's installer

Verify the installation by running `nvidia-smi` in your terminal. You should see information about your GPU, including the driver version.

### 2. Install Docker Desktop

Download and install Docker Desktop from [Docker's website](https://www.docker.com/products/docker-desktop).

#### Windows Configuration

For Windows, ensure Docker Desktop is configured to use WSL2:
1. Open Docker Desktop
2. Go to Settings > General
3. Ensure "Use the WSL 2 based engine" is checked
4. Go to Settings > Resources > WSL Integration
5. Enable integration with your WSL distro

### 3. Install NVIDIA Container Toolkit

#### Windows

For Windows with WSL2 backend, follow these steps:

1. Open your WSL2 Linux distribution
2. Run the following commands:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### Linux

For native Linux, run:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list \
   && sudo apt-get update \
   && sudo apt-get install -y nvidia-container-toolkit \
   && sudo systemctl restart docker
```

## Verification

We've created automated scripts to verify GPU support in Docker containers:

### Quick Verification

Run the following command to quickly check if NVIDIA Docker is properly configured:

```bash
python scripts/automation/docker_gpu_test.py --nvidia-only
```

If successful, you'll see:
```
✅ NVIDIA Docker is properly configured!
```

### Full Verification

Run the full verification test, which builds and runs a test container:

```bash
python scripts/automation/docker_gpu_test.py
```

This will:
1. Check NVIDIA Docker setup
2. Build a test container with PyTorch
3. Run a GPU computation in the container
4. Report whether GPU access is working correctly

### Test Our Project Containers

Verify GPU support in our project-specific containers:

```bash
python scripts/automation/docker_gpu_test.py --our-containers
```

This tests GPU support in our worker container.

### Comprehensive GPU Verification

For a more detailed GPU functionality check, run:

```bash
python scripts/automation/verify_gpu_support.py
```

This checks:
- NVIDIA-SMI availability
- CUDA version
- PyTorch GPU support
- TensorFlow GPU support
- Numba CUDA support
- GPU vs CPU performance benchmarks

## Running Our Containers with GPU Support

Our `docker-compose.yml` has been configured to use the NVIDIA runtime for API and worker services. To start all services:

```bash
docker-compose up
```

To run a specific service:

```bash
docker-compose up worker
```

To run a one-off command in a container with GPU access:

```bash
docker-compose run --rm worker python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

### Common Issues

#### NVIDIA-SMI not found in container

**Problem:**
```
nvidia-smi: command not found
```

**Solution:**
Make sure the NVIDIA Container Toolkit is properly installed and `runtime: nvidia` is specified in docker-compose.yml.

#### CUDA not available in PyTorch

**Problem:**
```
CUDA available: False
```

**Solution:**
1. Check that the correct NVIDIA driver is installed
2. Ensure NVIDIA Container Toolkit is properly configured
3. Verify that the container has GPU access with the `--gpus all` flag or `runtime: nvidia`

#### Docker reports "unknown runtime: nvidia"

**Problem:**
```
docker: Error response from daemon: unknown runtime specified nvidia.
```

**Solution:**
1. Make sure NVIDIA Container Toolkit is installed
2. Restart Docker service
3. On Linux, check that `/etc/docker/daemon.json` contains the NVIDIA runtime

#### TensorFlow doesn't see GPU

**Problem:**
```
No GPUs available for TensorFlow
```

**Solution:**
1. Check TensorFlow version compatibility with your CUDA version
2. Set proper environment variables in the container:
   ```
   NVIDIA_VISIBLE_DEVICES=all
   NVIDIA_DRIVER_CAPABILITIES=compute,utility
   ```

## Performance Tips

1. **Memory Allocation**: Our Docker configuration allocates 32GB of RAM to worker containers, which is optimized for our workloads with the 64GB system. If you have less RAM, adjust accordingly.

2. **Concurrency**: Worker concurrency is set to 8, optimized for our i9 processor. Adjust based on your CPU.

3. **Volume Mounting**: We use `:cached` mount mode for better performance. This is especially important for large datasets.

4. **GPU Memory**: Monitor GPU memory usage with `nvidia-smi` and adjust batch sizes in ML workloads to prevent out-of-memory errors.

## Further Reading

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)
- [Docker Compose GPU Configuration](https://docs.docker.com/compose/gpu-support/)
- [PyTorch GPU Guide](https://pytorch.org/docs/stable/notes/cuda.html)
- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu) 