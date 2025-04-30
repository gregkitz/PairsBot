# GPU Support for Docker Containers

This document explains how to configure, verify, and troubleshoot GPU support in our Docker-based development environment.

## Requirements

To use GPU acceleration in Docker containers, you need:

1. A CUDA-compatible NVIDIA GPU
2. Appropriate NVIDIA drivers installed on the host
3. NVIDIA Container Toolkit (nvidia-docker)
4. Docker Desktop (Windows/macOS) or Docker Engine (Linux)

## Verification

The system includes several scripts to verify GPU support:

- `scripts/automation/start_dev_environment.ps1` - Automatically checks GPU support during startup
- `scripts/automation/docker_gpu_test.py` - Standalone script to test Docker GPU support
- `scripts/automation/verify_gpu_support.py` - Verifies GPU support on the host system
- `scripts/temp_gpu_test.py` - Tests GPU inside containers for both PyTorch and TensorFlow

## Docker Configuration

Our Docker setup is configured to use GPU acceleration via:

1. Base image: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
2. GPU runtime configuration in docker-compose.yml using the modern device specification
3. Environment variables in the containers: `NVIDIA_VISIBLE_DEVICES=all` and `NVIDIA_DRIVER_CAPABILITIES=compute,utility`

## Troubleshooting

If GPU support is not working, check the following:

### Host System Issues:

1. Verify NVIDIA drivers are installed:
   ```
   nvidia-smi
   ```
   This should display your GPU and driver version.

2. Check NVIDIA Container Toolkit installation:
   ```
   docker info | findstr "Runtimes"
   ```
   You should see the NVIDIA runtime listed.

### Container Issues:

1. Test basic GPU access in containers:
   ```
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

2. Check GPU access in specific service containers:
   ```
   docker-compose up -d
   docker cp scripts/temp_gpu_test.py quant-trader-worker-1:/app/
   docker exec -it quant-trader-worker-1 python /app/temp_gpu_test.py
   ```

3. Check container GPU environment variables:
   ```
   docker exec -it quant-trader-worker-1 env | grep -i nvidia
   ```

## Common Issues

### 1. GPU Not Detected in Containers

- **Solution**: Ensure containers are started with the `--gpus all` flag or proper device configuration in docker-compose.yml
- Check that environment variables are set correctly
- Verify container has access permissions to the GPU devices

### 2. CUDA Version Mismatch

- **Solution**: Ensure CUDA version in containers is compatible with host driver version
- Update Dockerfile to use compatible CUDA base image

### 3. Memory Issues

- **Solution**: Adjust memory limits in docker-compose.yml
- Close other GPU-intensive applications

### 4. Library Version Conflicts

- **Solution**: Check library compatibility (PyTorch, TensorFlow, CUDA, cuDNN)
- Specify explicit versions in requirements.txt

## Performance Tuning

For optimal GPU performance:

1. Set appropriate resource limits in docker-compose.yml
2. Use cached volume mounts for improved I/O performance
3. Use the WSL2 backend on Windows (set in Docker Desktop settings)
4. Ensure proper cooling for sustained GPU workloads

## Docker Compose Configuration

Our docker-compose.yml uses the modern device specification format:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

This format is preferred over the older `runtime: nvidia` setting.

## Testing with Different ML Frameworks

The system has been tested with:

- PyTorch - Confirmed working with CUDA
- TensorFlow - Confirmed working with CUDA
- Numba - Supported through CUDA libraries

## Logs and Debugging

GPU-related logs are stored in:

- `logs/worker/` - For worker container GPU operations
- `logs/api/` - For API container GPU operations

Use `docker exec -it <container> nvidia-smi` to monitor GPU usage in real-time. 