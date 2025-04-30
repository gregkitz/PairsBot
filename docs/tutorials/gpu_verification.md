# GPU Support Verification Tutorial

This tutorial walks through the process of verifying GPU support in your containerized trading environment. Follow these steps to ensure your system is properly configured to utilize GPU acceleration for machine learning and data processing tasks.

## Prerequisites

Before beginning this tutorial, ensure you have:

1. NVIDIA GPU installed in your system
2. NVIDIA drivers installed (version 470.x or newer recommended)
3. Docker Desktop (Windows/macOS) or Docker Engine (Linux) installed
4. NVIDIA Container Toolkit (nvidia-docker) installed

## Step 1: Verify Host GPU

First, verify that your host system can detect the NVIDIA GPU:

```powershell
nvidia-smi
```

You should see output similar to:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.09              Driver Version: 571.96         CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4080 ...    On  |   00000000:01:00.0  On |                  N/A |
|  0%   40C    P8              9W /  320W |     567MiB /  16376MiB |     10%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

If you don't see this output, check that your NVIDIA drivers are properly installed.

## Step 2: Verify Docker GPU Support

Next, verify that Docker can access the GPU through the NVIDIA Container Toolkit:

```powershell
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

If this command runs successfully with output similar to Step 1, Docker is properly configured to access your GPU.

## Step 3: Use the Automated Environment Startup

The system includes an automated startup script that handles GPU detection and configuration:

```powershell
powershell -File scripts/automation/start_dev_environment.ps1
```

This script will:
- Check for prerequisites
- Verify GPU availability
- Build and start all containers with proper GPU configuration

Look for messages confirming GPU support:
```
Checking GPU support in Docker...
NVIDIA GPU detected on host system!
GPU access in Docker containers verified!
```

## Step 4: Start the Containers

If you prefer to start the containers manually, use:

```powershell
docker-compose up -d
```

This starts all services defined in the docker-compose.yml file with the GPU configuration we've set up.

## Step 5: Verify GPU Access in Containers

### Basic GPU Verification

To quickly check if a container has GPU access:

```powershell
docker exec -it quant-trader-worker-1 nvidia-smi
```

You should see output similar to Step 1.

### Comprehensive GPU Test

For a more thorough test that verifies frameworks can access the GPU:

```powershell
# Copy the test script to the container
docker cp scripts/temp_gpu_test.py quant-trader-worker-1:/app/

# Run the test script
docker exec -it quant-trader-worker-1 python /app/temp_gpu_test.py
```

This test will check:
- System environment variables
- PyTorch GPU access
- TensorFlow GPU access
- Matrix multiplication performance on GPU

Look for these confirmations in the output:
```
CUDA available: True
PyTorch GPU test successful!
TensorFlow GPU test successful!
```

## Step 6: Monitor GPU Usage

You can monitor GPU usage while running workloads with:

```powershell
docker exec -it quant-trader-worker-1 nvidia-smi -l 1
```

This shows GPU usage statistics updated every 1 second (press Ctrl+C to exit).

## Troubleshooting

If you encounter issues, check the following:

### No GPU Detection in Docker

If the basic `nvidia-smi` test fails in Docker:

1. Verify NVIDIA Container Toolkit installation:
   ```powershell
   docker info | findstr "Runtimes"
   ```

2. Check if Docker is configured to use the NVIDIA runtime:
   ```powershell
   cat $env:USERPROFILE\.docker\daemon.json
   ```

   It should contain:
   ```json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```

### Frameworks Can't Access GPU

If containers start but PyTorch/TensorFlow can't access the GPU:

1. Check container environment variables:
   ```powershell
   docker exec -it quant-trader-worker-1 env
   ```

   Look for:
   ```
   NVIDIA_VISIBLE_DEVICES=all
   NVIDIA_DRIVER_CAPABILITIES=compute,utility
   ```

2. Verify CUDA libraries are installed in the container:
   ```powershell
   docker exec -it quant-trader-worker-1 ls -la /usr/local/cuda
   ```

### Performance Issues

If GPU acceleration works but performance is poor:

1. Check if other processes are using the GPU:
   ```powershell
   nvidia-smi
   ```

2. Verify container resource limits in docker-compose.yml

## Conclusion

When all steps complete successfully, your containerized trading environment is properly configured to utilize GPU acceleration. This will significantly improve performance for machine learning model training and inference, as well as for computationally intensive data processing tasks.

For more detailed information, see:
- `docs/context/gpu_setup.md` - Complete GPU setup and troubleshooting guide
- `docs/context/environment_status.md` - Current environment status report
