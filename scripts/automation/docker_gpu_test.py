#!/usr/bin/env python3
"""
Script to test GPU access in Docker containers.

This script creates and runs a Docker container to verify that 
NVIDIA GPU is accessible from inside containers.
"""

import subprocess
import sys
import argparse
import time
import os
from pathlib import Path


def run_command(cmd, capture_output=True, check=False):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture_output, text=True, check=check)
    
    if capture_output:
        print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
    
    return result


def check_gpu_with_container(image_name, command):
    """Run a Docker container to check GPU access."""
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        image_name,
        *command
    ]
    
    return run_command(cmd)


def check_nvidia_docker():
    """Check if nvidia-docker is properly installed."""
    print("\n=== Checking NVIDIA Docker Setup ===")
    
    # Check docker version
    print("\nChecking Docker version:")
    run_command(["docker", "--version"], check=False)
    
    # Check if nvidia-docker is installed
    print("\nChecking NVIDIA Docker:")
    run_command(["docker", "info"], check=False)
    
    # Check NVIDIA runtime
    print("\nVerifying NVIDIA container toolkit:")
    cmd = ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.8.0-base-ubuntu22.04", "nvidia-smi"]
    result = run_command(cmd, check=False)
    
    if result.returncode != 0:
        print("❌ NVIDIA Docker is not properly configured.")
        print("Make sure you have installed the NVIDIA Container Toolkit and Docker is properly configured.")
        print("See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
        return False
    else:
        print("✅ NVIDIA Docker is properly configured!")
        return True


def build_test_container():
    """Build a test container with PyTorch."""
    print("\n=== Building GPU Test Container ===")
    
    # Create a temporary directory
    tmp_dir = Path("./docker_gpu_test")
    tmp_dir.mkdir(exist_ok=True)
    
    # Create a Dockerfile
    dockerfile = tmp_dir / "Dockerfile"
    dockerfile.write_text("""
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    curl \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Create a test script
COPY test_gpu.py /test_gpu.py

CMD ["python", "/test_gpu.py"]
""")
    
    # Create a test script
    test_script = tmp_dir / "test_gpu.py"
    test_script.write_text("""
import torch
import time

print("PyTorch GPU Test")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Run a simple GPU computation
    print("\\nRunning GPU computation test...")
    
    # Create tensors on GPU
    a = torch.randn(5000, 5000, device='cuda')
    b = torch.randn(5000, 5000, device='cuda')
    
    # Warmup
    torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Matrix multiplication time: {(end_time - start_time) * 1000:.2f} ms")
    print("GPU test completed successfully!")
else:
    print("❌ CUDA is not available")
    print("Make sure your Docker container has GPU access.")
""")
    
    # Build the container
    run_command(["docker", "build", "-t", "gpu-test:latest", str(tmp_dir)], check=False)
    
    # Clean up
    dockerfile.unlink()
    test_script.unlink()
    tmp_dir.rmdir()
    
    return "gpu-test:latest"


def test_our_containers():
    """Test our own Docker containers for GPU access."""
    print("\n=== Testing Our Docker Containers ===")
    
    # First, build our containers
    print("\nBuilding our containers:")
    run_command(["docker-compose", "build"], check=False)
    
    # Test GPU access in worker container
    print("\nTesting worker container:")
    cmd = [
        "docker-compose", "run", "--rm", "worker",
        "python", "-c", 
        "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); "
        "print(f'GPU count: {torch.cuda.device_count()}'); "
        "[print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
    ]
    run_command(cmd, check=False)
    
    # Test TensorFlow in worker container
    print("\nTesting TensorFlow in worker container:")
    cmd = [
        "docker-compose", "run", "--rm", "worker",
        "python", "-c", 
        "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); "
        "gpus = tf.config.list_physical_devices('GPU'); "
        "print(f'GPUs available: {len(gpus)}'); "
        "[print(f'GPU: {gpu}') for gpu in gpus]"
    ]
    run_command(cmd, check=False)


def main():
    """Main function to run GPU tests."""
    parser = argparse.ArgumentParser(description="Test GPU access in Docker containers")
    parser.add_argument("--build-only", action="store_true", help="Only build test container without running tests")
    parser.add_argument("--nvidia-only", action="store_true", help="Only check NVIDIA Docker setup")
    parser.add_argument("--our-containers", action="store_true", help="Test our own Docker containers")
    args = parser.parse_args()
    
    print("=" * 80)
    print("DOCKER GPU ACCESS VERIFICATION")
    print("=" * 80)
    
    # Check NVIDIA Docker setup
    if args.nvidia_only:
        success = check_nvidia_docker()
        return 0 if success else 1
    
    # Check if our containers have GPU access
    if args.our_containers:
        test_our_containers()
        return 0
    
    # Build and test the container
    if not args.build_only:
        success = check_nvidia_docker()
        if not success:
            return 1
    
    # Build the test container
    image_name = build_test_container()
    
    if args.build_only:
        print(f"\n✅ Built test container: {image_name}")
        print("You can run it with: docker run --rm --gpus all gpu-test:latest")
        return 0
    
    # Run the test container
    print("\n=== Running GPU Test Container ===")
    result = check_gpu_with_container(image_name, ["python", "/test_gpu.py"])
    
    # Check if the test was successful
    if "GPU test completed successfully!" in result.stdout:
        print("\n✅ GPU access in Docker is working correctly!")
        return 0
    else:
        print("\n❌ GPU access in Docker is not working correctly!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 