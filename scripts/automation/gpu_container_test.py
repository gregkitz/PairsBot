#!/usr/bin/env python3
"""
Simple script to test GPU availability inside a Docker container.
This script will check if the container can access NVIDIA GPUs.
"""

import os
import sys
import subprocess
import time

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def check_system_info():
    """Print basic system information."""
    print_section("SYSTEM INFORMATION")
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # Environment variables
    print("\nRelevant environment variables:")
    for var in ['CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'PATH']:
        if var in os.environ:
            print(f"  {var}: {os.environ[var]}")
        else:
            print(f"  {var}: Not set")

def run_command(cmd):
    """Run a shell command and print its output."""
    print(f"\nRunning: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

def check_nvidia_smi():
    """Check if nvidia-smi is available and working."""
    print_section("NVIDIA-SMI TEST")
    return run_command(["nvidia-smi"])

def check_cuda_version():
    """Check CUDA version."""
    print_section("CUDA VERSION")
    
    # Check if nvcc is available
    nvcc_available = run_command(["nvcc", "--version"])
    
    # Check CUDA version through Python packages
    try:
        import torch
        print("\nPyTorch CUDA information:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("\nPyTorch not installed")

    try:
        import tensorflow as tf
        print("\nTensorFlow CUDA information:")
        print(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Number of GPUs available: {len(gpus)}")
        for gpu in gpus:
            print(f"  {gpu}")
    except ImportError:
        print("\nTensorFlow not installed")
    
    return True

def run_pytorch_test():
    """Run a simple PyTorch test on GPU."""
    print_section("PYTORCH GPU TEST")
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA is not available for PyTorch")
            return False
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Create tensors on GPU
        print("\nRunning GPU computation test...")
        
        # Tensor dimensions
        n = 5000
        
        # Create tensors on GPU
        a = torch.randn(n, n, device='cuda')
        b = torch.randn(n, n, device='cuda')
        
        # Warmup
        torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"Matrix multiplication time: {(end_time - start_time) * 1000:.2f} ms")
        print("✅ PyTorch GPU test completed successfully!")
        return True
    
    except ImportError:
        print("PyTorch not installed")
        return False
    except Exception as e:
        print(f"Error during PyTorch test: {e}")
        return False

def run_tensorflow_test():
    """Run a simple TensorFlow test on GPU."""
    print_section("TENSORFLOW GPU TEST")
    
    try:
        import tensorflow as tf
        
        print(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            print("No GPUs available for TensorFlow")
            return False
        
        print(f"Number of GPUs available: {len(gpus)}")
        for gpu in gpus:
            print(f"  {gpu}")
        
        # Simple test to ensure computation works on GPU
        print("\nRunning test computation on GPU...")
        with tf.device('/GPU:0'):
            a = tf.random.normal([5000, 5000])
            b = tf.random.normal([5000, 5000])
            start_time = time.time()
            c = tf.matmul(a, b)
            # Force execution
            result = c.numpy()
            end_time = time.time()
            print(f"Matrix multiplication time: {(end_time - start_time) * 1000:.2f} ms")
        print("✅ TensorFlow GPU test completed successfully!")
        return True
    
    except ImportError:
        print("TensorFlow not installed")
        return False
    except Exception as e:
        print(f"Error during TensorFlow test: {e}")
        return False

def main():
    """Main function to run all GPU tests."""
    print_section("DOCKER GPU TEST")
    print("Testing GPU availability inside Docker container")
    
    # Check system information
    check_system_info()
    
    # Check nvidia-smi
    nvidia_smi_ok = check_nvidia_smi()
    
    # Check CUDA version
    cuda_version_ok = check_cuda_version()
    
    # Run PyTorch test
    pytorch_ok = run_pytorch_test()
    
    # Run TensorFlow test
    tensorflow_ok = run_tensorflow_test()
    
    # Print summary
    print_section("TEST SUMMARY")
    print(f"nvidia-smi available: {'✅ Yes' if nvidia_smi_ok else '❌ No'}")
    print(f"CUDA version check: {'✅ Passed' if cuda_version_ok else '❌ Failed'}")
    print(f"PyTorch GPU test: {'✅ Passed' if pytorch_ok else '❌ Failed'}")
    print(f"TensorFlow GPU test: {'✅ Passed' if tensorflow_ok else '❌ Failed'}")
    
    # Determine overall status
    if nvidia_smi_ok and (pytorch_ok or tensorflow_ok):
        print("\n✅ GPU is properly configured and accessible in Docker!")
        return 0
    else:
        print("\n❌ GPU configuration issues detected.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 