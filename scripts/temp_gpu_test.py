#!/usr/bin/env python3
"""
Temporary GPU test script.
"""

import sys
import time
import os

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

# System info
print_section("SYSTEM INFO")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Check environment variables
print_section("ENVIRONMENT VARIABLES")
gpu_vars = ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"]
for var in gpu_vars:
    if var in os.environ:
        print(f"{var}: {os.environ[var]}")
    else:
        print(f"{var}: Not set")

# PyTorch test
print_section("PYTORCH GPU TEST")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Run a simple test
        print("\nRunning PyTorch GPU computation test...")
        a = torch.randn(5000, 5000, device='cuda')
        b = torch.randn(5000, 5000, device='cuda')
        
        # Warmup
        torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Test
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
        print(f"Matrix multiplication time: {(end-start)*1000:.2f} ms")
        print("PyTorch GPU test successful!")
    else:
        print("PyTorch cannot use GPU")
except ImportError:
    print("PyTorch not installed")
except Exception as e:
    print(f"PyTorch test error: {e}")

# TensorFlow test
print_section("TENSORFLOW GPU TEST")
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"  {gpu}")
    
    if gpus:
        print("\nRunning TensorFlow GPU computation test...")
        with tf.device('/GPU:0'):
            a = tf.random.normal([5000, 5000])
            b = tf.random.normal([5000, 5000])
            start = time.time()
            c = tf.matmul(a, b)
            result = c.numpy()  # Force execution
            end = time.time()
            print(f"Matrix multiplication time: {(end-start)*1000:.2f} ms")
            print("TensorFlow GPU test successful!")
    else:
        print("TensorFlow cannot use GPU")
except ImportError:
    print("TensorFlow not installed")
except Exception as e:
    print(f"TensorFlow test error: {e}")

# Summary
print_section("SUMMARY")
print("GPU tests completed.") 