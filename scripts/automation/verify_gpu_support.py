#!/usr/bin/env python3
"""
Script to verify GPU support.

This script attempts to detect and use GPU resources to confirm that
the NVIDIA GPU is properly configured and accessible.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime


def run_command(cmd, capture_output=True, check=False):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture_output, text=True, check=check)
    
    if capture_output:
        print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
    
    return result


def check_nvidia_smi():
    """Check if nvidia-smi command is available and working."""
    try:
        print("Checking NVIDIA System Management Interface (nvidia-smi)...")
        result = subprocess.run(
            ["nvidia-smi"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print("[PASS] nvidia-smi is available and working")
        print("\nGPU Information:")
        print(result.stdout)
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"[FAIL] nvidia-smi failed: {e}")
        return False


def check_cuda_version():
    """Check CUDA version and availability."""
    try:
        print("\nChecking CUDA version...")
        result = subprocess.run(
            ["nvcc", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print("[PASS] CUDA toolkit is available")
        print("\nCUDA Version Information:")
        print(result.stdout)
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"[FAIL] CUDA toolkit check failed: {e}")
        return False


def check_pytorch_gpu():
    """Check if PyTorch can detect and use GPU."""
    try:
        print("\nChecking PyTorch GPU support...")
        # Create a temporary script to check PyTorch GPU
        temp_script = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Simple test to ensure computation works on GPU
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    z = torch.matmul(x, y)
    end_time.record()
    
    # Wait for everything to finish running
    torch.cuda.synchronize()
    print(f"Matrix multiplication time: {start_time.elapsed_time(end_time):.2f} ms")
    print("[PASS] PyTorch GPU computation successful")
else:
    print("[FAIL] CUDA is not available for PyTorch")
"""
        with open("pytorch_gpu_check.py", "w") as f:
            f.write(temp_script)
        
        # Run the temporary script
        result = subprocess.run(
            [sys.executable, "pytorch_gpu_check.py"],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if "[FAIL] CUDA is not available for PyTorch" in result.stdout:
            print("[FAIL] PyTorch cannot use GPU")
            return False
        else:
            return True
    except Exception as e:
        print(f"[FAIL] PyTorch GPU check failed: {e}")
        return False
    finally:
        # Clean up the temporary script
        if os.path.exists("pytorch_gpu_check.py"):
            os.remove("pytorch_gpu_check.py")


def check_tensorflow_gpu():
    """Check if TensorFlow can detect and use GPU."""
    try:
        print("\nChecking TensorFlow GPU support...")
        # Create a temporary script to check TensorFlow GPU
        temp_script = """
import tensorflow as tf
import time
print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Number of GPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"  {gpu}")
    
    # Simple test to ensure computation works on GPU
    print("Running test computation on GPU...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([10000, 1000])
        b = tf.random.normal([1000, 2000])
        start_time = time.time()
        c = tf.matmul(a, b)
        # Force execution
        result = c.numpy()
        end_time = time.time()
        print(f"Matrix multiplication time: {(end_time - start_time) * 1000:.2f} ms")
    print("[PASS] TensorFlow GPU computation successful")
else:
    print("[FAIL] No GPUs available for TensorFlow")
"""
        with open("tensorflow_gpu_check.py", "w") as f:
            f.write(temp_script)
        
        # Run the temporary script
        result = subprocess.run(
            [sys.executable, "tensorflow_gpu_check.py"],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if "[FAIL] No GPUs available for TensorFlow" in result.stdout:
            print("[FAIL] TensorFlow cannot use GPU")
            return False
        else:
            return True
    except Exception as e:
        print(f"[FAIL] TensorFlow GPU check failed: {e}")
        return False
    finally:
        # Clean up the temporary script
        if os.path.exists("tensorflow_gpu_check.py"):
            os.remove("tensorflow_gpu_check.py")


def check_numba_gpu():
    """Check if Numba can detect and use GPU via CUDA."""
    try:
        print("\nChecking Numba CUDA support...")
        # Create a temporary script to check Numba CUDA
        temp_script = """
from numba import cuda
import numpy as np
import time

print(f"Numba version: {numba.__version__}")
if cuda.is_available():
    print("CUDA is available for Numba")
    print(f"Number of CUDA devices: {cuda.get_num_devices()}")
    for i in range(cuda.get_num_devices()):
        d = cuda.get_current_device()
        print(f"Device {i}: {d.name} (Compute Capability {d.compute_capability[0]}.{d.compute_capability[1]})")
    
    # Simple test to ensure computation works on GPU
    @cuda.jit
    def cuda_add(a, b, c):
        i = cuda.grid(1)
        if i < a.shape[0]:
            c[i] = a[i] + b[i]
    
    print("Running test computation on GPU...")
    n = 10000000
    a = np.ones(n, dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)
    
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    start_time = time.time()
    cuda_add[blocks_per_grid, threads_per_block](a, b, c)
    cuda.synchronize()
    end_time = time.time()
    
    print(f"Numba CUDA computation time: {(end_time - start_time) * 1000:.2f} ms")
    # Verify result
    if np.all(c == 2.0):
        print("[PASS] Numba CUDA computation successful")
    else:
        print("[FAIL] Numba CUDA computation produced incorrect results")
else:
    print("[FAIL] CUDA is not available for Numba")
"""
        with open("numba_gpu_check.py", "w") as f:
            f.write(temp_script)
        
        # Run the temporary script
        result = subprocess.run(
            [sys.executable, "numba_gpu_check.py"],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if "[FAIL] CUDA is not available for Numba" in result.stdout:
            print("[FAIL] Numba cannot use GPU")
            return False
        else:
            return True
    except Exception as e:
        print(f"[FAIL] Numba GPU check failed: {e}")
        return False
    finally:
        # Clean up the temporary script
        if os.path.exists("numba_gpu_check.py"):
            os.remove("numba_gpu_check.py")


def run_simple_benchmark():
    """Run a simple benchmark comparing CPU vs GPU performance."""
    try:
        print("\nRunning simple CPU vs GPU benchmark...")
        # Create a benchmark script
        benchmark_script = """
import torch
import numpy as np
import time

# Matrix size
size = 5000

print(f"Benchmark: Matrix multiplication with size {size}x{size}")

# CPU benchmark with NumPy
print("\\nRunning on CPU with NumPy...")
a_np = np.random.rand(size, size).astype(np.float32)
b_np = np.random.rand(size, size).astype(np.float32)

start_time = time.time()
c_np = np.matmul(a_np, b_np)
end_time = time.time()
cpu_time = end_time - start_time
print(f"CPU time: {cpu_time:.4f} seconds")

# GPU benchmark with PyTorch
if torch.cuda.is_available():
    print("\\nRunning on GPU with PyTorch...")
    a_torch = torch.tensor(a_np, device='cuda')
    b_torch = torch.tensor(b_np, device='cuda')
    
    # Warm-up run
    _ = torch.matmul(a_torch, b_torch)
    torch.cuda.synchronize()
    
    start_time = time.time()
    c_torch = torch.matmul(a_torch, b_torch)
    torch.cuda.synchronize()
    end_time = time.time()
    gpu_time = end_time - start_time
    print(f"GPU time: {gpu_time:.4f} seconds")
    
    # Calculate speedup
    speedup = cpu_time / gpu_time
    print(f"\\nGPU speedup: {speedup:.2f}x faster than CPU")
    
    # Verify results match (approximately)
    c_torch_cpu = c_torch.cpu().numpy()
    max_diff = np.max(np.abs(c_np - c_torch_cpu))
    print(f"Maximum difference between CPU and GPU results: {max_diff:.6f}")
else:
    print("\\nGPU benchmark skipped: CUDA not available")
"""
        with open("gpu_benchmark.py", "w") as f:
            f.write(benchmark_script)
        
        # Run the benchmark
        result = subprocess.run(
            [sys.executable, "gpu_benchmark.py"],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        return True
    except Exception as e:
        print(f"[FAIL] Benchmark failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists("gpu_benchmark.py"):
            os.remove("gpu_benchmark.py")


def save_report(results):
    """Save the GPU verification report."""
    # Create output directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "summary": {
            "success": all(results.values()),
            "passed_tests": sum(1 for v in results.values() if v),
            "total_tests": len(results)
        }
    }
    
    # Save the report
    report_file = f"reports/gpu_verification_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to {report_file}")


def main():
    """Run all GPU verification checks."""
    print("=" * 80)
    print("GPU SUPPORT VERIFICATION")
    print("=" * 80)
    
    results = {}
    
    # Run the checks
    results["nvidia_smi"] = check_nvidia_smi()
    results["cuda_version"] = check_cuda_version()
    results["pytorch_gpu"] = check_pytorch_gpu()
    results["tensorflow_gpu"] = check_tensorflow_gpu()
    results["numba_gpu"] = check_numba_gpu()
    
    # Run benchmark if basic checks pass
    if all([results["nvidia_smi"], results["cuda_version"]]):
        results["benchmark"] = run_simple_benchmark()
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for test, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test.ljust(20)}: {status}")
    
    overall = all(results.values())
    print("\nOverall GPU Support: " + ("[PASS] PROPERLY CONFIGURED" if overall else "[FAIL] NOT PROPERLY CONFIGURED"))
    
    # Save report
    save_report(results)
    
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main()) 