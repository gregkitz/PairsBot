#!/usr/bin/env python3
"""
Script to automatically optimize Celery configuration based on system hardware.

This script analyzes the available CPU, memory, and GPU resources and generates
optimal Celery worker configuration for the local machine.
"""

import os
import platform
import subprocess
import sys
import psutil
import json
from pathlib import Path


def get_cpu_info():
    """Get CPU information and recommend optimal worker count."""
    cpu_count = os.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Get CPU model information
    if platform.system() == "Windows":
        from subprocess import PIPE, run
        cmd = "wmic cpu get name"
        result = run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
        cpu_model = result.stdout.strip().split("\n")[1]
    else:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('model name'):
                    cpu_model = line.split(':', 1)[1].strip()
                    break
            else:
                cpu_model = "Unknown CPU"
    
    # Calculate optimal worker count (n-1 for i9 processors, n-2 for others)
    if "i9" in cpu_model:
        optimal_workers = max(1, cpu_count - 1)
    else:
        optimal_workers = max(1, cpu_count - 2)
    
    return {
        "cpu_model": cpu_model,
        "cpu_count": cpu_count,
        "cpu_usage": f"{cpu_percent}%",
        "recommended_workers": optimal_workers,
        "recommended_threads_per_worker": 2,
    }


def get_memory_info():
    """Get memory information and recommend optimal memory per worker."""
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # GB
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
    
    # Reserve 8GB for OS and other processes
    usable_memory = max(0, total_memory - 8)
    
    # Determine worker count from CPU info to calculate memory per worker
    cpu_info = get_cpu_info()
    worker_count = cpu_info["recommended_workers"]
    
    # Memory per worker in GB (leaving some overhead)
    memory_per_worker = usable_memory / worker_count if worker_count > 0 else 0
    
    return {
        "total_memory_gb": round(total_memory, 2),
        "available_memory_gb": round(available_memory, 2),
        "usable_memory_gb": round(usable_memory, 2),
        "recommended_memory_per_worker_gb": round(memory_per_worker, 2),
    }


def get_gpu_info():
    """Get GPU information if available."""
    gpu_info = {"available": False}
    
    try:
        # Try to detect NVIDIA GPUs
        if platform.system() == "Windows":
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used", "--format=csv,noheader"],
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_info["available"] = True
                gpu_info["type"] = "NVIDIA"
                
                gpu_data = []
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        gpu_data.append({
                            "name": parts[0],
                            "total_memory": parts[1],
                            "free_memory": parts[2],
                            "used_memory": parts[3]
                        })
                gpu_info["gpus"] = gpu_data
        else:
            # Linux detection
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used", "--format=csv,noheader"],
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_info["available"] = True
                gpu_info["type"] = "NVIDIA"
                
                gpu_data = []
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        gpu_data.append({
                            "name": parts[0],
                            "total_memory": parts[1],
                            "free_memory": parts[2],
                            "used_memory": parts[3]
                        })
                gpu_info["gpus"] = gpu_data
    except Exception as e:
        print(f"Error detecting GPU: {e}")
    
    return gpu_info


def update_docker_compose(cpu_info, memory_info):
    """Update docker-compose.yml with optimized values."""
    compose_file = Path("docker-compose.yml")
    
    if not compose_file.exists():
        print("docker-compose.yml not found in the current directory")
        return False
    
    # Calculate resource limits
    worker_count = cpu_info["recommended_workers"]
    cpus_per_worker = str(cpu_info["recommended_threads_per_worker"])
    memory_per_worker = f"{int(memory_info['recommended_memory_per_worker_gb'])}G"
    
    # Read current docker-compose.yml
    with open(compose_file, "r") as f:
        compose_content = f.read()
    
    # Find the worker section and update the command and resource limits
    updated_content = []
    in_worker_section = False
    found_worker = False
    for line in compose_content.split("\n"):
        if line.strip() == "worker:":
            in_worker_section = True
            found_worker = True
        elif in_worker_section and line.strip().startswith("command:"):
            # Update worker command with optimized concurrency
            line = f"    command: celery -A src.tasks.celery_app worker --loglevel=info --concurrency={worker_count} --prefetch-multiplier=1"
        elif in_worker_section and "deploy:" in line:
            in_worker_section = False
        
        updated_content.append(line)
    
    # Write updated docker-compose.yml
    with open(compose_file, "w") as f:
        f.write("\n".join(updated_content))
    
    print(f"Updated docker-compose.yml with {worker_count} workers")
    return True


def update_celery_app(cpu_info, memory_info, gpu_info):
    """Update src/tasks/celery_app.py with optimized settings."""
    celery_file = Path("src/tasks/celery_app.py")
    
    if not celery_file.exists():
        print("src/tasks/celery_app.py not found")
        return False
    
    # Read current celery_app.py
    with open(celery_file, "r") as f:
        content = f.read()
    
    # Calculate optimal settings
    worker_count = cpu_info["recommended_workers"]
    
    # Check if GPU is available and adjust configuration
    gpu_config = ""
    if gpu_info["available"]:
        gpu_config = f"""
# GPU configuration
celery_app.conf.update(
    worker_proc_enable_prefork=False,  # Disable prefork for GPU support
    task_default_priority=5,  # Medium priority by default
    broker_heartbeat=10,      # More frequent heartbeats
)
"""
    
    # Update worker configuration section
    if "# Configure Celery" in content:
        # Find the configuration section and add our optimizations
        celery_config_start = content.find("# Configure Celery")
        celery_config_end = content.find(")", celery_config_start)
        
        # Update the configuration
        updated_config = f"""# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=86400,  # 24 hours max runtime
    worker_prefetch_multiplier=1,  # Don't prefetch tasks (better for long-running tasks)
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks to prevent memory leaks
    worker_concurrency={worker_count},  # Optimized for this machine
    task_acks_late=True,     # Prevent task loss on worker failure
    task_reject_on_worker_lost=True,  # Requeue tasks if worker dies
    task_default_rate_limit='10/m',  # Prevent overloading
)
{gpu_config}"""
        
        # Replace the configuration section
        updated_content = content[:celery_config_start] + updated_config + content[celery_config_end+1:]
        
        # Write updated file
        with open(celery_file, "w") as f:
            f.write(updated_content)
        
        print(f"Updated src/tasks/celery_app.py with optimized settings for {worker_count} workers")
        return True
    else:
        print("Could not find configuration section in src/tasks/celery_app.py")
        return False


def generate_configs():
    """Generate optimized configuration for Celery and Docker."""
    # Get system information
    print("Analyzing system resources...")
    cpu_info = get_cpu_info()
    memory_info = get_memory_info()
    gpu_info = get_gpu_info()
    
    # Create report
    report = {
        "cpu": cpu_info,
        "memory": memory_info,
        "gpu": gpu_info,
        "recommendations": {
            "worker_count": cpu_info["recommended_workers"],
            "memory_per_worker": f"{int(memory_info['recommended_memory_per_worker_gb'])}G",
            "threads_per_worker": cpu_info["recommended_threads_per_worker"],
            "gpu_enabled": gpu_info["available"]
        }
    }
    
    # Print report
    print("\n--- System Analysis ---")
    print(f"CPU: {cpu_info['cpu_model']} ({cpu_info['cpu_count']} cores)")
    print(f"Memory: {memory_info['total_memory_gb']:.2f} GB total, {memory_info['available_memory_gb']:.2f} GB available")
    
    if gpu_info["available"]:
        for i, gpu in enumerate(gpu_info["gpus"]):
            print(f"GPU {i+1}: {gpu['name']} with {gpu['total_memory']}")
    else:
        print("GPU: Not detected")
    
    print("\n--- Recommended Configuration ---")
    print(f"Celery Workers: {cpu_info['recommended_workers']}")
    print(f"Threads per Worker: {cpu_info['recommended_threads_per_worker']}")
    print(f"Memory per Worker: {int(memory_info['recommended_memory_per_worker_gb'])} GB")
    
    # Save report to file
    with open("celery_optimization_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\nSaved detailed report to celery_optimization_report.json")
    
    # Update configuration files
    print("\nUpdating configuration files...")
    update_docker_compose(cpu_info, memory_info)
    update_celery_app(cpu_info, memory_info, gpu_info)
    
    print("\nConfiguration optimization completed!")


if __name__ == "__main__":
    generate_configs() 