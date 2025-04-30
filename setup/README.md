# Containerized Setup Guide

This guide provides step-by-step instructions for setting up the containerized distributed processing system for quantitative trading.

## Prerequisites

1. **Docker Desktop**
   - Install Docker Desktop for Windows: https://www.docker.com/products/docker-desktop
   - Ensure WSL2 backend is enabled for better performance
   - Allocate sufficient resources (recommended: 8GB RAM, 4 CPUs)

2. **Data Files**
   - Ensure you have the necessary data files in the `data` directory
   - For actual trading, you'll need the 15 years of futures data mentioned in the requirements

## Setting Up the Environment

### 1. Clone the Repository

```powershell
git clone https://github.com/yourusername/quant-trader.git
cd quant-trader
```

### 2. Configure Docker Resources

Ensure Docker Desktop has enough resources:
- Open Docker Desktop > Settings > Resources
- Allocate at least 8GB RAM and 4 CPUs
- Ensure disk image size is at least 60GB

### 3. Start the Containers

Run the provided PowerShell script to start all containers:

```powershell
.\scripts\start-containers.ps1
```

This will:
- Check that Docker is running
- Build and start all containers (API, Redis, Worker, Flower)
- Display URLs for accessing the services

### 4. Verify the Setup

1. **Check API Health**
   - Open a browser and navigate to: http://localhost:8000/health
   - You should see: `{"status": "healthy"}`

2. **Check Flower Dashboard**
   - Navigate to: http://localhost:5555
   - Verify that worker nodes are registered and active

## Using the System

### Submitting Tasks

You can submit tasks using the PowerShell script:

```powershell
# Submit a training task
.\scripts\submit-task.ps1 -TaskType train-models -Pair GC_SI -Timeframe 1hour -StartDate 2022-01-01 -EndDate 2023-01-01

# Submit a backtest task
.\scripts\submit-task.ps1 -TaskType backtest -Pairs GC_SI,ZN_ZB -StartDate 2022-01-01 -EndDate 2023-01-01 -Timeframe 1hour -UseML

# Submit a parameter optimization task
.\scripts\submit-task.ps1 -TaskType optimize-parameters -PairsFile output/pairs_list.json -StartDate 2022-01-01 -EndDate 2023-01-01 -QuickMode

# Wait for the task to complete
.\scripts\submit-task.ps1 -TaskType train-models -Pair GC_SI -Timeframe 1hour -StartDate 2022-01-01 -EndDate 2023-01-01 -WaitForCompletion
```

### Monitoring Tasks

1. **Using Flower Dashboard**
   - Navigate to http://localhost:5555
   - View active, pending, and completed tasks
   - Check worker status and resource usage

2. **Using API**
   - Get task status: `http://localhost:8000/tasks/{task_id}`
   - Health check: `http://localhost:8000/health`

### Stopping the Containers

When you're done, stop the containers:

```powershell
.\scripts\stop-containers.ps1
```

## Docker Container Details

### 1. API Container

- Runs the FastAPI application
- Provides endpoints for task submission and monitoring
- Exposes port 8000

### 2. Redis Container

- Acts as message broker for Celery tasks
- Stores task results
- Exposes port 6379

### 3. Worker Container

- Executes long-running tasks
- Processes model training, backtesting, and optimization jobs
- Can be scaled up by modifying docker-compose.yml

### 4. Flower Container

- Provides a web dashboard for monitoring tasks
- Shows worker status and task history
- Exposes port 5555

## Data Persistence

Data is persisted through Docker volumes:

- **redis-data**: Persists Redis data across container restarts
- **Mounted volumes**: Data, models, output, and logs are mounted from the host

## Customization

### Scaling Workers

To add more worker nodes, modify docker-compose.yml:

```yaml
worker:
  # ... existing configuration ...
  deploy:
    replicas: 3  # Number of worker instances
```

Then run:

```powershell
docker-compose up -d --scale worker=3
```

### Configuring Resources

Adjust container resources in docker-compose.yml:

```yaml
worker:
  # ... existing configuration ...
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 4G
```

## Troubleshooting

### Container Startup Issues

If containers fail to start:

1. Check Docker logs:
   ```powershell
   docker-compose logs
   ```

2. Ensure all ports are available:
   - 8000 (API)
   - 6379 (Redis)
   - 5555 (Flower)

3. Check Docker Desktop is running correctly:
   ```powershell
   docker info
   ```

### Task Execution Issues

If tasks fail to execute:

1. Check worker logs:
   ```powershell
   docker-compose logs worker
   ```

2. Verify Redis connection:
   ```powershell
   docker-compose exec redis redis-cli ping
   ```

3. Check disk space and memory:
   ```powershell
   docker system df
   ``` 