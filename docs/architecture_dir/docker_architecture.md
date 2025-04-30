# Docker Architecture for Distributed Processing

## Overview

This document provides a detailed explanation of our Docker-based distributed processing architecture. This architecture uses containers to isolate and manage different components of our trading system, enabling better scalability, reliability, and maintainability.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Docker Compose Environment                           │
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐│
│  │             │     │             │     │             │     │             ││
│  │  FastAPI    │────►│   Redis     │◄────│   Celery    │     │   Flower    ││
│  │  Server     │     │   Queue     │     │   Workers   │────►│  Dashboard  ││
│  │             │     │             │     │             │     │             ││
│  └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘│
│         ▲                                       │                           │
│         │                                       │                           │
└─────────┼───────────────────────────────────────┼───────────────────────────┘
          │                                       │
          │                                       │
┌─────────┼───────────────────────────────────────┼───────────────────────────┐
│         │                                       │                           │
│         │                                       ▼                           │
│  ┌─────────────┐                      ┌─────────────────────┐              │
│  │  Client     │                      │ Shared Volume        │              │
│  │ Application │                      │ - Data               │              │
│  │ (External)  │                      │ - Models             │              │
│  └─────────────┘                      │ - Results            │              │
│                                       └─────────────────────┘              │
│                           Host Machine                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. FastAPI Server Container
- **Purpose**: Provides RESTful API endpoints for task submission and monitoring
- **Key Files**: 
  - `src/api/main.py`: API implementation
  - `src/api/__init__.py`: Package initialization
- **Ports**: Exposes port 8000 to the host
- **Dependencies**: Connects to Redis container

### 2. Redis Queue Container
- **Purpose**: Message broker for task queue
- **Configuration**: Using Redis 6.2 Alpine image
- **Ports**: Internal port 6379, not exposed outside Docker network
- **Persistence**: Redis data is ephemeral (not persisted)

### 3. Celery Worker Containers
- **Purpose**: Processes background tasks like model training, backtesting, and optimization
- **Key Files**:
  - `src/tasks/celery_app.py`: Celery configuration
  - `src/tasks/train_tasks.py`: Model training tasks
  - `src/tasks/backtest_tasks.py`: Backtesting tasks
  - `src/tasks/optimization_tasks.py`: Optimization tasks
- **Scaling**: Multiple workers can be deployed to handle increased workload
- **Dependencies**: Connects to Redis container

### 4. Flower Dashboard Container
- **Purpose**: Web UI for monitoring Celery tasks
- **Ports**: Exposes port 5555 to the host
- **Dependencies**: Connects to Redis container

### 5. Shared Volume
- **Purpose**: Provides persistent storage accessible by all containers
- **Content**:
  - Data files
  - Trained models
  - Backtest results
  - Configuration files

## Container Communication

1. **API-to-Queue**: 
   - The FastAPI server submits tasks to Redis
   - Communication uses the Celery protocol

2. **Queue-to-Workers**:
   - Workers poll Redis for new tasks
   - Tasks are distributed among available workers

3. **Worker-to-Shared Volume**:
   - Workers read data from the shared volume
   - Results are written back to the shared volume

4. **Flower-to-Queue**:
   - Flower dashboard connects to Redis to monitor tasks
   - Provides real-time task status and history

## Configuration Files

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    volumes:
      - .:/app
    ports:
      - "8000:8000"
    depends_on:
      - redis

  worker:
    build:
      context: .
      dockerfile: Dockerfile
    command: celery -A src.tasks.celery_app worker --loglevel=info
    volumes:
      - .:/app
    depends_on:
      - redis

  redis:
    image: redis:6.2-alpine
    ports:
      - "6379:6379"

  flower:
    build:
      context: .
      dockerfile: Dockerfile
    command: celery -A src.tasks.celery_app flower --port=5555
    volumes:
      - .:/app
    ports:
      - "5555:5555"
    depends_on:
      - redis
```

## Task Flow

1. **Task Submission**:
   - Client submits a task via API request to FastAPI server
   - Example: POST to `/api/v1/tasks/backtest` with backtest parameters

2. **Task Queuing**:
   - API server validates the request
   - If valid, it enqueues the task in Redis via Celery

3. **Task Processing**:
   - A worker picks up the task from Redis
   - Worker executes the task (e.g., runs a backtest)
   - Results are stored in the shared volume

4. **Task Monitoring**:
   - Flower dashboard shows real-time task status
   - API endpoints provide programmatic access to task status

## Advantages of This Architecture

1. **Scalability**:
   - Workers can be scaled independently as needed
   - Multiple workers can process tasks in parallel

2. **Resource Isolation**:
   - Each component operates in its own container
   - Resource constraints can be set per container

3. **Deployment Flexibility**:
   - Can be deployed on a single machine or across multiple hosts
   - Works consistently across different environments

4. **Resilience**:
   - Components can be restarted independently
   - System can continue functioning if one component fails

## Development Workflow

1. **Local Development**:
   - Modify code locally
   - Run `docker-compose up` to start the system
   - Changes are immediately available via volume mounting

2. **Testing**:
   - Use the API to submit test tasks
   - Monitor execution via Flower dashboard
   - Examine logs using `docker-compose logs <service>`

3. **Deployment**:
   - Use the same docker-compose file for production
   - Configure environment variables for production settings
   - Scale workers as needed with `docker-compose up --scale worker=3`

## Performance Considerations

1. **CPU Allocation**:
   - Worker containers can be limited to specific CPU cores/shares
   - Prevents resource contention between containers

2. **Memory Management**:
   - Memory limits should be set for each container
   - Prevents memory-intensive tasks from affecting other containers

3. **Storage Performance**:
   - I/O-intensive operations should be optimized
   - Consider using volumes with high performance for data-intensive tasks

## Next Steps

1. **Monitoring Enhancement**:
   - Add Prometheus/Grafana for detailed system monitoring
   - Implement custom health checks for each service

2. **Load Balancing**:
   - Add Nginx or Traefik as a reverse proxy for API scaling
   - Implement rate limiting for API requests

3. **Security Hardening**:
   - Implement network security policies between containers
   - Add authentication for API and monitoring interfaces

4. **Backup Strategy**:
   - Implement regular backups of shared volumes
   - Create disaster recovery procedures 