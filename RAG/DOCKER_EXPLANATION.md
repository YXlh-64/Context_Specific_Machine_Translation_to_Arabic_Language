# Docker Usage in NLP Translation Pipeline - Complete Explanation

## 🎯 Why Docker is Used Here

Docker is used in this project to run **external services** that your Python applications need, specifically:

1. **Qdrant** - Vector database for semantic search (REQUIRED)
2. **Redis** - Caching service (OPTIONAL but recommended)

### Why Not Install These Directly?

You could install Qdrant and Redis directly on your system, but Docker provides several advantages:

#### ✅ **Advantages of Using Docker:**

1. **Easy Setup**: No complex installation steps, just one command
2. **Isolation**: Services run in containers, don't pollute your system
3. **Consistency**: Same environment across different machines (Linux, Mac, Windows)
4. **Portability**: Easy to move between development and production
5. **Version Management**: Easy to use specific versions
6. **Clean Removal**: Delete container = completely remove the service
7. **No Conflicts**: Won't interfere with other software on your system

#### ❌ **Without Docker (Traditional Installation):**

- Need to download and compile Qdrant from source
- Need to configure system services
- Different installation steps for each OS
- Harder to uninstall cleanly
- Potential conflicts with other software

---

## 🏗️ How Docker Works in This Project

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Your Computer                         │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │         Python Applications (Your Code)               │  │
│  │  ┌──────────────┐  ┌──────────────┐              │  │
│  │  │  Glossary    │  │   RAG        │              │  │
│  │  │  System      │  │   System     │              │  │
│  │  │  Port 8001   │  │  Port 8002   │              │  │
│  │  └──────┬───────┘  └──────┬───────┘              │  │
│  │         │                 │                       │  │
│  └─────────┼─────────────────┼───────────────────────┘  │
│            │                 │                           │
│            ▼                 ▼                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │         Docker Containers (External Services)     │  │
│  │                                                     │  │
│  │  ┌──────────────┐         ┌──────────────┐       │  │
│  │  │   Redis      │         │   Qdrant     │       │  │
│  │  │   Port 6379  │         │   Port 6333   │       │  │
│  │  │  (Optional)  │         │  (Required)  │       │  │
│  │  └──────────────┘         └──────────────┘       │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### Key Concept: Docker Containers vs Your Python Apps

**Important**: Your Python applications (Glossary System, RAG System, Prompt Construction) are **NOT** running in Docker. They run directly on your system.

**Docker is only used for**:
- Qdrant (vector database)
- Redis (caching)

Your Python apps connect to these Docker containers via **network ports** (localhost:6333 for Qdrant, localhost:6379 for Redis).

---

## 🔧 How Docker is Used - Step by Step

### Step 1: Starting Qdrant with Docker

```bash
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant:latest
```

**Breaking down this command:**

- `docker run` - Start a new container
- `-d` - Run in "detached" mode (background)
- `-p 6333:6333` - Map port 6333 from container to port 6333 on your machine
- `-p 6334:6334` - Map port 6334 (Qdrant dashboard)
- `--name qdrant` - Give the container a name for easy reference
- `qdrant/qdrant:latest` - The Docker image to use (official Qdrant image)

**What happens:**
1. Docker downloads the Qdrant image (if not already downloaded)
2. Creates a container from that image
3. Starts Qdrant inside the container
4. Makes it accessible on `localhost:6333`

### Step 2: Your Python App Connects to Qdrant

When your RAG System starts, it connects to Qdrant like this:

```python
# In RAG-SYSTEM/app/core/config.py
QDRANT_HOST: str = "localhost"  # The Docker container
QDRANT_PORT: int = 6333         # The exposed port

# In RAG-SYSTEM/app/services/setup_qdrant.py
client = QdrantClient(host="localhost", port=6333)
```

**Connection Flow:**
```
Python App (localhost:8002)
    │
    │ HTTP Request
    ▼
Docker Container (localhost:6333)
    │
    ▼
Qdrant Service (inside container)
```

### Step 3: Starting Redis with Docker (Optional)

```bash
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

**Breaking down this command:**

- `redis:7-alpine` - Lightweight Redis image
- `-p 6379:6379` - Map Redis port
- `--name redis` - Container name

Your Python apps connect to Redis the same way:
```python
# In glossary-system/app/core/config.py
REDIS_HOST: str = "localhost"
REDIS_PORT: int = 6379
```

---

## 📊 Docker Container Lifecycle

### Starting Containers

```bash
# Start Qdrant
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant:latest

# Start Redis
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

### Checking if Containers are Running

```bash
# List all running containers
docker ps

# Output example:
# CONTAINER ID   IMAGE                  STATUS         PORTS
# abc123def456   qdrant/qdrant:latest   Up 2 hours     0.0.0.0:6333->6333/tcp
# def456ghi789   redis:7-alpine         Up 2 hours     0.0.0.0:6379->6379/tcp
```

### Stopping Containers

```bash
# Stop a container (keeps it for later)
docker stop qdrant
docker stop redis

# Start stopped containers
docker start qdrant
docker start redis
```

### Removing Containers

```bash
# Stop and remove container
docker stop qdrant
docker rm qdrant

# Or force remove (stops first)
docker rm -f qdrant
```

### Viewing Container Logs

```bash
# See what's happening inside the container
docker logs qdrant
docker logs redis

# Follow logs in real-time
docker logs -f qdrant
```

---

## 🔌 How Your Python Apps Connect

### Connection Configuration

Your Python applications connect to Docker containers using **standard network connections**:

#### 1. Qdrant Connection (RAG System)

**Configuration File**: `RAG-SYSTEM/app/core/config.py`

```python
class Settings:
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")  # ← Docker container
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))   # ← Exposed port
```

**Connection Code**: `RAG-SYSTEM/app/services/setup_qdrant.py`

```python
def get_qdrant_client() -> QdrantClient:
    client = QdrantClient(
        host=settings.QDRANT_HOST,  # "localhost"
        port=settings.QDRANT_PORT   # 6333
    )
    return client
```

**What happens:**
- Python app makes HTTP request to `http://localhost:6333`
- Docker forwards this to the Qdrant service inside the container
- Qdrant processes the request and returns response
- Response goes back through Docker to your Python app

#### 2. Redis Connection (All Services)

**Configuration**: `glossary-system/app/core/config.py`

```python
class Settings:
    REDIS_HOST: str = "localhost"  # ← Docker container
    REDIS_PORT: int = 6379         # ← Exposed port
```

**Connection**: Uses `redis` Python library
```python
import redis
r = redis.Redis(host='localhost', port=6379, db=0)
```

---

## 🐳 Docker Commands Reference

### Essential Commands

```bash
# Check if Docker is installed
docker --version

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Start a container
docker start <container_name>

# Stop a container
docker stop <container_name>

# Remove a container
docker rm <container_name>

# View container logs
docker logs <container_name>

# Execute command inside container
docker exec -it <container_name> <command>

# Check container resource usage
docker stats
```

### For This Project

```bash
# Start Qdrant
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant:latest

# Start Redis
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Check if they're running
docker ps | grep -E "qdrant|redis"

# Stop both
docker stop qdrant redis

# Start both
docker start qdrant redis

# Remove both (careful - deletes data!)
docker rm -f qdrant redis
```

---

## 💾 Data Persistence

### Important: Container Data

**By default, data in Docker containers is temporary!**

When you remove a container, all data inside is lost. For production, you should use **volumes**:

```bash
# Create a volume for Qdrant data
docker volume create qdrant_data

# Run Qdrant with volume
docker run -d \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_data:/qdrant/storage \
  --name qdrant \
  qdrant/qdrant:latest
```

**What this does:**
- `-v qdrant_data:/qdrant/storage` - Maps a Docker volume to the storage directory
- Data persists even if you remove the container
- Volume can be reused by new containers

### For Development

For development, you can use the simple command (data is temporary):
```bash
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant:latest
```

---

## 🔍 Troubleshooting Docker

### Problem: "Cannot connect to Docker daemon"

**Solution**: Docker service is not running
```bash
# Start Docker service (Linux)
sudo systemctl start docker

# Or check Docker Desktop is running (Mac/Windows)
```

### Problem: "Port already in use"

**Solution**: Another container or service is using the port
```bash
# Find what's using the port
lsof -i :6333
lsof -i :6379

# Stop the conflicting container
docker stop <container_name>

# Or use a different port
docker run -d -p 6334:6333 --name qdrant qdrant/qdrant:latest
```

### Problem: "Container name already exists"

**Solution**: Remove the old container first
```bash
# Remove existing container
docker rm -f qdrant

# Then create new one
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant:latest
```

### Problem: Container keeps stopping

**Solution**: Check logs to see why
```bash
docker logs qdrant
docker logs redis
```

### Verify Connection

```bash
# Test Qdrant
curl http://localhost:6333/health

# Test Redis
redis-cli ping
# Or if Redis is in Docker:
docker exec redis redis-cli ping
```

---

## 🆚 Docker vs Direct Installation

### Using Docker (Current Approach)

**Pros:**
- ✅ One command to start
- ✅ No system configuration needed
- ✅ Easy to remove completely
- ✅ Works the same on all OS
- ✅ Isolated from system

**Cons:**
- ❌ Requires Docker installed
- ❌ Uses some system resources
- ❌ Need to learn Docker basics

### Direct Installation (Alternative)

**Pros:**
- ✅ No Docker needed
- ✅ Slightly faster (no container overhead)
- ✅ More control over configuration

**Cons:**
- ❌ Complex installation process
- ❌ Different steps for each OS
- ❌ Harder to uninstall
- ❌ Can conflict with other software

**For this project, Docker is recommended** because:
1. Qdrant installation is complex
2. You might need different versions for different projects
3. Easy cleanup when done with project

---

## 📚 Understanding Docker Concepts

### Image vs Container

- **Image**: A template/blueprint (like a class in programming)
  - Example: `qdrant/qdrant:latest` is an image
  
- **Container**: A running instance of an image (like an object)
  - Example: `qdrant` container is running instance of `qdrant/qdrant:latest`

**Analogy**: 
- Image = Recipe
- Container = Actual cake made from recipe

### Port Mapping

`-p 6333:6333` means:
- **Left side (6333)**: Port on your computer
- **Right side (6333)**: Port inside the container
- **Mapping**: Requests to `localhost:6333` → forwarded to container's port 6333

### Detached Mode (`-d`)

- Without `-d`: Container runs in foreground (blocks terminal)
- With `-d`: Container runs in background (you get terminal back)

---

## 🎓 Summary

**Why Docker?**
- Makes it easy to run Qdrant and Redis without complex installation
- Isolates services from your system
- Consistent across different operating systems

**How it works?**
1. Docker runs Qdrant/Redis in containers
2. Containers expose ports (6333, 6379)
3. Your Python apps connect to `localhost:6333` and `localhost:6379`
4. Docker forwards requests to services inside containers

**Key Commands:**
```bash
# Start services
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant:latest
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Check status
docker ps

# Stop services
docker stop qdrant redis

# Start again
docker start qdrant redis
```

**Your Python apps don't know Docker exists** - they just connect to `localhost:6333` like any other service!

---

## 🔗 Additional Resources

- **Docker Official Docs**: https://docs.docker.com/
- **Qdrant Docker Hub**: https://hub.docker.com/r/qdrant/qdrant
- **Redis Docker Hub**: https://hub.docker.com/_/redis
- **Docker Tutorial**: https://docs.docker.com/get-started/

---

**Remember**: Docker is just a convenient way to run Qdrant and Redis. Your Python applications run normally on your system and connect to these services via network ports, just like they would connect to any other service running on your computer!
