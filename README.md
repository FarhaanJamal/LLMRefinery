# LLM Refinery

An automated LLM fine-tuning, quantization, and evaluation platform optimized for a single 24GB VRAM RTX 3090. Decoupled architecture: local WSL2 control plane + remote GPU compute plane connected via Tailscale VPN.

## Architecture

```
┌──────────────────────────────────┐       Tailscale VPN       ┌──────────────────────────┐
│     Local Control Plane (WSL2)   │◄─────────────────────────►│   Remote GPU Pod (3090)  │
│                                  │                           │                          │
│  React UI (:5173)                │    socat port forwarding   │  Celery Worker           │
│  FastAPI  (:8080)                │    (alt ports 1xxxx)      │    ├── peft_train.py      │
│  MongoDB  (:27017)               │                           │    ├── quantize.py        │
│  Redis    (:6379)                │                           │    ├── evaluate.py        │
│  MinIO    (:9000/:9001)          │                           │    └── vLLM serve (:8000) │
│  MLflow   (:5000)                │                           │                          │
│  Flower   (:5555)                │                           │                          │
└──────────────────────────────────┘                           └──────────────────────────┘
```

**Data flow:** Upload `.jsonl` → FastAPI → MinIO + MongoDB → Redis queue → Celery worker (GPU pod) → MLflow metrics → Pareto chart

## Quick Start

### Prerequisites

- Docker Desktop with WSL2 integration
- Node.js 18+
- Tailscale installed on both local machine and GPU pod
- RunPod account (or any SSH-accessible GPU instance)

### Starting the Platform

#### 1. Local Control Plane (WSL2)

update the pod IP in `.env` first, then:

```bash
# One command does everything: Docker services, firewall, socat forwarders
bash scripts/start-local.sh

# Start frontend (separate terminal)
cd frontend && npm install && npm run dev
```

#### 2. Remote GPU Pod (RunPod)

```bash
# SSH into the pod
ssh root@<POD_IP> -p <PORT> -i ~/.ssh/id_runpod

# One command: installs deps, starts Tailscale, socat tunnels, Celery worker
bash /workspace/start.sh
```

### First-Time Pod Setup

If setting up a fresh pod:

```bash
# 1. Copy compute_node from local
scp -P <PORT> -i ~/.ssh/id_runpod -r /mnt/c/WSL/llm-refinery/compute_node/ root@<HOST>:/workspace/compute_node/

# 2. Copy config files
scp -P <PORT> -i ~/.ssh/id_runpod /mnt/c/WSL/llm-refinery/compute_node/config.env root@<HOST>:/workspace/config.env
scp -P <PORT> -i ~/.ssh/id_runpod /mnt/c/WSL/llm-refinery/compute_node/start.sh root@<HOST>:/workspace/start.sh

# 3. Update auth key in /workspace/start.sh (get from https://login.tailscale.com/admin/settings/keys)

# 4. Run start.sh
bash /workspace/start.sh
```

### Verify

```bash
# Full end-to-end tests (pod must be running)
bash tests/test.sh

# Local-only tests (no pod needed)
bash tests/test.sh --local
```

## Services

| Service  | Port  | Purpose                          |
|----------|-------|----------------------------------|
| FastAPI  | 8080  | API server (`/docs` for Swagger) |
| React UI | 5173  | Upload, results, chat dashboard  |
| MongoDB  | 27017 | Dataset metadata, job records    |
| Redis    | 6379  | Celery message broker            |
| MinIO    | 9000  | S3-compatible dataset storage    |
| MinIO UI | 9001  | Storage console (minioadmin/minioadmin) |
| MLflow   | 5000  | Experiment tracking UI           |
| Flower   | 5555  | Celery task monitor              |

## API Endpoints

| Method | Endpoint                   | Description                        |
|--------|----------------------------|------------------------------------|
| POST   | `/api/dataset/upload`      | Upload `.jsonl` dataset            |
| POST   | `/api/experiment/start`    | Queue a fine-tune/quantize job     |
| PATCH  | `/api/job/{job_id}/status` | Update job status (worker callback)|
| GET    | `/api/experiments/results` | Fetch all experiment metrics       |

## Networking

The pod can't directly reach Docker containers on the local machine. Socat port forwarders bridge the gap using alternate ports:

| Service | Local Port | Alt Port | Direction |
|---------|-----------|----------|-----------|
| FastAPI | 8080      | 18080    | Pod → Local |
| MLflow  | 5000      | 15000    | Pod → Local |
| Redis   | 6379      | 16379    | Pod → Local |
| MinIO   | 9000      | 19000    | Pod → Local |

**Local side** (`scripts/tailscale-forward.sh`): socat listens on Tailscale IP alt ports → forwards to localhost standard ports.

**Pod side** (`start.sh`): socat listens on localhost alt ports → `tailscale nc` to control plane Tailscale IP alt ports.

## Pipeline (per job)

```
Upload .jsonl → Queue Job → Fine-tune (QLoRA) → Quantize (AWQ/None) → Evaluate → Log to MLflow
```

## Project Structure

```
llm-refinery/
├── docker-compose.yml
├── .env
├── backend/
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── services/
│       ├── minio_client.py
│       ├── redis_client.py
│       └── mlflow_client.py
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── api/client.js
│       └── components/
│           ├── UploadForm.jsx
│           ├── ParetoChart.jsx
│           ├── ChatInterface.jsx
│           └── HowToUse.jsx
├── compute_node/
│   ├── worker.py
│   ├── peft_train.py
│   ├── quantize.py
│   ├── evaluate.py
│   ├── config.env
│   ├── start.sh
│   └── requirements.txt
├── scripts/
│   ├── start-local.sh
│   └── tailscale-forward.sh
└── tests/
    └── test.sh
```

## What Survives Pod Restarts

| Survives | Doesn't Survive |
|----------|----------------|
| `/workspace/` (code, models, HF cache) | Running processes (Tailscale, socat, Celery) |
| Tailscale binary | apt packages (socat, redis-tools) |
| Python packages (pip) | iptables rules |

Just run `bash /workspace/start.sh` after each pod restart — it handles everything.
