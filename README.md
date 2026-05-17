# LLM Refinery

An automated LLM fine-tuning, quantization, and evaluation platform optimized for a single RTX 3090 (24 GB VRAM). Decoupled architecture: local WSL2 control plane + remote GPU compute plane connected via Tailscale VPN.

##Demo

https://github.com/user-attachments/assets/b42cf485-ec6f-4f44-8479-2a8f03afe4a1


## Architecture

```
┌─────────────────────────────────────┐     Tailscale VPN      ┌──────────────────────────────┐
│      Local Control Plane (WSL2)     │◄──────────────────────►│    Remote GPU Pod (3090)     │
│                                     │   socat port-forward   │                              │
│  React UI        (:5173)            │   (alt ports 1xxxx)    │  Celery Worker (solo pool)   │
│  FastAPI         (:8080)            │                        │    ├── peft_train.py (QLoRA) │
│  MongoDB         (:27017)           │                        │    ├── quantize.py (AWQ)     │
│  Mongo Express   (:8081)            │                        │    ├── evaluate.py (lm-eval) │
│  Redis           (:6379)            │                        │    ├── serve.py (vLLM mgr)   │
│  MinIO           (:9000 / :9001)    │                        │    └── vLLM (:8000)          │
│  MLflow          (:5000)            │                        │                              │
│  Flower          (:5555)            │                        │                              │
└─────────────────────────────────────┘                        └──────────────────────────────┘
```

**Data flow:** Upload `.jsonl` → FastAPI → MinIO + MongoDB → Redis queue → Celery worker (GPU pod) → fine-tune → quantize → evaluate → MLflow metrics → Results chart + model deploy → vLLM chat

## Features

- **QLoRA fine-tuning** — configurable LoRA rank and alpha
- **AWQ quantization** — with optional skip (`none`)
- **Automated evaluation** — quick (custom accuracy) or full (MMLU via lm-eval)
- **Real-time progress tracking** — SSE-powered granular pipeline status (queued → training → quantizing → evaluating → completed)
- **Interactive results** — scatter chart with dynamic axis selection, model comparison table
- **One-click deploy** — serve any trained model via vLLM (single model at a time)
- **Streaming chat** — OpenAI-compatible chat interface with SSE streaming
- **Experiment management** — delete experiments (cleans MLflow + MinIO + MongoDB)

## Quick Start

### Prerequisites

- Docker Desktop with WSL2 integration
- Node.js 18+
- Tailscale installed on both local machine and GPU pod
- RunPod account (or any SSH-accessible GPU instance with an RTX 3090)

### 1. Configure

Set your GPU pod's Tailscale IP in `.env`:

```bash
GPU_POD_IP=100.x.x.x
```

### 2. Start Local Control Plane (WSL2)

```bash
# One command: Docker services + firewall + socat forwarders
bash scripts/start-local.sh

# Start frontend (separate terminal)
cd frontend && npm install && npm run dev
```

### 3. Start Remote GPU Pod (RunPod)

```bash
# SSH into the pod, then:
bash /workspace/start.sh
```

### First-Time Pod Setup

```bash
# 1. Copy compute_node to pod
scp -P <PORT> -i ~/.ssh/id_runpod -r compute_node/ root@<HOST>:/workspace/compute_node/

# 2. Copy config + startup script
scp -P <PORT> -i ~/.ssh/id_runpod compute_node/config.env root@<HOST>:/workspace/config.env
scp -P <PORT> -i ~/.ssh/id_runpod compute_node/start.sh root@<HOST>:/workspace/start.sh

# 3. Edit config on the pod:
#    - /workspace/config.env  → set CONTROL_PLANE_IP to your local Tailscale IP (tailscale ip -4)
#    - /workspace/config.env  → set HF_TOKEN to your HuggingFace token (for gated models)
#    - /workspace/start.sh    → set your Tailscale auth key (from https://login.tailscale.com/admin/settings/keys)

# 4. Run start.sh
bash /workspace/start.sh
```

### Verify

```bash
# Full end-to-end test (pod must be running)
bash tests/test.sh

# Local-only tests (no pod needed)
bash tests/test.sh --local
```

## Services

| Service       | Port  | Purpose                                |
|---------------|-------|----------------------------------------|
| React UI      | 5173  | Upload, results, chat dashboard        |
| FastAPI       | 8080  | API server (`/docs` for Swagger)       |
| MongoDB       | 27017 | Dataset metadata, job & deploy records |
| Mongo Express | 8081  | MongoDB web UI                         |
| Redis         | 6379  | Celery message broker                  |
| MinIO         | 9000  | S3-compatible dataset storage          |
| MinIO Console | 9001  | Storage UI (minioadmin/minioadmin)     |
| MLflow        | 5000  | Experiment tracking UI                 |
| Flower        | 5555  | Celery task monitor                    |

## API Endpoints

| Method | Endpoint                        | Description                          |
|--------|---------------------------------|--------------------------------------|
| POST   | `/api/dataset/upload`           | Upload `.jsonl` dataset              |
| POST   | `/api/experiment/start`         | Queue a fine-tune/quantize job       |
| GET    | `/api/job/{job_id}`             | Get job status                       |
| GET    | `/api/jobs/active`              | List non-terminal jobs               |
| PATCH  | `/api/job/{job_id}/status`      | Update job status (worker callback)  |
| GET    | `/api/experiments/results`      | Fetch all experiment metrics         |
| DELETE | `/api/experiments/{run_id}`     | Delete experiment + artifacts        |
| POST   | `/api/models/deploy`            | Deploy model via vLLM                |
| DELETE | `/api/models/deploy`            | Undeploy current model               |
| GET    | `/api/models/serving-status`    | Current deployment state             |
| PATCH  | `/api/job/{run_id}/deploy-status` | Deploy status callback (worker)    |
| GET    | `/api/events`                   | SSE event stream (job updates)       |
| POST   | `/api/chat/completions`         | Chat proxy to vLLM (streaming SSE)   |

## Networking

The pod can't directly reach Docker containers in WSL2. Socat port forwarders bridge the gap via Tailscale using alternate ports:

| Service | Docker Port | Alt Port | Direction      |
|---------|-------------|----------|----------------|
| FastAPI | 8080        | 18080    | Pod → Local    |
| MLflow  | 5000        | 15000    | Pod → Local    |
| Redis   | 6379        | 16379    | Pod → Local    |
| MinIO   | 9000        | 19000    | Pod → Local    |
| vLLM    | 8000        | 18000    | Local → Pod    |

- **Local side** (`scripts/tailscale-forward.sh`): socat on Tailscale IP alt ports → localhost Docker ports, plus reverse tunnel for vLLM
- **Pod side** (`start.sh`): socat on localhost alt ports → `tailscale nc` to control plane

## Pipeline

```
Upload .jsonl → Queue Job
                    ↓
            Fine-tune (QLoRA)
                    ↓
            Quantize (AWQ/None)                                      
                    ↓
                Evaluate
                    ↓
              Log to MLflow → Select Model → Deploy (vLLM) → Chat Interface

```

## Project Structure

```
llm-refinery/
├── .env                        # Single source of truth (GPU_POD_IP, service URLs)
├── docker-compose.yml          # MongoDB, Redis, MinIO, MLflow, Flower, FastAPI backend
├── backend/
│   ├── main.py                 # FastAPI app — all API endpoints + SSE + chat proxy
│   ├── Dockerfile
│   ├── requirements.txt
│   └── services/
│       ├── minio_client.py     # S3 upload/download/delete
│       ├── redis_client.py     # Celery job dispatch
│       └── mlflow_client.py    # MLflow results retrieval
├── frontend/
│   └── src/
│       ├── App.jsx             # Tab layout (Upload, Results, Chat, How to Use)
│       ├── api/client.js       # API client (axios + fetch streaming)
│       └── components/
│           ├── UploadForm.jsx  # Dataset upload + experiment config + progress tracker
│           ├── ParetoChart.jsx # Scatter chart, model list, compare table, deploy/delete
│           ├── ChatInterface.jsx # Streaming chat with deployed model
│           └── HowToUse.jsx    # Onboarding guide
├── compute_node/
│   ├── worker.py               # Celery tasks (pipeline, deploy, undeploy)
│   ├── peft_train.py           # QLoRA fine-tuning
│   ├── quantize.py             # AWQ quantization
│   ├── evaluate.py             # Accuracy + latency + VRAM + optional MMLU
│   ├── serve.py                # vLLM process manager (start/stop/health)
│   ├── cuda_setup.py           # CUDA library pre-loader
│   ├── config.env              # Control plane Tailscale IP
│   ├── start.sh                # Pod bootstrap (Tailscale + socat + Celery)
│   └── requirements.txt
├── scripts/
│   ├── start-local.sh          # Local startup (Docker + firewall + socat)
│   └── tailscale-forward.sh    # Port forwarding + vLLM reverse tunnel
├── tests/
│   └── test.sh                 # E2E test suite
└── dataset/
    └── medical_test.jsonl      # Sample dataset
```

## What Survives Pod Restarts

| Survives                                     | Doesn't Survive                                   |
|----------------------------------------------|----------------------------------------------------|
| `/workspace/` (code, models, HF cache)       | Running processes (Tailscale, socat, Celery, vLLM) |
| Tailscale binary                             | apt packages (socat, redis-tools)                  |
| Python packages (pip)                        | iptables rules                                     |

Run `bash /workspace/start.sh` after each pod restart — it reinstalls system deps and starts everything.
