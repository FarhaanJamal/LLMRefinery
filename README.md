# LLM Refinery

An automated LLM fine-tuning, quantization, and evaluation platform optimized for a single 24GB VRAM RTX 3090.

## Architecture

```
┌──────────────────────────────────┐       Tailscale VPN       ┌──────────────────────────┐
│     Local Control Plane (WSL2)   │◄─────────────────────────►│   Remote GPU Pod (3090)  │
│                                  │                           │                          │
│  React UI (:5173)                │                           │  Celery Worker           │
│  FastAPI  (:8080)                │                           │    ├── peft_train.py      │
│  MongoDB  (:27017)               │                           │    ├── quantize.py        │
│  Redis    (:6379)                │                           │    ├── evaluate.py        │
│  MinIO    (:9000/:9001)          │                           │    └── vLLM serve (:8000) │
│  MLflow   (:5000)                │                           │                          │
└──────────────────────────────────┘                           └──────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker Desktop with WSL2 integration
- Node.js 18+

### 1. Start Infrastructure

```bash
docker compose up -d --build
```

This starts MongoDB, Redis, MinIO, MLflow, and the FastAPI backend.

### 2. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173`

### 3. Verify

```bash
bash tests/test_phase1.sh
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

## API Endpoints

| Method | Endpoint                   | Description                        |
|--------|----------------------------|------------------------------------|
| POST   | `/api/dataset/upload`      | Upload `.jsonl` dataset            |
| POST   | `/api/experiment/start`    | Queue a fine-tune/quantize job     |
| GET    | `/api/experiments/results` | Fetch all experiment metrics       |

## Pipeline (per job)

```
Upload .jsonl → Queue Job → Fine-tune (QLoRA) → Quantize (AWQ/GPTQ/None) → Evaluate → Log to MLflow
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
│   └── requirements.txt
└── tests/
    └── test_phase1.sh
```
