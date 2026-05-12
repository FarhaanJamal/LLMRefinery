import json
import uuid
from datetime import datetime, timezone
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymongo import MongoClient
import httpx
import os

from services.minio_client import upload_fileobj
from services.redis_client import send_job
from services.mlflow_client import get_all_results

# --------------- App Setup ---------------

app = FastAPI(title="LLM Refinery", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URL = os.getenv("MONGO_URL", "mongodb://mongodb:27017/llm_refinery")
mongo_client = MongoClient(MONGO_URL)
db = mongo_client.get_default_database()

# --------------- Schemas ---------------


class ExperimentParams(BaseModel):
    r: int = 16
    alpha: int = 32
    quant_type: str = "awq"  # "awq" or "none"
    eval_mode: str = "quick"  # "quick" or "full"


class ExperimentRequest(BaseModel):
    model: str
    task: str = "qlora"
    params: ExperimentParams
    dataset_path: str


class JobStatusUpdate(BaseModel):
    status: str


class DeployRequest(BaseModel):
    run_id: str


class DeployStatusUpdate(BaseModel):
    status: str


GPU_POD_IP = os.getenv("GPU_POD_IP", "100.125.222.87")
VLLM_PORT = os.getenv("VLLM_PORT", "8000")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", f"http://{GPU_POD_IP}:{VLLM_PORT}")


# --------------- Endpoints ---------------


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    # Validate file extension
    if not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Only .jsonl files are accepted.")

    # Read and validate each line is valid JSON
    contents = await file.read()
    lines = contents.decode("utf-8").strip().split("\n")

    if not lines or lines == [""]:
        raise HTTPException(status_code=400, detail="File is empty.")

    row_count = 0
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail=f"Invalid JSON on line {i + 1}.")
        if not isinstance(obj, dict):
            raise HTTPException(status_code=400, detail=f"Line {i + 1} is not a JSON object.")
        row_count += 1

    # Upload to MinIO
    dataset_id = str(uuid.uuid4())
    object_name = f"{dataset_id}.jsonl"
    data = BytesIO(contents)
    s3_path = upload_fileobj(data, len(contents), object_name, content_type="application/jsonl")

    # Save metadata to MongoDB
    db.datasets.insert_one({
        "dataset_id": dataset_id,
        "filename": file.filename,
        "row_count": row_count,
        "s3_path": s3_path,
        "uploaded_at": datetime.now(timezone.utc),
    })

    return {"dataset_id": dataset_id, "s3_path": s3_path, "row_count": row_count}


@app.post("/api/experiment/start")
async def start_experiment(request: ExperimentRequest):
    # Validate quant_type
    if request.params.quant_type not in ("awq", "none"):
        raise HTTPException(status_code=400, detail="quant_type must be 'awq' or 'none'.")
    if request.params.eval_mode not in ("quick", "full"):
        raise HTTPException(status_code=400, detail="eval_mode must be 'quick' or 'full'.")

    job_id = str(uuid.uuid4())

    payload = {
        "job_id": job_id,
        "model": request.model,
        "task": request.task,
        "params": {
            "r": request.params.r,
            "alpha": request.params.alpha,
            "quant_type": request.params.quant_type,
            "eval_mode": request.params.eval_mode,
        },
        "dataset_path": request.dataset_path,
    }

    # Queue the job via Celery
    send_job("compute.run_pipeline", payload)

    # Save job record to MongoDB
    db.jobs.insert_one({
        "job_id": job_id,
        "model": request.model,
        "task": request.task,
        "params": payload["params"],
        "dataset_path": request.dataset_path,
        "status": "queued",
        "created_at": datetime.now(timezone.utc),
    })

    return {"job_id": job_id, "status": "queued"}


@app.patch("/api/job/{job_id}/status")
async def update_job_status(job_id: str, body: JobStatusUpdate):
    if body.status not in ("running", "completed", "failed"):
        raise HTTPException(status_code=400, detail="Invalid status.")
    result = db.jobs.update_one(
        {"job_id": job_id},
        {"$set": {"status": body.status}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"job_id": job_id, "status": body.status}


@app.get("/api/experiments/results")
async def get_experiment_results():
    results = get_all_results()
    return {"experiments": results}


# --------------- Deployment Endpoints ---------------


@app.post("/api/models/deploy")
async def deploy_model(body: DeployRequest):
    """Deploy a trained model via vLLM on the GPU pod."""
    # Look up model info from MLflow
    results = get_all_results()
    run = next((r for r in results if r["run_id"] == body.run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found in MLflow.")
    if not run.get("model_artifact_path"):
        raise HTTPException(status_code=400, detail="No model artifact path for this run.")

    # Only one model at a time — stop any existing deployment
    existing = db.deployments.find_one({"status": "running"})
    if existing and existing.get("run_id") != body.run_id:
        # Undeploy the existing model first
        send_job("compute.undeploy_model", {"run_id": existing["run_id"]})
        db.deployments.update_one(
            {"run_id": existing["run_id"]},
            {"$set": {"status": "stopped"}},
        )

    # Upsert deployment record
    db.deployments.update_one(
        {"run_id": body.run_id},
        {"$set": {
            "run_id": body.run_id,
            "model": run["model"],
            "model_artifact_path": run["model_artifact_path"],
            "quantization_type": run["quantization_type"],
            "status": "deploying",
            "deployed_at": datetime.now(timezone.utc),
        }},
        upsert=True,
    )

    # Queue Celery task
    send_job("compute.deploy_model", {
        "run_id": body.run_id,
        "model_path": run["model_artifact_path"],
        "quant_type": run["quantization_type"],
    })

    return {"run_id": body.run_id, "status": "deploying"}


@app.delete("/api/models/deploy")
async def undeploy_model():
    """Stop the currently deployed model."""
    deployment = db.deployments.find_one({"status": "running"})
    if not deployment:
        raise HTTPException(status_code=404, detail="No model currently deployed.")

    run_id = deployment["run_id"]
    send_job("compute.undeploy_model", {"run_id": run_id})
    db.deployments.update_one(
        {"run_id": run_id},
        {"$set": {"status": "stopping"}},
    )
    return {"run_id": run_id, "status": "stopping"}


@app.get("/api/models/serving-status")
async def serving_status():
    """Return the current deployment state."""
    deployment = db.deployments.find_one(
        {"status": {"$in": ["deploying", "running", "stopping"]}},
        {"_id": 0},
    )
    if not deployment:
        return {"status": "none"}
    # Convert datetime for JSON serialization
    if "deployed_at" in deployment:
        deployment["deployed_at"] = deployment["deployed_at"].isoformat()
    return deployment


@app.patch("/api/job/{run_id}/deploy-status")
async def update_deploy_status(run_id: str, body: DeployStatusUpdate):
    """Callback from compute node to update deployment status."""
    if body.status not in ("running", "stopped", "failed"):
        raise HTTPException(status_code=400, detail="Invalid deploy status.")
    result = db.deployments.update_one(
        {"run_id": run_id},
        {"$set": {"status": body.status}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Deployment not found.")
    return {"run_id": run_id, "status": body.status}


@app.post("/api/chat/completions")
async def chat_completions(request: dict):
    """Proxy chat requests to vLLM on the GPU pod."""
    deployment = db.deployments.find_one({"status": "running"})
    if not deployment:
        raise HTTPException(status_code=503, detail="No model currently deployed.")

    # vLLM requires a "model" field — use the artifact path the model was loaded from
    if "model" not in request:
        request["model"] = deployment.get("model_artifact_path", "default")

    stream = request.get("stream", False)

    if stream:
        async def event_stream():
            try:
                timeout = httpx.Timeout(10.0, read=300.0)  # generous read timeout for streaming
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST",
                        f"{VLLM_BASE_URL}/v1/chat/completions",
                        json=request,
                    ) as resp:
                        if resp.status_code != 200:
                            body = await resp.aread()
                            yield f"data: {body.decode()}\n\n"
                            return
                        async for chunk in resp.aiter_bytes():
                            yield chunk
            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{VLLM_BASE_URL}/v1/chat/completions",
                json=request,
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
