import json
import uuid
from datetime import datetime, timezone
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
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
    quant_type: str = "awq"  # "awq", "gptq", or "none"


class ExperimentRequest(BaseModel):
    model: str
    task: str = "qlora"
    params: ExperimentParams
    dataset_path: str


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
    if request.params.quant_type not in ("awq", "gptq", "none"):
        raise HTTPException(status_code=400, detail="quant_type must be 'awq', 'gptq', or 'none'.")

    job_id = str(uuid.uuid4())

    payload = {
        "job_id": job_id,
        "model": request.model,
        "task": request.task,
        "params": {
            "r": request.params.r,
            "alpha": request.params.alpha,
            "quant_type": request.params.quant_type,
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


@app.get("/api/experiments/results")
async def get_experiment_results():
    results = get_all_results()
    return {"experiments": results}
