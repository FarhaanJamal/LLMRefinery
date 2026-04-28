import os
import random
import time

import mlflow
import requests
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:16379/0")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:15000")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:18080")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = Celery("compute", broker=REDIS_URL, backend=REDIS_URL)
app.conf.worker_concurrency = 1


def _update_job_status(job_id: str, status: str):
    """Notify backend to update MongoDB job status."""
    try:
        requests.patch(
            f"{BACKEND_URL}/api/job/{job_id}/status",
            json={"status": status},
            timeout=5,
        )
    except Exception as e:
        print(f"[WARN] Failed to update job status: {e}")


@app.task(name="compute.run_pipeline")
def run_pipeline(payload):
    """
    Full pipeline: fine-tune → quantize → evaluate.
    Stub implementation — real ML logic added in Phase 3.
    """
    job_id = payload["job_id"]
    model = payload["model"]
    task = payload.get("task", "qlora")
    params = payload["params"]
    dataset_path = payload["dataset_path"]

    _update_job_status(job_id, "running")

    print(f"[STUB] Pipeline started: job_id={job_id}")
    print(f"[STUB] Model: {model}, Task: {task}")
    print(f"[STUB] Params: {params}")
    print(f"[STUB] Dataset: {dataset_path}")

    # --- Log to MLflow ---
    experiment_name = "llm-refinery"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"job-{job_id[:8]}"):
        # Log params
        mlflow.log_param("job_id", job_id)
        mlflow.log_param("model", model)
        mlflow.log_param("quantization_type", params.get("quant_type", "none"))
        mlflow.log_param("lora_rank", params.get("r", 16))
        mlflow.log_param("lora_alpha", params.get("alpha", 32))
        mlflow.log_param("model_artifact_path", f"/workspace/models/{job_id}")

        # Phase 3: Replace with actual calls
        # 1. peft_train.run(payload)
        # 2. quantize.run(payload)
        # 3. evaluate.run(payload)

        # Stub metrics (random values for Pareto chart testing)
        mlflow.log_metric("eval_accuracy_score", round(random.uniform(0.60, 0.95), 4))
        mlflow.log_metric("inference_latency", round(random.uniform(5.0, 50.0), 2))
        mlflow.log_metric("vram_max_allocated", round(random.uniform(8.0, 22.0), 2))
        quant = params.get("quant_type", "none")
        mlflow.log_metric("compression_ratio", round(random.uniform(2.0, 4.0), 2) if quant != "none" else 1.0)
        mlflow.log_metric("time_to_train", round(random.uniform(60.0, 600.0), 1))

    print(f"[STUB] Pipeline complete: job_id={job_id}")
    _update_job_status(job_id, "completed")
    return {"job_id": job_id, "status": "completed"}
