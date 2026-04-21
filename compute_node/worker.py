import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery("compute", broker=REDIS_URL, backend=REDIS_URL)
app.conf.worker_concurrency = 1


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

    print(f"[STUB] Pipeline started: job_id={job_id}")
    print(f"[STUB] Model: {model}, Task: {task}")
    print(f"[STUB] Params: {params}")
    print(f"[STUB] Dataset: {dataset_path}")

    # Phase 3: Replace with actual calls
    # 1. peft_train.run(payload)
    # 2. quantize.run(payload)
    # 3. evaluate.run(payload)

    print(f"[STUB] Pipeline complete: job_id={job_id}")
    return {"job_id": job_id, "status": "completed"}
