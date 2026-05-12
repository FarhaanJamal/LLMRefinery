import cuda_setup  # noqa: F401 — must be first to pre-load CUDA 13 libs
import os
import shutil
import traceback

import mlflow
import requests
from celery import Celery

import peft_train
import quantize
import evaluate
import serve
from services.minio_client import download_dataset
from services.dataset_utils import load_and_split

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
    Full pipeline: download dataset → fine-tune → quantize → evaluate → log to MLflow.
    """
    job_id = payload["job_id"]
    model_name = payload["model"]
    params = payload["params"]
    dataset_path = payload["dataset_path"]

    _update_job_status(job_id, "training")

    print(f"[Pipeline] Started: job_id={job_id}")
    print(f"[Pipeline] Model: {model_name}, Params: {params}")
    print(f"[Pipeline] Dataset: {dataset_path}")

    experiment_name = "llm-refinery"
    mlflow.set_experiment(experiment_name)

    try:
        # Start MLflow run NOW so duration tracks the full pipeline
        with mlflow.start_run(run_name=f"job-{job_id[:8]}"):
            # Log params upfront
            mlflow.log_param("job_id", job_id)
            mlflow.log_param("model", model_name)
            mlflow.log_param("quantization_type", params.get("quant_type", "none"))
            mlflow.log_param("lora_rank", params.get("r", 16))
            mlflow.log_param("lora_alpha", params.get("alpha", 32))
            mlflow.log_param("eval_mode", params.get("eval_mode", "quick"))

            # --- 1. Download dataset from MinIO ---
            print("[Pipeline] Step 1/4: Downloading dataset...")
            local_dataset_path = download_dataset(dataset_path, job_id)

            # --- 2. Load & split dataset ---
            print("[Pipeline] Step 2/4: Loading and splitting dataset...")
            train_ds, test_ds = load_and_split(local_dataset_path)

            # --- 3. Fine-tune (QLoRA) ---
            print("[Pipeline] Step 3/4: Fine-tuning...")
            train_result = peft_train.run(payload, train_ds)
            merged_path = train_result["merged_path"]
            peft_time = train_result["time_to_train"]

            # --- 4. Quantize ---
            print("[Pipeline] Step 4/4: Quantizing...")
            _update_job_status(job_id, "quantizing")
            quant_result = quantize.run(payload, merged_path, train_dataset=train_ds)
            model_path = quant_result["model_path"]
            compression_ratio = quant_result["compression_ratio"]
            quantize_time = quant_result["quantize_time"]

            total_train_time = peft_time + quantize_time

            # --- 5. Evaluate ---
            print("[Pipeline] Evaluating...")
            _update_job_status(job_id, "evaluating")
            metrics = evaluate.run(
                payload=payload,
                model_path=model_path,
                test_dataset=test_ds,
                compression_ratio=compression_ratio,
                time_to_train=total_train_time,
            )

            # --- 6. Log results to MLflow ---
            print("[Pipeline] Logging results to MLflow...")
            mlflow.log_param("model_artifact_path", model_path)

            # System/training params (not model quality metrics)
            mlflow.log_param("time_to_train", round(total_train_time, 1))
            mlflow.log_param("compression_ratio", compression_ratio)

            # Model quality metrics
            mlflow.log_metric("eval_accuracy_score", metrics["eval_accuracy_score"])
            mlflow.log_metric("inference_latency", metrics["inference_latency"])
            mlflow.log_metric("vram_max_allocated", metrics["vram_max_allocated"])

            # Full eval benchmarks (if present)
            for key, value in metrics.items():
                if key.startswith("mmlu_"):
                    mlflow.log_metric(key, value)

        # --- 7. Cleanup temp dataset ---
        tmp_dir = f"/tmp/llm-refinery/{job_id}"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        print(f"[Pipeline] Complete: job_id={job_id}")
        _update_job_status(job_id, "completed")
        return {"job_id": job_id, "status": "completed", "metrics": metrics}

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"[Pipeline] FAILED: job_id={job_id}\n{error_msg}")

        # Log failure to MLflow
        try:
            with mlflow.start_run(run_name=f"job-{job_id[:8]}-FAILED"):
                mlflow.log_param("job_id", job_id)
                mlflow.log_param("model", model_name)
                mlflow.log_param("error", str(e)[:250])
        except Exception:
            pass

        _update_job_status(job_id, "failed")
        return {"job_id": job_id, "status": "failed", "error": str(e)}


def _update_deploy_status(run_id: str, status: str):
    """Notify backend of deployment status change."""
    try:
        requests.patch(
            f"{BACKEND_URL}/api/job/{run_id}/deploy-status",
            json={"status": status},
            timeout=5,
        )
    except Exception as e:
        print(f"[WARN] Failed to update deploy status: {e}")


@app.task(name="compute.deploy_model")
def deploy_model(payload):
    """
    Deploy a trained model via vLLM.
    payload: { run_id, model_path, quant_type }
    """
    run_id = payload["run_id"]
    model_path = payload["model_path"]
    quant_type = payload.get("quant_type", "none")

    print(f"[Deploy] Starting vLLM for run_id={run_id}")
    print(f"[Deploy] Model path: {model_path}, quant: {quant_type}")

    try:
        serve.start_serving(model_path, quant_type)
        print(f"[Deploy] vLLM is healthy for run_id={run_id}")
        _update_deploy_status(run_id, "running")
        return {"run_id": run_id, "status": "running"}
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"[Deploy] FAILED: run_id={run_id}\n{error_msg}")
        _update_deploy_status(run_id, "failed")
        return {"run_id": run_id, "status": "failed", "error": str(e)}


@app.task(name="compute.undeploy_model")
def undeploy_model(payload):
    """
    Stop the running vLLM server.
    payload: { run_id }
    """
    run_id = payload["run_id"]

    print(f"[Undeploy] Stopping vLLM for run_id={run_id}")
    serve.stop_serving()
    _update_deploy_status(run_id, "stopped")
    print(f"[Undeploy] Stopped for run_id={run_id}")
    return {"run_id": run_id, "status": "stopped"}
