import cuda_setup  # noqa: F401 — must be first to pre-load CUDA 13 libs
import os
import shutil
import traceback

import mlflow
import requests
from celery import Celery
from transformers import AutoTokenizer

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


def _prepare_calib_data(train_dataset, tokenizer, n_samples=128) -> list[str]:
    """Convert training dataset messages into calibration text strings."""
    calib_texts = []
    for i, example in enumerate(train_dataset):
        if i >= n_samples:
            break
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        calib_texts.append(text)
    return calib_texts


def _log_mlflow_run(
    run_name: str,
    job_id: str,
    model_name: str,
    params: dict,
    model_path: str,
    quant_type_label: str,
    time_to_train: float,
    compression_ratio: float,
    metrics: dict,
):
    """Log a single MLflow run with all params and metrics."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("job_id", job_id)
        mlflow.log_param("model", model_name)
        mlflow.log_param("quantization_type", quant_type_label)
        mlflow.log_param("lora_rank", params.get("r", 16))
        mlflow.log_param("lora_alpha", params.get("alpha", 32))
        mlflow.log_param("eval_mode", params.get("eval_mode", "quick"))
        mlflow.log_param("model_artifact_path", model_path)
        mlflow.log_param("time_to_train", round(time_to_train, 1))
        mlflow.log_param("compression_ratio", compression_ratio)

        mlflow.log_metric("eval_accuracy_score", metrics["eval_accuracy_score"])
        mlflow.log_metric("inference_latency", metrics["inference_latency"])
        mlflow.log_metric("vram_max_allocated", metrics["vram_max_allocated"])

        for key, value in metrics.items():
            if key.startswith("mmlu_"):
                mlflow.log_metric(key, value)


@app.task(name="compute.run_pipeline")
def run_pipeline(payload):
    """
    Full pipeline: download dataset → fine-tune → quantize → evaluate → log to MLflow.
    When quant_type="awq", produces two MLflow runs: one for FP16 merged, one for AWQ.
    """
    job_id = payload["job_id"]
    model_name = payload["model"]
    params = payload["params"]
    dataset_path = payload["dataset_path"]
    quant_type = params.get("quant_type", "none")

    _update_job_status(job_id, "training")

    print(f"[Pipeline] Started: job_id={job_id}")
    print(f"[Pipeline] Model: {model_name}, Params: {params}")
    print(f"[Pipeline] Dataset: {dataset_path}")

    experiment_name = "llm-refinery"
    mlflow.set_experiment(experiment_name)

    try:
        # --- 1. Download dataset from MinIO ---
        print("[Pipeline] Step 1/4: Downloading dataset...")
        local_dataset_path = download_dataset(dataset_path, job_id)

        # --- 2. Load & split dataset ---
        print("[Pipeline] Step 2/4: Loading and splitting dataset...")
        train_ds, test_ds = load_and_split(local_dataset_path)

        # --- 3. Load tokenizer once for the whole pipeline ---
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # --- 4. Fine-tune (QLoRA) ---
        print("[Pipeline] Step 3/4: Fine-tuning...")
        train_result = peft_train.run(payload, train_ds, tokenizer=tokenizer)
        merged_path = train_result["merged_path"]
        peft_time = train_result["time_to_train"]

        # --- 5. Evaluate merged (FP16) model ---
        if quant_type == "awq":
            # For AWQ jobs: evaluate merged first, then quantize and evaluate again
            print("[Pipeline] Evaluating merged FP16 model...")
            _update_job_status(job_id, "evaluating")
            merged_metrics = evaluate.run(
                payload=payload,
                model_path=merged_path,
                test_dataset=test_ds,
                compression_ratio=1.0,
                time_to_train=peft_time,
            )

            # Log FP16 run to MLflow
            print("[Pipeline] Logging FP16 results to MLflow...")
            _log_mlflow_run(
                run_name=f"job-{job_id[:8]}-fp16",
                job_id=job_id,
                model_name=model_name,
                params=params,
                model_path=merged_path,
                quant_type_label="none",
                time_to_train=peft_time,
                compression_ratio=1.0,
                metrics=merged_metrics,
            )

        # --- 6. Quantize ---
        print("[Pipeline] Step 4/4: Quantizing...")
        _update_job_status(job_id, "quantizing")

        # Pre-compute calibration texts using the shared tokenizer
        calib_data = None
        if quant_type == "awq":
            calib_data = _prepare_calib_data(train_ds, tokenizer)

        quant_result = quantize.run(payload, merged_path, calib_data=calib_data)
        model_path = quant_result["model_path"]
        compression_ratio = quant_result["compression_ratio"]
        quantize_time = quant_result["quantize_time"]

        total_train_time = peft_time + quantize_time

        # --- 7. Evaluate final model ---
        print("[Pipeline] Evaluating...")
        _update_job_status(job_id, "evaluating")
        metrics = evaluate.run(
            payload=payload,
            model_path=model_path,
            test_dataset=test_ds,
            compression_ratio=compression_ratio,
            time_to_train=total_train_time,
        )

        # --- 8. Log results to MLflow ---
        print("[Pipeline] Logging results to MLflow...")
        run_name = f"job-{job_id[:8]}" if quant_type == "none" else f"job-{job_id[:8]}-awq"
        _log_mlflow_run(
            run_name=run_name,
            job_id=job_id,
            model_name=model_name,
            params=params,
            model_path=model_path,
            quant_type_label=quant_type,
            time_to_train=total_train_time,
            compression_ratio=compression_ratio,
            metrics=metrics,
        )

        # --- 9. Cleanup temp dataset ---
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


MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "/workspace/models")


@app.task(name="compute.cleanup_artifacts")
def cleanup_artifacts(payload):
    """
    Delete model artifacts from the pod.
    payload: { job_id, subdir? }
    If subdir is provided, delete only that specific path.
    Otherwise delete the entire models/{job_id}/ directory.
    """
    job_id = payload["job_id"]
    subdir = payload.get("subdir")

    if subdir:
        # Delete only a specific artifact path (e.g. merged/ or quantized/)
        if os.path.exists(subdir):
            shutil.rmtree(subdir)
            print(f"[Cleanup] Deleted artifact subdir: {subdir}")
        else:
            print(f"[Cleanup] No artifact subdir found: {subdir}")
    else:
        # Delete the entire job artifact directory
        artifact_dir = os.path.join(MODEL_OUTPUT_DIR, job_id)
        if os.path.exists(artifact_dir):
            shutil.rmtree(artifact_dir)
            print(f"[Cleanup] Deleted artifacts: {artifact_dir}")
        else:
            print(f"[Cleanup] No artifacts found: {artifact_dir}")

    return {"job_id": job_id, "deleted": True}
