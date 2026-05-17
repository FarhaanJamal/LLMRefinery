import os
import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_run_info(run_id: str) -> dict | None:
    """Fetch a single run by ID. Returns None if not found."""
    client = mlflow.tracking.MlflowClient()
    try:
        run = client.get_run(run_id)
    except Exception:
        return None
    data = run.data
    quant_type = data.params.get("quantization_type", "none")
    return {
        "run_id": run.info.run_id,
        "job_id": data.params.get("job_id", ""),
        "model": data.params.get("model", ""),
        "quantization_type": quant_type,
        "model_artifact_path": data.params.get("model_artifact_path", ""),
    }


def count_runs_for_job(job_id: str) -> int:
    """Count how many active (non-deleted) MLflow runs share this job_id."""
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in client.search_experiments()],
        filter_string=f"params.job_id = '{job_id}'",
    )
    return len(runs)


def get_all_results() -> list[dict]:
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()

    results = []
    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        for run in runs:
            data = run.data
            quant_type = data.params.get("quantization_type", "none")
            results.append({
                "run_id": run.info.run_id,
                "status": run.info.status,
                "duration_ms": (run.info.end_time or 0) - (run.info.start_time or 0),
                "job_id": data.params.get("job_id", ""),
                "model": data.params.get("model", ""),
                "quantization_type": quant_type,
                "lora_rank": data.params.get("lora_rank", ""),
                "lora_alpha": data.params.get("lora_alpha", ""),
                "lora_dropout": data.params.get("lora_dropout", ""),
                "target_modules": data.params.get("target_modules", ""),
                "num_train_epochs": data.params.get("num_train_epochs", ""),
                "max_steps": data.params.get("max_steps", ""),
                "learning_rate": data.params.get("learning_rate", ""),
                "batch_size": data.params.get("batch_size", ""),
                "gradient_accumulation_steps": data.params.get("gradient_accumulation_steps", ""),
                "lr_scheduler_type": data.params.get("lr_scheduler_type", ""),
                "warmup_steps": data.params.get("warmup_steps", ""),
                "max_grad_norm": data.params.get("max_grad_norm", ""),
                "seed": data.params.get("seed", ""),
                "max_seq_length": data.params.get("max_seq_length", ""),
                "w_bit": data.params.get("w_bit", ""),
                "q_group_size": data.params.get("q_group_size", ""),
                "eval_mode": data.params.get("eval_mode", "quick"),
                "max_new_tokens": data.params.get("max_new_tokens", ""),
                # Path to servable model on the GPU pod
                "model_artifact_path": data.params.get("model_artifact_path", ""),
                # System/training params
                "time_to_train": float(data.params.get("time_to_train", 0)),
                "compression_ratio": float(data.params.get("compression_ratio",
                    1.0 if quant_type == "none" else 0.0)),
                # Model quality metrics
                "accuracy": data.metrics.get("eval_accuracy_score", 0.0),
                "latency": data.metrics.get("inference_latency", 0.0),
                "vram_max_allocated": data.metrics.get("vram_max_allocated", 0.0),
            })

    return results
