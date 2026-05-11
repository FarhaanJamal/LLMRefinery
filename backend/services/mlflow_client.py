import os
import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


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
                "eval_mode": data.params.get("eval_mode", "quick"),
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
