"""
Quantization: AWQ on the merged model.
If quant_type=="none", skips quantization and uses merged FP16 as-is.
"""

import gc
import os
import time
from pathlib import Path

import torch


MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "/workspace/models")


def _get_dir_size_gb(path: str) -> float:
    """Get total size of a directory in GB."""
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 ** 3)


def _quantize_awq(merged_path: str, output_path: str, calib_data: list[str] = None) -> None:
    """Quantize with AutoAWQ (4-bit, group_size=128)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print("[Quantize] Loading model for AWQ quantization...")
    model = AutoAWQForCausalLM.from_pretrained(merged_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(merged_path)

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    print("[Quantize] Running AWQ quantization...")
    kwargs = {"tokenizer": tokenizer, "quant_config": quant_config}
    if calib_data:
        kwargs["calib_data"] = calib_data
    model.quantize(**kwargs)
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"[Quantize] AWQ model saved to {output_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


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


def run(payload: dict, merged_path: str, train_dataset=None) -> dict:
    """
    Quantize the merged model (or skip if quant_type=="none").

    Args:
        payload: job config with params.quant_type
        merged_path: path to the merged FP16 model from peft_train
        train_dataset: optional HuggingFace Dataset for calibration

    Returns:
        {
            "model_path": str,  # path to final servable model
            "compression_ratio": float,
            "quantize_time": float,  # seconds
        }
    """
    job_id = payload["job_id"]
    quant_type = payload["params"].get("quant_type", "awq")

    output_dir = Path(MODEL_OUTPUT_DIR) / job_id
    quantized_path = output_dir / "quantized"
    quantized_path.mkdir(parents=True, exist_ok=True)

    merged_size = _get_dir_size_gb(merged_path)

    if quant_type == "none":
        print("[Quantize] Skipping — using merged FP16 model as-is")
        return {
            "model_path": merged_path,
            "compression_ratio": 1.0,
            "quantize_time": 0.0,
        }

    print(f"[Quantize] Starting {quant_type.upper()} quantization...")
    start_time = time.time()

    # Prepare calibration data from training dataset
    calib_data = None
    if train_dataset is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(merged_path)
        calib_data = _prepare_calib_data(train_dataset, tokenizer)
        print(f"[Quantize] Using {len(calib_data)} samples from training data for calibration")

    if quant_type == "awq":
        _quantize_awq(merged_path, str(quantized_path), calib_data)
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}. Use 'awq' or 'none'.")

    quantize_time = time.time() - start_time
    quantized_size = _get_dir_size_gb(str(quantized_path))
    compression_ratio = merged_size / quantized_size if quantized_size > 0 else 1.0

    print(f"[Quantize] Done in {quantize_time:.1f}s — "
          f"{merged_size:.2f}GB → {quantized_size:.2f}GB "
          f"(compression={compression_ratio:.2f}x)")

    return {
        "model_path": str(quantized_path),
        "compression_ratio": round(compression_ratio, 2),
        "quantize_time": quantize_time,
    }


# Run on pod: cd /workspace/compute_node && python quantize.py
if __name__ == "__main__":
    import cuda_setup  # noqa: F401 — must be first to pre-load CUDA 13 libs
    
    # Expects a merged model from peft_train.py at this path
    merged_path = os.path.join(MODEL_OUTPUT_DIR, "test-001", "merged")

    # Load the same training dataset for calibration
    from services.dataset_utils import load_and_split
    train_ds, _ = load_and_split(os.path.join(os.path.dirname(__file__), "tmp", "test.jsonl"))

    # Test AWQ quantization
    payload = {
        "job_id": "test-001",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "params": {"quant_type": "awq"},
    }
    result = run(payload, merged_path, train_dataset=train_ds)
    print(result)
