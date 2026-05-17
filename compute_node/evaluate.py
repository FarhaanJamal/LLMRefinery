"""
Evaluation: quick eval (ROUGE-L on test split + latency) and
optional full eval (lm-evaluation-harness benchmarks).
"""
import gc
import time

import json
import torch
from datasets import Dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer


def _is_awq_model(model_path: str) -> bool:
    """Check if the model at model_path is AWQ-quantized."""
    config_path = f"{model_path}/config.json"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        return "quantization_config" in cfg and cfg["quantization_config"].get("quant_method") == "awq"
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def _load_model_and_tokenizer(model_path: str):
    """Load the final model (quantized or merged) for evaluation."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base models lack a chat template — set a default
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "### {{ message['role'] | capitalize }}:\n"
            "{{ message['content'] }}"
            "{% if not loop.last %}\n\n{% endif %}"
            "{% endfor %}"
            "{{ eos_token }}"
        )

    if _is_awq_model(model_path):
        # Load AWQ models via autoawq directly — transformers' AWQ path
        # unconditionally requires gptqmodel which we don't want.
        from awq import AutoAWQForCausalLM
        awq_model = AutoAWQForCausalLM.from_quantized(
            model_path, fuse_layers=False, device_map="auto",
        )
        model = awq_model.model  # unwrap to standard transformers model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def _format_messages(messages: list[dict], tokenizer, add_generation_prompt: bool = False) -> str:
    """Format chat messages using the tokenizer's chat template, with fallback for base models."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"### System:\n{content}")
        elif role == "user":
            parts.append(f"### User:\n{content}")
        elif role == "assistant":
            parts.append(f"### Assistant:\n{content}")
    text = "\n\n".join(parts)
    if add_generation_prompt:
        text += "\n\n### Assistant:\n"
    else:
        text += tokenizer.eos_token
    return text


def _generate_response(model, tokenizer, messages: list[dict], max_new_tokens: int = 256, max_seq_length: int = 512) -> str:
    """Generate a response given chat messages (exclude the last assistant turn)."""
    # Build prompt from all messages except the last assistant response
    prompt_messages = []
    for msg in messages:
        if msg["role"] == "assistant":
            break
        prompt_messages.append(msg)

    prompt = _format_messages(prompt_messages, tokenizer, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=None,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _quick_eval(model, tokenizer, test_dataset: Dataset, max_samples: int = 50, max_new_tokens: int = 256, max_seq_length: int = 512) -> dict:
    """
    Run inference on test split, compute ROUGE-L and latency.
    Returns dict of metrics.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = []
    total_tokens = 0
    total_time = 0.0

    torch.cuda.reset_peak_memory_stats()

    n_samples = min(len(test_dataset), max_samples)
    for i in range(n_samples):
        sample = test_dataset[i]
        messages = sample["messages"]
        reference = sample["text_target"]

        start = time.time()
        generated = _generate_response(model, tokenizer, messages, max_new_tokens=max_new_tokens, max_seq_length=max_seq_length)
        elapsed = time.time() - start

        # Count generated tokens
        gen_tokens = len(tokenizer.encode(generated, add_special_tokens=False))
        total_tokens += gen_tokens
        total_time += elapsed

        # ROUGE-L
        score = scorer.score(reference, generated)
        rouge_scores.append(score["rougeL"].fmeasure)

        if i < 3:
            print(f"[Eval] Sample {i}: ROUGE-L={score['rougeL'].fmeasure:.3f} "
                  f"({gen_tokens} tokens in {elapsed:.2f}s)")

    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
    vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"[Eval] Quick eval done: ROUGE-L={avg_rouge:.4f}, "
          f"latency={tokens_per_sec:.1f} tok/s, VRAM={vram_gb:.2f}GB")

    return {
        "eval_accuracy_score": round(avg_rouge, 4),
        "inference_latency": round(tokens_per_sec, 2),
        "vram_max_allocated": round(vram_gb, 2),
    }


def _full_eval(model_path: str) -> dict:
    """
    Run lm-evaluation-harness on medical-related MMLU tasks.
    Returns dict of benchmark scores.
    """
    import lm_eval

    print("[Eval] Running lm-evaluation-harness (medical MMLU tasks)...")

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=float16,trust_remote_code=True",
        tasks=["mmlu_medical_genetics", "mmlu_anatomy", "mmlu_clinical_knowledge"],
        batch_size="auto",
        num_fewshot=5,
    )

    metrics = {}
    for task_name, task_result in results.get("results", {}).items():
        acc = task_result.get("acc,none", task_result.get("acc", 0.0))
        metrics[task_name] = round(acc, 4)
        print(f"[Eval] {task_name}: {acc:.4f}")

    return metrics


def run(
    payload: dict,
    model_path: str,
    test_dataset: Dataset,
    compression_ratio: float,
    time_to_train: float,
) -> dict:
    """
    Evaluate the final model.

    Args:
        payload: job config (includes params.eval_mode)
        model_path: path to quantized or merged model
        test_dataset: HuggingFace Dataset with "messages" and "text_target"
        compression_ratio: from quantize step
        time_to_train: peft_time + quantize_time

    Returns:
        dict of all metrics for MLflow logging
    """
    job_id = payload["job_id"]
    eval_mode = payload.get("params", {}).get("eval_mode", "quick")
    max_new_tokens = payload.get("params", {}).get("max_new_tokens", 256)
    max_seq_length = payload.get("params", {}).get("max_seq_length", 512)

    print(f"[Eval] Starting evaluation: job_id={job_id}, mode={eval_mode}")
    print(f"[Eval] Model: {model_path}, test samples: {len(test_dataset)}")

    # --- Quick eval (always) ---
    model, tokenizer = _load_model_and_tokenizer(model_path)
    metrics = _quick_eval(model, tokenizer, test_dataset, max_new_tokens=max_new_tokens, max_seq_length=max_seq_length)

    # Cleanup model before full eval (lm-eval loads its own)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Add pass-through metrics
    metrics["compression_ratio"] = compression_ratio
    metrics["time_to_train"] = time_to_train

    # --- Full eval (optional) ---
    if eval_mode == "full":
        bench_metrics = _full_eval(model_path)
        metrics.update(bench_metrics)

        gc.collect()
        torch.cuda.empty_cache()

    print(f"[Eval] All metrics: {metrics}")
    return metrics


# Run on pod: cd /workspace/compute_node && python evaluate.py
if __name__ == "__main__":
    import os
    from services.dataset_utils import load_and_split
    import cuda_setup  # noqa: F401 — must be first to pre-load CUDA 13 libs

    # Load test split from the same dataset used for training
    _, test_ds = load_and_split(os.path.join(os.path.dirname(__file__), "tmp", "test.jsonl"))

    # Point to the merged model from a previous peft_train run
    # Change this path if your job_id differs
    model_path = "/workspace/models/test-001/merged"

    payload = {
        "job_id": "test-001",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "params": {"r": 8, "alpha": 16, "quant_type": "none", "eval_mode": "quick"},
    }

    metrics = run(
        payload=payload,
        model_path=model_path,
        test_dataset=test_ds,
        compression_ratio=1.0,
        time_to_train=0.0,
    )
    print(metrics)
