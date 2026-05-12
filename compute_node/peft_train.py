"""
QLoRA fine-tuning: load base model in 4-bit, inject LoRA adapter,
train with SFTTrainer, save adapter, merge into base, cleanup VRAM.
"""
import gc
import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "/workspace/models")


def _formatting_func(example, tokenizer):
    """Format messages into a single training string using the model's chat template."""
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return text


def run(payload: dict, train_dataset: Dataset) -> dict:
    """
    QLoRA fine-tune a model on the training dataset.

    Args:
        payload: job config with model, params, job_id
        train_dataset: HuggingFace Dataset with "messages" column

    Returns:
        {
            "merged_path": str,  # path to merged full model
            "adapter_path": str,  # path to LoRA adapter
            "time_to_train": float,  # seconds
        }
    """
    job_id = payload["job_id"]
    model_name = payload["model"]
    params = payload["params"]
    lora_r = params.get("r", 16)
    lora_alpha = params.get("alpha", 32)

    output_dir = Path(MODEL_OUTPUT_DIR) / job_id
    adapter_path = output_dir / "lora_adapter"
    merged_path = output_dir / "merged"
    adapter_path.mkdir(parents=True, exist_ok=True)
    merged_path.mkdir(parents=True, exist_ok=True)

    print(f"[Train] Starting QLoRA fine-tune: {model_name}")
    print(f"[Train] LoRA r={lora_r}, alpha={lora_alpha}, samples={len(train_dataset)}")
    start_time = time.time()

    # --- 1. Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 2. Load model in 4-bit ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Use flash_attention_2 if available and GPU supports it, otherwise fall back
    attn_impl = "eager"
    if torch.cuda.get_device_capability()[0] >= 8:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model = prepare_model_for_kbit_training(model)

    # --- 3. Apply LoRA ---
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. Training arguments ---
    training_args = SFTConfig(
        output_dir=str(adapter_path),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        max_grad_norm=0.3,
        seed=42,
        max_length=512,
    )

    # --- 5. Train ---
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        formatting_func=lambda ex: _formatting_func(ex, tokenizer),
    )
    trainer.train()

    # --- 6. Save adapter ---
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"[Train] Adapter saved to {adapter_path}")

    # --- 7. Merge LoRA into base and save ---
    # Cannot merge directly from 4-bit model — reload base in fp16,
    # apply the saved adapter, then merge for clean fp16 weights.
    print("[Train] Reloading base model in fp16 for clean merge...")
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    merged_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    print(f"[Train] Merged model saved to {merged_path}")

    train_time = time.time() - start_time
    print(f"[Train] Fine-tuning complete in {train_time:.1f}s")

    # --- 8. VRAM cleanup ---
    del base_model, merged_model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "merged_path": str(merged_path),
        "adapter_path": str(adapter_path),
        "time_to_train": train_time,
    }
# Run on pod: cd /workspace/compute_node && python peft_train.py
if __name__ == "__main__":
    from services.dataset_utils import load_and_split
    import cuda_setup  # noqa: F401 — must be first to pre-load CUDA 13 libs
    
    # Use a tiny model for testing
    train_ds, test_ds = load_and_split(os.path.join(os.path.dirname(__file__), "tmp", "test.jsonl"))
    
    payload = {
        "job_id": "test-001",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "params": {"r": 8, "alpha": 16, "quant_type": "none"},
    }
    result = run(payload, train_ds)
    print(result)