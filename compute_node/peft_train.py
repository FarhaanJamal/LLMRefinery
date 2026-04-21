"""
QLoRA fine-tuning script.
Stub — real implementation in Phase 3.

Will:
  1. Load base model in 4-bit via bitsandbytes (load_in_4bit=True)
  2. Inject PEFT LoRA adapter
  3. Train on dataset pulled from MinIO
  4. Save LoRA adapter to ~/models/{job_id}/lora_adapter/
  5. Clean up VRAM
"""
import gc


def run(payload):
    job_id = payload["job_id"]
    print(f"[STUB] peft_train: job_id={job_id} — not implemented yet")

    # Phase 3: actual training logic here

    # VRAM cleanup (will use torch in Phase 3)
    gc.collect()

    return {"job_id": job_id, "step": "peft_train", "status": "completed"}
