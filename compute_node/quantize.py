"""
Quantization script (AWQ / GPTQ).
Stub — real implementation in Phase 3.

Will:
  1. Merge LoRA weights into base model
  2. Run AutoAWQ or AutoGPTQ with calibration data
  3. Save quantized model to ~/models/{job_id}/quantized/
  4. Clean up VRAM
"""
import gc


def run(payload):
    job_id = payload["job_id"]
    quant_type = payload["params"].get("quant_type", "awq")
    print(f"[STUB] quantize ({quant_type}): job_id={job_id} — not implemented yet")

    if quant_type == "none":
        print(f"[STUB] Skipping quantization, using merged FP16 model")

    # Phase 3: actual quantization logic here

    # VRAM cleanup
    gc.collect()

    return {"job_id": job_id, "step": "quantize", "status": "completed"}
