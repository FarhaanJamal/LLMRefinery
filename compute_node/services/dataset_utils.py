"""
Dataset utilities for the compute node.
Auto-detects format, converts to chat messages, splits train/test.
"""
import json
from pathlib import Path

from datasets import Dataset


def _detect_format(sample: dict) -> str:
    """Detect dataset format from keys in the first sample."""
    keys = set(sample.keys())
    if "conversations" in keys:
        return "sharegpt"
    if "instruction" in keys:
        return "alpaca"
    if "question" in keys and "answer" in keys:
        return "qa"
    if "text" in keys:
        return "text"
    raise ValueError(
        f"Unknown dataset format. Keys found: {keys}. "
        "Supported: alpaca (instruction/output), sharegpt (conversations), "
        "qa (question/answer), text (text)."
    )


def _alpaca_to_messages(sample: dict) -> list[dict]:
    """Convert Alpaca format to chat messages."""
    user_content = sample["instruction"]
    if sample.get("input"):
        user_content += f"\n\n{sample['input']}"
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": sample["output"]},
    ]


def _sharegpt_to_messages(sample: dict) -> list[dict]:
    """Convert ShareGPT/ChatML format to chat messages."""
    messages = []
    for turn in sample["conversations"]:
        role = turn.get("from", turn.get("role", "user"))
        # Normalize role names
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "model"):
            role = "assistant"
        elif role == "system":
            role = "system"
        messages.append({"role": role, "content": turn.get("value", turn.get("content", ""))})
    return messages


def _qa_to_messages(sample: dict) -> list[dict]:
    """Convert Q&A format to chat messages."""
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


def load_and_split(
    file_path: str,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Load a .jsonl file, auto-detect format, convert to chat messages,
    and split into train/test datasets.

    Each sample in the returned datasets has:
        - "messages": list of {"role": ..., "content": ...}
        - "text_target": the assistant's final response (for eval comparison)

    Args:
        file_path: path to .jsonl file
        test_ratio: fraction for test split (default 0.2)
        seed: random seed for reproducible splits

    Returns:
        (train_dataset, test_dataset) as HuggingFace Dataset objects
    """
    path = Path(file_path)
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if not samples:
        raise ValueError(f"Dataset is empty: {file_path}")

    # Detect format from first sample
    fmt = _detect_format(samples[0])
    print(f"[Dataset] Detected format: {fmt} ({len(samples)} samples)")

    # Convert all samples to messages format
    converter = {
        "alpaca": _alpaca_to_messages,
        "sharegpt": _sharegpt_to_messages,
        "qa": _qa_to_messages,
    }

    processed = []
    for sample in samples:
        if fmt == "text":
            # Pre-formatted text — no conversion needed
            processed.append({
                "messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": sample["text"]}],
                "text_target": sample["text"],
            })
        else:
            messages = converter[fmt](sample)
            # Extract the last assistant response as the eval target
            target = ""
            for msg in reversed(messages):
                if msg["role"] == "assistant":
                    target = msg["content"]
                    break
            processed.append({
                "messages": messages,
                "text_target": target,
            })

    # Create HuggingFace Dataset and split
    ds = Dataset.from_list(processed)
    split = ds.train_test_split(test_size=test_ratio, seed=seed)

    print(f"[Dataset] Split: {len(split['train'])} train, {len(split['test'])} test")
    return split["train"], split["test"]
