"""
MinIO client for the compute node.
Downloads datasets from the control plane's MinIO (via socat tunnel).
"""
import os
from pathlib import Path

from minio import Minio

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "127.0.0.1:19000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "datasets")


def get_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def download_dataset(s3_path: str, job_id: str) -> str:
    """
    Download a dataset from MinIO to a local temp directory.

    Args:
        s3_path: e.g. "s3://datasets/uuid.jsonl"
        job_id: used to create isolated temp dir

    Returns:
        Local file path (e.g. /tmp/llm-refinery/job_id/dataset.jsonl)
    """
    # Parse s3 path: "s3://datasets/uuid.jsonl" → bucket="datasets", key="uuid.jsonl"
    path = s3_path.replace("s3://", "")
    parts = path.split("/", 1)
    bucket = parts[0]
    object_name = parts[1] if len(parts) > 1 else ""

    # Create local directory
    local_dir = Path(f"/tmp/llm-refinery/{job_id}")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / "dataset.jsonl"

    # Download
    client = get_client()
    client.fget_object(bucket, object_name, str(local_path))

    print(f"[MinIO] Downloaded {s3_path} → {local_path}")
    return str(local_path)
