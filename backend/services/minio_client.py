import os
from minio import Minio
from minio.error import S3Error

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "datasets")

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)


def ensure_bucket():
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)


def upload_file(file_path: str, object_name: str, content_type: str = "application/octet-stream"):
    ensure_bucket()
    client.fput_object(MINIO_BUCKET, object_name, file_path, content_type=content_type)
    return f"s3://{MINIO_BUCKET}/{object_name}"


def upload_fileobj(data, length: int, object_name: str, content_type: str = "application/octet-stream"):
    ensure_bucket()
    client.put_object(MINIO_BUCKET, object_name, data, length, content_type=content_type)
    return f"s3://{MINIO_BUCKET}/{object_name}"


def download_file(object_name: str, file_path: str):
    client.fget_object(MINIO_BUCKET, object_name, file_path)


def delete_file(object_name: str):
    try:
        client.remove_object(MINIO_BUCKET, object_name)
    except S3Error:
        pass  # object may not exist
