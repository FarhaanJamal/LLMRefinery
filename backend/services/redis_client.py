import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery("llm_refinery", broker=REDIS_URL, backend=REDIS_URL)


def send_job(task_name: str, payload: dict):
    return celery_app.send_task(task_name, args=[payload])
