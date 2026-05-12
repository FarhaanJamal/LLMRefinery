"""vLLM process manager — start / stop / health-check a single vLLM server."""

import os
import signal
import site
import subprocess
import time
from pathlib import Path

import requests

VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
PID_FILE = Path("/tmp/vllm.pid")
HEALTH_URL = f"http://localhost:{VLLM_PORT}/health"

# Startup tuning for a single RTX 3090 (24 GB)
GPU_MEM_UTIL = float(os.getenv("VLLM_GPU_MEM_UTIL", "0.85"))


def _ensure_gptqmodel_stub():
    """Install a minimal gptqmodel stub into site-packages (once).

    Newer transformers unconditionally imports gptqmodel for AWQ models.
    vLLM has its own AWQ kernels and never calls gptqmodel, but it spawns
    child processes that also go through transformers.  Writing the stub
    into site-packages makes it visible to every Python process on the pod.
    """
    sp = site.getsitepackages()[0]  # e.g. /usr/local/lib/python3.11/dist-packages
    pkg = Path(sp) / "gptqmodel"
    if (pkg / "__init__.py").exists():
        return  # already installed (real or stub)

    quant = pkg / "quantization" / "awq" / "modules"
    utils = pkg / "utils"
    for d in [quant, utils]:
        d.mkdir(parents=True, exist_ok=True)

    (pkg / "__init__.py").write_text("__version__ = '0.0.0'\n")

    (pkg / "quantization" / "__init__.py").write_text(
        "from enum import Enum\n"
        "class METHOD(Enum):\n"
        "    AWQ_GEMM = 'awq_gemm'\n"
        "    AWQ = 'awq'\n"
        "    GPTQ = 'gptq'\n"
    )
    (pkg / "quantization" / "awq" / "__init__.py").write_text("")
    (pkg / "quantization" / "awq" / "modules" / "__init__.py").write_text("")
    (pkg / "quantization" / "awq" / "modules" / "act.py").write_text(
        "import torch.nn as nn\n"
        "class ScaledActivation(nn.Module):\n"
        "    def __init__(self, module=None, scales=None):\n"
        "        super().__init__()\n"
        "        self.module = module\n"
    )
    (utils / "__init__.py").write_text("")
    (utils / "importer.py").write_text(
        "import torch.nn as nn\n\n"
        "class _StubQuantLinear(nn.Module):\n"
        "    def __init__(self, **kwargs):\n"
        "        super().__init__()\n\n"
        "def hf_select_quant_linear_v2(**kwargs):\n"
        "    return _StubQuantLinear\n"
    )
    (utils / "model.py").write_text(
        "def hf_gptqmodel_post_init(model, use_act_order=False):\n"
        "    return model\n"
    )
    print(f"[Serve] Installed gptqmodel stub in {pkg}")


def start_serving(model_path: str, quant_type: str, timeout: int = 180) -> None:
    """Spawn vLLM as a detached subprocess. Blocks until the health endpoint is up.

    Raises RuntimeError if the server doesn't become healthy within *timeout*
    seconds or if the process exits early.
    """
    stop_serving()  # ensure nothing is already running

    # Install gptqmodel stub into site-packages so vLLM subprocesses find it
    _ensure_gptqmodel_stub()

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(VLLM_PORT),
        "--dtype", "half",
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--trust-remote-code",
        "--enforce-eager",  # skip torch.compile — workaround for inductor duplicate template bug
    ]
    # No --quantization flag: vLLM auto-detects AWQ from the model's config.json

    log_file = open("/tmp/vllm.log", "w")
    env = os.environ.copy()
    env["TORCHDYNAMO_DISABLE"] = "1"  # avoid torch inductor duplicate template bug
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # new process group so we can kill the tree
        env=env,
    )

    PID_FILE.write_text(str(proc.pid))

    # Wait for healthy
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Check process is still alive
        poll = proc.poll()
        if poll is not None:
            log_file.close()
            tail = _read_log_tail()
            raise RuntimeError(
                f"vLLM exited with code {poll} before becoming healthy.\n{tail}"
            )
        if is_healthy():
            return
        time.sleep(3)

    # Timed out — kill and report
    stop_serving()
    tail = _read_log_tail()
    raise RuntimeError(
        f"vLLM did not become healthy within {timeout}s.\n{tail}"
    )


def stop_serving() -> None:
    """Kill the running vLLM process group, if any."""
    if not PID_FILE.exists():
        return

    try:
        pid = int(PID_FILE.read_text().strip())
        # Kill the whole process group
        os.killpg(pid, signal.SIGTERM)
        # Give it a moment to shut down gracefully
        for _ in range(10):
            try:
                os.kill(pid, 0)  # probe — raises OSError if dead
                time.sleep(1)
            except OSError:
                break
        else:
            # Still alive after 10s — force kill
            os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass  # already dead
    finally:
        PID_FILE.unlink(missing_ok=True)


def is_running() -> bool:
    """Return True if the PID is alive AND the health endpoint responds."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # check process exists
    except (OSError, ValueError):
        PID_FILE.unlink(missing_ok=True)
        return False
    return is_healthy()


def is_healthy() -> bool:
    """Probe the vLLM /health endpoint."""
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def _read_log_tail(lines: int = 30) -> str:
    """Return the last *lines* of the vLLM log for diagnostics."""
    log = Path("/tmp/vllm.log")
    if not log.exists():
        return "(no log file)"
    all_lines = log.read_text().splitlines()
    return "\n".join(all_lines[-lines:])
