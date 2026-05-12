"""
Pre-load CUDA 13 runtime libraries so bitsandbytes/torch can find them
regardless of whether LD_LIBRARY_PATH is set in the shell.

Import this module BEFORE torch, bitsandbytes, or transformers.
"""
import ctypes
import os
from pathlib import Path

_CUDA13_LIB_DIR = Path("/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib")

if _CUDA13_LIB_DIR.exists():
    # Set env var for any child processes
    os.environ["LD_LIBRARY_PATH"] = (
        str(_CUDA13_LIB_DIR) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    # Pre-load shared libs into process so dlopen finds them
    for so_file in sorted(_CUDA13_LIB_DIR.glob("*.so.*")):
        try:
            ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
