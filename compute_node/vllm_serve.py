"""Wrapper that patches gptqmodel imports before launching vLLM.

Transformers unconditionally requires gptqmodel for AWQ model inspection,
but vLLM has its own AWQ kernels and never calls gptqmodel.
This script injects a fake gptqmodel into sys.modules before any
transformers import happens, then delegates to vLLM's server entrypoint.
"""

import sys
import importlib
import importlib.abc
import importlib.machinery
from enum import Enum
from types import ModuleType


# All fake submodules we need to provide
_STUB_MODULES = {
    "gptqmodel",
    "gptqmodel.quantization",
    "gptqmodel.quantization.awq",
    "gptqmodel.quantization.awq.modules",
    "gptqmodel.quantization.awq.modules.act",
    "gptqmodel.utils",
    "gptqmodel.utils.importer",
    "gptqmodel.utils.model",
}


class _GptqModelFinder(importlib.abc.MetaPathFinder):
    """Custom finder so importlib.util.find_spec('gptqmodel') works."""

    def find_spec(self, fullname, path, target=None):
        if fullname in _STUB_MODULES:
            return importlib.machinery.ModuleSpec(fullname, loader=None)
        return None


def _patch_gptqmodel():
    """Insert a fake gptqmodel package hierarchy into sys.modules."""
    if "gptqmodel" in sys.modules:
        return  # already available (real or previously patched)

    # Install finder FIRST so find_spec() calls return a valid spec
    sys.meta_path.insert(0, _GptqModelFinder())

    # Root package
    root = ModuleType("gptqmodel")
    root.__version__ = "0.0.0"
    root.__path__ = []

    # gptqmodel.quantization
    quant = ModuleType("gptqmodel.quantization")
    quant.__path__ = []

    class METHOD(Enum):
        AWQ_GEMM = "awq_gemm"
        AWQ = "awq"
        GPTQ = "gptq"

    quant.METHOD = METHOD

    # gptqmodel.quantization.awq
    awq = ModuleType("gptqmodel.quantization.awq")
    awq.__path__ = []

    # gptqmodel.quantization.awq.modules
    awq_modules = ModuleType("gptqmodel.quantization.awq.modules")
    awq_modules.__path__ = []

    # gptqmodel.quantization.awq.modules.act
    import torch.nn as nn

    class ScaledActivation(nn.Module):
        def __init__(self, module=None, scales=None):
            super().__init__()
            self.module = module

    awq_act = ModuleType("gptqmodel.quantization.awq.modules.act")
    awq_act.ScaledActivation = ScaledActivation

    # gptqmodel.utils
    utils = ModuleType("gptqmodel.utils")
    utils.__path__ = []

    # gptqmodel.utils.importer
    class _StubQuantLinear(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

    importer = ModuleType("gptqmodel.utils.importer")
    importer.hf_select_quant_linear_v2 = lambda **kwargs: _StubQuantLinear

    # gptqmodel.utils.model
    model_utils = ModuleType("gptqmodel.utils.model")
    model_utils.hf_gptqmodel_post_init = lambda model, use_act_order=False: model

    # Register everything
    modules = {
        "gptqmodel": root,
        "gptqmodel.quantization": quant,
        "gptqmodel.quantization.awq": awq,
        "gptqmodel.quantization.awq.modules": awq_modules,
        "gptqmodel.quantization.awq.modules.act": awq_act,
        "gptqmodel.utils": utils,
        "gptqmodel.utils.importer": importer,
        "gptqmodel.utils.model": model_utils,
    }

    # Set __spec__ on each so find_spec() returns non-None when checking sys.modules
    for name, mod in modules.items():
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)

    sys.modules.update(modules)


# Patch BEFORE any transformers/vllm import
_patch_gptqmodel()

# Now launch vLLM's OpenAI-compatible server (equivalent to python -m ...)
import runpy  # noqa: E402
runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
