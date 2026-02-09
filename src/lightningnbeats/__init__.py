import os
import torch
from .models import *
from .loaders import *
from .losses import *
from .blocks.blocks import *


def get_best_accelerator() -> str:
    """Detect the best available accelerator for PyTorch.

    Returns 'cuda' if a CUDA GPU is available, 'mps' if Apple Metal
    Performance Shaders is available (Apple Silicon), otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

