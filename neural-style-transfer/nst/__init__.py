"""Neural Style Transfer (Gatys et al., CVPR 2016) in PyTorch."""

from .config import TransferConfig, OPTIMIZERS, INIT_METHODS, DEFAULT_ITERATIONS
from .engine import Progress, TransferCancelled, run_style_transfer, resolve_device
from .models import SUPPORTED_MODELS

__version__ = "1.0.0"

__all__ = [
    "TransferConfig",
    "Progress",
    "TransferCancelled",
    "run_style_transfer",
    "resolve_device",
    "SUPPORTED_MODELS",
    "OPTIMIZERS",
    "INIT_METHODS",
    "DEFAULT_ITERATIONS",
]
