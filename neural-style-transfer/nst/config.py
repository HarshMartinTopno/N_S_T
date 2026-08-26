"""Run configuration for a single style transfer job."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .models import SUPPORTED_MODELS

OPTIMIZERS = ("lbfgs", "adam")
INIT_METHODS = ("content", "random", "style")

DEFAULT_ITERATIONS = {"lbfgs": 300, "adam": 1500}


@dataclass
class TransferConfig:
    """Everything one style transfer run needs.

    Weight intuition:
      content_weight  higher -> the subject stays recognizable
      style_weight    higher -> brushwork and palette take over
      tv_weight       higher -> smoother output, at the cost of fine detail
    """

    content_img: Path
    style_img: Path
    output_dir: Path

    height: int = 400
    content_weight: float = 1e5
    style_weight: float = 3e4
    tv_weight: float = 1e0

    model: str = "vgg19"
    optimizer: str = "lbfgs"
    init_method: str = "content"

    iterations: Optional[int] = None
    saving_freq: int = -1          # -1 -> only the final image; N -> every N steps
    preview_freq: int = 10         # write a live preview every N steps; 0 disables
    seed: Optional[int] = None

    def __post_init__(self):
        self.content_img = Path(self.content_img)
        self.style_img = Path(self.style_img)
        self.output_dir = Path(self.output_dir)

        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"model must be one of {SUPPORTED_MODELS}, got {self.model!r}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if self.init_method not in INIT_METHODS:
            raise ValueError(f"init_method must be one of {INIT_METHODS}, got {self.init_method!r}")
        if self.height < 64:
            raise ValueError("height must be at least 64 pixels")

        if self.iterations is None:
            self.iterations = DEFAULT_ITERATIONS[self.optimizer]
        if self.iterations < 1:
            raise ValueError("iterations must be positive")

    def as_dict(self) -> dict:
        data = asdict(self)
        for key in ("content_img", "style_img", "output_dir"):
            data[key] = str(data[key])
        return data

    def output_name(self) -> str:
        return (f"{self.content_img.stem}__{self.style_img.stem}"
                f"_{self.model}_{self.optimizer}_h{self.height}"
                f"_cw{self.content_weight:g}_sw{self.style_weight:g}_tv{self.tv_weight:g}.jpg")
