"""The optimization loop.

Nothing in the network is trained. A single image tensor is declared a leaf with
``requires_grad=True`` and gradient descent is run directly on its pixels until
its VGG features match the content image and its Gram matrices match the style
image.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.optim import LBFGS, Adam

from .config import TransferConfig
from .image_utils import prepare_img, save_tensor_as_image
from .losses import build_loss, gram_matrix
from .models import build_model

logger = logging.getLogger(__name__)

__all__ = ["Progress", "TransferCancelled", "run_style_transfer", "resolve_device"]


class TransferCancelled(Exception):
    """Raised inside the optimization loop when a caller asks it to stop."""


@dataclass
class Progress:
    """One snapshot of the optimization, handed to the progress callback."""

    iteration: int
    total_iterations: int
    total_loss: float
    content_loss: float
    style_loss: float
    tv_loss: float
    preview_path: Optional[Path] = None

    @property
    def fraction(self) -> float:
        return min(1.0, (self.iteration + 1) / max(self.total_iterations, 1))


ProgressCallback = Callable[[Progress], None]
ShouldStop = Callable[[], bool]


def resolve_device(prefer: str = "auto") -> torch.device:
    """Pick a compute device. ``prefer`` is ``auto``, ``cuda``, ``mps`` or ``cpu``."""
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(prefer)


def _initial_image(cfg: TransferConfig, content_img: torch.Tensor,
                   device: torch.device) -> torch.Tensor:
    """Build the starting point for optimization.

    ``content`` converges fastest and keeps structure; ``random`` gives the style
    the most freedom but needs many more iterations; ``style`` is a middle ground.
    """
    if cfg.init_method == "content":
        init = content_img.clone()
    elif cfg.init_method == "random":
        noise = np.random.normal(loc=0.0, scale=90.0,
                                 size=tuple(content_img.shape)).astype(np.float32)
        init = torch.from_numpy(noise).to(device)
    else:  # style
        target_hw = tuple(int(v) for v in content_img.shape[2:])
        init = prepare_img(cfg.style_img, target_hw, device).clone()

    return init.requires_grad_(True)


def run_style_transfer(
    cfg: TransferConfig,
    on_progress: Optional[ProgressCallback] = None,
    should_stop: Optional[ShouldStop] = None,
    device: Optional[torch.device] = None,
) -> Path:
    """Run one transfer and return the path of the final image.

    ``on_progress`` is called once per iteration. ``should_stop`` is polled once
    per iteration; returning ``True`` aborts with :class:`TransferCancelled`.
    """
    device = device or resolve_device()
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = cfg.output_dir / cfg.output_name()
    preview_path = cfg.output_dir / "preview.jpg"

    logger.info("Running %s on %s (%d iterations, height=%d)",
                cfg.model, device, cfg.iterations, cfg.height)

    content_img = prepare_img(cfg.content_img, cfg.height, device)
    style_img = prepare_img(cfg.style_img, cfg.height, device)

    optimizing_img = _initial_image(cfg, content_img, device)

    neural_net, content_index, style_indices = build_model(cfg.model, device)

    with torch.no_grad():
        target_content = neural_net(content_img)[content_index].squeeze(0)
        target_style = [gram_matrix(fm)
                        for i, fm in enumerate(neural_net(style_img))
                        if i in style_indices]

    state = {"iteration": 0}

    def compute_losses():
        return build_loss(
            neural_net, optimizing_img, target_content, target_style,
            content_index, style_indices,
            cfg.content_weight, cfg.style_weight, cfg.tv_weight,
        )

    def after_iteration(total, content, style, tv) -> None:
        i = state["iteration"]
        with torch.no_grad():
            wrote_preview = None
            if cfg.preview_freq and (i % cfg.preview_freq == 0 or i == cfg.iterations - 1):
                save_tensor_as_image(optimizing_img, preview_path)
                wrote_preview = preview_path
            if cfg.saving_freq > 0 and i % cfg.saving_freq == 0:
                save_tensor_as_image(optimizing_img,
                                     cfg.output_dir / f"step_{i:05d}.jpg")

            if on_progress is not None:
                on_progress(Progress(
                    iteration=i,
                    total_iterations=cfg.iterations,
                    total_loss=float(total.item()),
                    content_loss=float(cfg.content_weight * content.item()),
                    style_loss=float(cfg.style_weight * style.item()),
                    tv_loss=float(cfg.tv_weight * tv.item()),
                    preview_path=wrote_preview,
                ))
        state["iteration"] = i + 1

        if should_stop is not None and should_stop():
            raise TransferCancelled()

    try:
        if cfg.optimizer == "adam":
            optimizer = Adam((optimizing_img,), lr=1e1)
            for _ in range(cfg.iterations):
                total, content, style, tv = compute_losses()
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()
                after_iteration(total, content, style, tv)
        else:
            optimizer = LBFGS((optimizing_img,), max_iter=cfg.iterations,
                              line_search_fn="strong_wolfe")

            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad(set_to_none=True)
                total, content, style, tv = compute_losses()
                if total.requires_grad:
                    total.backward()
                after_iteration(total, content, style, tv)
                return total

            optimizer.step(closure)
    except TransferCancelled:
        logger.info("Transfer cancelled at iteration %d", state["iteration"])
        raise

    save_tensor_as_image(optimizing_img, final_path)
    logger.info("Wrote %s", final_path)
    return final_path
