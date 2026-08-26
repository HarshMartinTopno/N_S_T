"""The three loss terms that define Neural Style Transfer."""

from typing import List, Sequence, Tuple

import torch

__all__ = ["gram_matrix", "total_variation", "build_loss"]


def gram_matrix(x: torch.Tensor, should_normalize: bool = True) -> torch.Tensor:
    """Correlations between feature channels -- the texture signature of a layer.

    Spatial layout is collapsed, so what survives is *which features fire
    together*, not *where*. That is exactly what "style" means here.
    """
    b, ch, h, w = x.size()
    features = x.view(b, ch, h * w)
    gram = features.bmm(features.transpose(1, 2))
    if should_normalize:
        gram = gram / (ch * h * w)
    return gram


def total_variation(y: torch.Tensor) -> torch.Tensor:
    """Sum of absolute neighbouring-pixel differences. Penalizes speckle noise."""
    return (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:]))
            + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))


def build_loss(
    neural_net: torch.nn.Module,
    optimizing_img: torch.Tensor,
    target_content: torch.Tensor,
    target_style: Sequence[torch.Tensor],
    content_index: int,
    style_indices: Sequence[int],
    content_weight: float,
    style_weight: float,
    tv_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(total, content, style, tv)`` losses, all unweighted except total."""
    feature_maps = neural_net(optimizing_img)

    current_content = feature_maps[content_index].squeeze(0)
    content_loss = torch.nn.functional.mse_loss(current_content, target_content)

    current_style: List[torch.Tensor] = [
        gram_matrix(fm) for i, fm in enumerate(feature_maps) if i in style_indices
    ]
    style_loss = torch.zeros((), device=optimizing_img.device)
    for gram_target, gram_current in zip(target_style, current_style):
        style_loss = style_loss + torch.nn.functional.mse_loss(
            gram_current[0], gram_target[0], reduction="sum"
        )
    style_loss = style_loss / max(len(target_style), 1)

    tv_loss = total_variation(optimizing_img)

    total_loss = (content_weight * content_loss
                  + style_weight * style_loss
                  + tv_weight * tv_loss)

    return total_loss, content_loss, style_loss, tv_loss
