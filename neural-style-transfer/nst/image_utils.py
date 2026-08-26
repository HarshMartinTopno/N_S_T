"""Image I/O and the ImageNet normalization the VGG encoders expect.

Images live in [0, 255] float space with the ImageNet mean subtracted and a
unit standard deviation. Keeping the std at 1 means the loss values stay in the
same units the original paper reported, so the published weight ratios transfer.
"""

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1.0, 1.0, 1.0]

_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255.0)),
    transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL),
])

TargetShape = Union[int, Sequence[int], None]


def load_image(img_path: Union[str, Path], target_shape: TargetShape = None) -> np.ndarray:
    """Read an image as float32 RGB in [0, 1], optionally resized.

    ``target_shape`` may be an int (target height, width scaled to keep aspect
    ratio) or a ``(height, width)`` pair.
    """
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")

    if target_shape is not None:
        if isinstance(target_shape, int):
            if target_shape != -1:
                width, height = img.size
                new_height = target_shape
                new_width = max(1, int(width * (new_height / height)))
                img = img.resize((new_width, new_height), Image.BICUBIC)
        else:
            height, width = int(target_shape[0]), int(target_shape[1])
            img = img.resize((width, height), Image.BICUBIC)

    return np.asarray(img, dtype=np.float32) / 255.0


def prepare_img(img_path: Union[str, Path], target_shape: TargetShape,
                device: torch.device) -> torch.Tensor:
    """Load an image and return a normalized ``(1, 3, H, W)`` tensor."""
    img = load_image(img_path, target_shape=target_shape)
    return _TO_TENSOR(img).to(device).unsqueeze(0)


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Undo normalization and return an ``(H, W, 3)`` uint8 RGB array."""
    img = tensor.squeeze(0).detach().cpu().numpy()
    img = np.moveaxis(img, 0, 2)
    img = img + np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def save_tensor_as_image(tensor: torch.Tensor, img_path: Union[str, Path]) -> Path:
    """Write an optimized tensor to disk as a viewable image."""
    img_path = Path(img_path)
    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8(tensor)).save(img_path)
    return img_path
