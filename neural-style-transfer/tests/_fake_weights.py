"""Swap the pretrained VGG loader for a randomly-initialized one.

The tests verify plumbing -- tap indices, tensor shapes, loss maths, the job
lifecycle -- none of which depend on the learned weights. Skipping the ~550 MB
download keeps the suite fast and runnable offline and in CI.

Set NST_TEST_PRETRAINED=1 to test against the real ImageNet weights instead.
"""

import os

from torchvision import models

import nst.models as nst_models


def install() -> bool:
    """Patch the loader unless NST_TEST_PRETRAINED is set. Returns True if patched."""
    if os.getenv("NST_TEST_PRETRAINED") == "1":
        return False
    nst_models._vgg_features = lambda name, show_progress=False: (
        {"vgg16": models.vgg16, "vgg19": models.vgg19}[name](weights=None).features
    )
    return True
