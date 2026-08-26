"""VGG feature extractors used as frozen perceptual encoders.

Each model exposes a fixed set of intermediate activations. The network is never
trained -- only the pixels of the generated image are optimized -- so every
parameter is frozen at construction time.
"""

from collections import namedtuple

import torch
from torchvision import models

__all__ = ["Vgg16", "Vgg16Experimental", "Vgg19", "build_model", "SUPPORTED_MODELS"]

SUPPORTED_MODELS = ("vgg16", "vgg16-experimental", "vgg19")


def _vgg_features(name: str, show_progress: bool = False):
    """Load pretrained VGG features, tolerating old and new torchvision APIs."""
    try:  # torchvision >= 0.13
        from torchvision.models import VGG16_Weights, VGG19_Weights

        weights = {"vgg16": VGG16_Weights.IMAGENET1K_V1,
                   "vgg19": VGG19_Weights.IMAGENET1K_V1}[name]
        builder = {"vgg16": models.vgg16, "vgg19": models.vgg19}[name]
        return builder(weights=weights, progress=show_progress).features
    except ImportError:  # torchvision < 0.13
        builder = {"vgg16": models.vgg16, "vgg19": models.vgg19}[name]
        return builder(pretrained=True, progress=show_progress).features


def _freeze(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


class Vgg16(torch.nn.Module):
    """VGG16 exposing the four ReLU blocks that work well in practice."""

    def __init__(self, requires_grad: bool = False, show_progress: bool = False):
        super().__init__()
        features = _vgg_features("vgg16", show_progress)

        self.layer_names = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        self.content_feature_maps_index = 1              # relu2_2
        self.style_feature_maps_indices = list(range(len(self.layer_names)))

        self.slice1 = torch.nn.Sequential(*list(features[0:4]))
        self.slice2 = torch.nn.Sequential(*list(features[4:9]))
        self.slice3 = torch.nn.Sequential(*list(features[9:16]))
        self.slice4 = torch.nn.Sequential(*list(features[16:23]))

        if not requires_grad:
            _freeze(self)

    def forward(self, x):
        relu1_2 = self.slice1(x)
        relu2_2 = self.slice2(relu1_2)
        relu3_3 = self.slice3(relu2_2)
        relu4_3 = self.slice4(relu3_3)
        out = namedtuple("VggOutputs", self.layer_names)
        return out(relu1_2, relu2_2, relu3_3, relu4_3)


class Vgg16Experimental(torch.nn.Module):
    """VGG16 with a wider set of taps, for experimenting with layer choices."""

    def __init__(self, requires_grad: bool = False, show_progress: bool = False):
        super().__init__()
        features = _vgg_features("vgg16", show_progress)

        self.layer_names = ["relu1_1", "relu2_1", "relu2_2", "relu3_1",
                            "relu3_2", "relu4_1", "relu4_3", "relu5_1"]
        self.content_feature_maps_index = 4              # relu3_2
        self.style_feature_maps_indices = list(range(len(self.layer_names)))

        # (name, end_index_exclusive) -- cumulative slices of vgg16.features
        cuts = [("relu1_1", 2), ("relu2_1", 7), ("relu2_2", 9), ("relu3_1", 12),
                ("relu3_2", 14), ("relu4_1", 19), ("relu4_3", 23), ("relu5_1", 26)]
        start = 0
        self.slices = torch.nn.ModuleList()
        for _, end in cuts:
            self.slices.append(torch.nn.Sequential(*list(features[start:end])))
            start = end

        if not requires_grad:
            _freeze(self)

    def forward(self, x):
        taps = []
        for slice_ in self.slices:
            x = slice_(x)
            taps.append(x)
        out = namedtuple("VggOutputs", self.layer_names)
        return out(*taps)


class Vgg19(torch.nn.Module):
    """VGG19 with the taps used in the original Gatys et al. paper.

    Style comes from conv/relu {1,2,3,4,5}_1, content from conv4_2.
    ``use_relu`` picks the post-activation taps (slightly better in practice);
    setting it to ``False`` reproduces the paper's raw conv taps.
    """

    def __init__(self, requires_grad: bool = False, show_progress: bool = False,
                 use_relu: bool = True):
        super().__init__()
        features = _vgg_features("vgg19", show_progress)

        if use_relu:
            self.layer_names = ["relu1_1", "relu2_1", "relu3_1",
                                "relu4_1", "conv4_2", "relu5_1"]
            offset = 1
        else:
            self.layer_names = ["conv1_1", "conv2_1", "conv3_1",
                                "conv4_1", "conv4_2", "conv5_1"]
            offset = 0

        self.content_feature_maps_index = 4              # conv4_2
        self.style_feature_maps_indices = [
            i for i in range(len(self.layer_names)) if i != 4
        ]

        bounds = [(0, 1 + offset), (1 + offset, 6 + offset), (6 + offset, 11 + offset),
                  (11 + offset, 20 + offset), (20 + offset, 22), (22, 29 + offset)]
        self.slices = torch.nn.ModuleList(
            torch.nn.Sequential(*list(features[a:b])) for a, b in bounds
        )

        if not requires_grad:
            _freeze(self)

    def forward(self, x):
        taps = []
        for slice_ in self.slices:
            x = slice_(x)
            taps.append(x)
        out = namedtuple("VggOutputs", self.layer_names)
        return out(*taps)


def build_model(name: str, device):
    """Return ``(model, content_index, style_indices)`` ready for inference."""
    name = name.lower()
    if name == "vgg16":
        model = Vgg16(requires_grad=False)
    elif name == "vgg16-experimental":
        model = Vgg16Experimental(requires_grad=False)
    elif name == "vgg19":
        model = Vgg19(requires_grad=False)
    else:
        raise ValueError(
            f"Unknown model {name!r}. Choose one of {', '.join(SUPPORTED_MODELS)}."
        )

    return (model.to(device).eval(),
            model.content_feature_maps_index,
            model.style_feature_maps_indices)
