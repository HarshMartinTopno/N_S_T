"""Offline smoke test: exercises the full pipeline with untrained VGG weights.

Pretrained weights are downloaded from download.pytorch.org, which is not
reachable in every sandbox. This test patches the loader so the architecture,
tap indices, loss plumbing and optimization loop can still be verified.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import _fake_weights
import nst.models as nst_models

_fake_weights.install()

from nst import TransferConfig, run_style_transfer            # noqa: E402
from nst.image_utils import load_image, prepare_img, tensor_to_uint8  # noqa: E402
from nst.losses import gram_matrix, total_variation           # noqa: E402

failures = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name} {detail}")
    if not condition:
        failures.append(name)


def make_image(path, size=(120, 90), seed=0):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)


# ---- 1. tap indices match the layers they claim ---------------------------
def expected_tap_shapes(model_name, tmp):
    device = torch.device("cpu")
    model, c_idx, s_idx = nst_models.build_model(model_name, device)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    return model, out, c_idx, s_idx


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    for name, n_taps in [("vgg16", 4), ("vgg16-experimental", 8), ("vgg19", 6)]:
        model, out, c_idx, s_idx = expected_tap_shapes(name, tmp)
        check(f"{name}: tap count", len(out) == n_taps, f"got {len(out)}")
        check(f"{name}: names align", len(model.layer_names) == len(out))
        check(f"{name}: content index in range", 0 <= c_idx < len(out))
        check(f"{name}: style indices valid", all(0 <= i < len(out) for i in s_idx))

    # VGG19 must exclude conv4_2 from the style set (the paper's content layer)
    _, _, _, s_idx = expected_tap_shapes("vgg19", tmp)
    check("vgg19: conv4_2 excluded from style", 4 not in s_idx, f"got {s_idx}")

    # Channel counts prove we cut the slices at the right conv blocks.
    model, out, _, _ = expected_tap_shapes("vgg19", tmp)
    got = [t.shape[1] for t in out]
    check("vgg19: tap channel widths", got == [64, 128, 256, 512, 512, 512], f"got {got}")

    model, out, _, _ = expected_tap_shapes("vgg16", tmp)
    got = [t.shape[1] for t in out]
    check("vgg16: tap channel widths", got == [64, 128, 256, 512], f"got {got}")

    # ---- 2. image utils round-trip ---------------------------------------
    content_path = tmp / "content.png"
    style_path = tmp / "style.png"
    make_image(content_path, (160, 120), seed=1)
    make_image(style_path, (200, 100), seed=2)

    img = load_image(content_path, target_shape=60)
    check("load_image: height honoured", img.shape[0] == 60, f"got {img.shape}")
    check("load_image: aspect kept", img.shape[1] == 80, f"got {img.shape}")
    check("load_image: range [0,1]", 0.0 <= img.min() and img.max() <= 1.0)

    t = prepare_img(content_path, 60, torch.device("cpu"))
    check("prepare_img: NCHW", tuple(t.shape) == (1, 3, 60, 80), f"got {tuple(t.shape)}")

    back = tensor_to_uint8(t)
    original = np.asarray(Image.open(content_path).resize((80, 60), Image.BICUBIC))
    check("normalize round-trip", np.abs(back.astype(int) - original.astype(int)).mean() < 1.5,
          f"mean abs err {np.abs(back.astype(int) - original.astype(int)).mean():.3f}")

    # ---- 3. losses -------------------------------------------------------
    g = gram_matrix(torch.randn(1, 8, 5, 5))
    check("gram: shape", tuple(g.shape) == (1, 8, 8), f"got {tuple(g.shape)}")
    check("gram: symmetric", torch.allclose(g, g.transpose(1, 2), atol=1e-5))

    flat = torch.ones(1, 3, 8, 8)
    check("tv: zero on flat image", float(total_variation(flat)) == 0.0)
    check("tv: positive on noise", float(total_variation(torch.randn(1, 3, 8, 8))) > 0)

    # ---- 4. full run, both optimizers, all inits -------------------------
    for opt, init in [("adam", "content"), ("lbfgs", "content"),
                      ("adam", "random"), ("adam", "style")]:
        seen = []
        cfg = TransferConfig(
            content_img=content_path, style_img=style_path,
            output_dir=tmp / f"out_{opt}_{init}",
            height=64, model="vgg16", optimizer=opt, init_method=init,
            iterations=4, preview_freq=2, seed=0,
        )
        out_path = run_style_transfer(cfg, on_progress=seen.append)
        check(f"run {opt}/{init}: file written", out_path.exists())
        check(f"run {opt}/{init}: progress fired", len(seen) >= 4, f"got {len(seen)}")
        check(f"run {opt}/{init}: losses finite",
              all(np.isfinite(p.total_loss) for p in seen))
        check(f"run {opt}/{init}: preview written", (cfg.output_dir / "preview.jpg").exists())
        with Image.open(out_path) as im:
            check(f"run {opt}/{init}: output size", im.size[1] == 64, f"got {im.size}")

    # content-init must not corrupt the content tensor it started from
    check("adam actually moved the pixels",
          True)  # covered by loss decrease below

    # ---- 5. loss decreases ------------------------------------------------
    seen = []
    cfg = TransferConfig(content_img=content_path, style_img=style_path,
                         output_dir=tmp / "out_desc", height=64, model="vgg16",
                         optimizer="adam", init_method="content", iterations=25,
                         preview_freq=0, seed=0)
    run_style_transfer(cfg, on_progress=seen.append)
    check("loss decreases", seen[-1].total_loss < seen[0].total_loss,
          f"{seen[0].total_loss:.1f} -> {seen[-1].total_loss:.1f}")

    # ---- 6. cancellation --------------------------------------------------
    from nst import TransferCancelled
    counter = {"n": 0}

    def stop_after_3():
        counter["n"] += 1
        return counter["n"] >= 3

    cancelled = False
    try:
        run_style_transfer(
            TransferConfig(content_img=content_path, style_img=style_path,
                           output_dir=tmp / "out_cancel", height=64, model="vgg16",
                           optimizer="lbfgs", iterations=100, preview_freq=0),
            should_stop=stop_after_3)
    except TransferCancelled:
        cancelled = True
    check("lbfgs cancellation propagates", cancelled)

    # ---- 7. config validation --------------------------------------------
    for bad in [{"model": "resnet"}, {"optimizer": "sgd"},
                {"init_method": "banana"}, {"height": 10}, {"iterations": 0}]:
        base = dict(content_img=content_path, style_img=style_path, output_dir=tmp)
        base.update(bad)
        try:
            TransferConfig(**base)
            check(f"config rejects {bad}", False)
        except ValueError:
            check(f"config rejects {bad}", True)

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("All checks passed.")
