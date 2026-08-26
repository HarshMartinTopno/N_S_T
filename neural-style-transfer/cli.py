#!/usr/bin/env python3
"""Run neural style transfer from the command line.

    python cli.py --content data/content-images/golden_gate.jpg \
                  --style   data/style-images/city.jpeg \
                  --height 400 --optimizer lbfgs --iterations 300
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from nst import (
    DEFAULT_ITERATIONS,
    INIT_METHODS,
    OPTIMIZERS,
    SUPPORTED_MODELS,
    TransferConfig,
    resolve_device,
    run_style_transfer,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Neural style transfer (Gatys et al., 2016)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--content", required=True, type=Path, help="content image path")
    p.add_argument("--style", required=True, type=Path, help="style image path")
    p.add_argument("--output-dir", type=Path, default=Path("data/output-images"))

    p.add_argument("--height", type=int, default=400,
                   help="output height in pixels; width follows the aspect ratio")
    p.add_argument("--content-weight", type=float, default=1e5)
    p.add_argument("--style-weight", type=float, default=3e4)
    p.add_argument("--tv-weight", type=float, default=1e0)

    p.add_argument("--model", choices=SUPPORTED_MODELS, default="vgg19")
    p.add_argument("--optimizer", choices=OPTIMIZERS, default="lbfgs")
    p.add_argument("--init", dest="init_method", choices=INIT_METHODS, default="content")
    p.add_argument("--iterations", type=int, default=None,
                   help=f"defaults to {DEFAULT_ITERATIONS}")

    p.add_argument("--saving-freq", type=int, default=-1,
                   help="save an intermediate frame every N iterations (-1 = off)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--quiet", action="store_true", help="only print the final path")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(message)s")

    cfg = TransferConfig(
        content_img=args.content,
        style_img=args.style,
        output_dir=args.output_dir,
        height=args.height,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        model=args.model,
        optimizer=args.optimizer,
        init_method=args.init_method,
        iterations=args.iterations,
        saving_freq=args.saving_freq,
        preview_freq=0,
        seed=args.seed,
    )

    started = time.time()

    def report(p):
        if args.quiet:
            return
        sys.stdout.write(
            f"\r[{p.iteration + 1:>5}/{p.total_iterations}] "
            f"total={p.total_loss:12.2f}  content={p.content_loss:11.2f}  "
            f"style={p.style_loss:11.2f}  tv={p.tv_loss:10.2f}"
        )
        sys.stdout.flush()

    out = run_style_transfer(cfg, on_progress=report,
                             device=resolve_device(args.device))

    if not args.quiet:
        print(f"\nDone in {time.time() - started:.1f}s")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
