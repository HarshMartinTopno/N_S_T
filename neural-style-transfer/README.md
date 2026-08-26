# Neural Style Transfer

A PyTorch implementation of [*Image Style Transfer Using Convolutional Neural Networks*](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) (Gatys, Ecker & Bethge, CVPR 2016), with a web frontend, a CLI, and a Docker image.

Upload a photo and a painting. The server optimizes a third image until it has the photo's layout and the painting's texture, streaming a live preview back to the browser while it works.

```
                 ┌──────────────┐
  content.jpg ──▶│              │
                 │  frozen VGG  │──▶  feature maps  ──▶  loss  ──┐
  style.jpg   ──▶│              │                                │
                 └──────────────┘                                │
                        ▲                                        │
                        │                                        ▼
                  generated.jpg ◀───────── gradient descent on the PIXELS
```

---

## Table of contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Web app](#web-app)
- [Command line](#command-line)
- [How it works](#how-it-works)
- [Tuning the parameters](#tuning-the-parameters)
- [HTTP API](#http-api)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Tests](#tests)
- [Performance notes](#performance-notes)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## What it does

The network is never trained. A pretrained VGG is frozen and used only as a
feature extractor; the thing being optimized is the output image itself. Each
step asks: *do this image's deep features match the content photo, and do its
feature correlations match the style painting?* Gradient descent then edits the
pixels to close both gaps at once.

Three encoders are available:

| Encoder | Style taps | Content tap | Notes |
|---|---|---|---|
| `vgg19` | `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`, `relu5_1` | `conv4_2` | The paper's configuration. Default. |
| `vgg16` | `relu1_2`, `relu2_2`, `relu3_3`, `relu4_3` | `relu2_2` | Lighter and faster; tighter to the content. |
| `vgg16-experimental` | eight taps across all five blocks | `relu3_2` | For experimenting with layer choice. |

---

## Quick start

Requires Python 3.9+.

```bash
git clone <your-repo-url>
cd neural-style-transfer

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python app.py
```

Open <http://localhost:8000>.

On first run torchvision downloads the VGG weights (~550 MB for VGG19). That
happens once and is cached in `~/.cache/torch`.

**CPU-only machine?** Install the smaller CPU build of PyTorch first — it saves
several gigabytes:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

---

## Web app

The page has two halves.

**Left — inputs.** Drop or click to add a content image and a style image, then
set height, the three loss weights, encoder, optimizer, starting point, and
iteration count. Every control has a sane default; you can just press
**Run transfer**.

**Right — output.** A preview is written every 5 iterations and swapped into the
stage, so you watch the image resolve rather than staring at a spinner. Below it
is the loss readout: the progress bar, a three-segment budget bar showing which
loss term currently dominates the total, and the live numbers for content, style,
tv and total. **Stop** aborts a run and keeps the last preview.

Jobs run one at a time on a background worker, so the HTTP request never blocks
and a slow run cannot tie up the server. Job files are deleted an hour after they
finish.

---

## Command line

```bash
python cli.py --content data/content-images/golden_gate.jpg \
              --style   data/style-images/city.jpeg \
              --height 400 \
              --optimizer lbfgs \
              --iterations 300
```

| Flag | Default | Meaning |
|---|---|---|
| `--content`, `--style` | *required* | Input image paths |
| `--output-dir` | `data/output-images` | Where the result is written |
| `--height` | `400` | Output height in px; width follows the aspect ratio |
| `--content-weight` | `1e5` | How hard to preserve the photo's structure |
| `--style-weight` | `3e4` | How hard to apply the painting's texture |
| `--tv-weight` | `1e0` | Smoothing; suppresses speckle noise |
| `--model` | `vgg19` | `vgg19`, `vgg16`, `vgg16-experimental` |
| `--optimizer` | `lbfgs` | `lbfgs` or `adam` |
| `--init` | `content` | `content`, `random`, or `style` |
| `--iterations` | 300 (L-BFGS) / 1500 (Adam) | Optimization steps |
| `--saving-freq` | `-1` | Save a frame every N steps (`-1` = final only) |
| `--seed` | `None` | Makes `random` init reproducible |
| `--device` | `auto` | `auto`, `cuda`, `mps`, `cpu` |

Saving intermediate frames gives you an animation of the optimization:

```bash
python cli.py --content c.jpg --style s.jpg --saving-freq 5
ffmpeg -framerate 30 -pattern_type glob -i 'data/output-images/step_*.jpg' out.mp4
```

You can also drive it from Python:

```python
from nst import TransferConfig, run_style_transfer

cfg = TransferConfig(
    content_img="photo.jpg",
    style_img="painting.jpg",
    output_dir="out",
    height=512,
    style_weight=1e5,
)
path = run_style_transfer(cfg, on_progress=lambda p: print(p.iteration, p.total_loss))
```

---

## How it works

### 1. Preprocessing

Images are read as RGB, resized to the requested height, scaled to `[0, 255]`,
and the ImageNet channel mean is subtracted. The standard deviation is left at
`1`, which keeps loss values in the same units the paper reported — that is why
the published weight ratios transfer directly to this code.

### 2. Content loss

Deep VGG activations discard texture and keep layout. Matching them at a single
mid-to-deep layer is what "same scene" means:

```
L_content = MSE( F_generated[content_layer], F_content[content_layer] )
```

### 3. Style loss

Style is texture without position. The Gram matrix of a layer's feature map
collapses the spatial axes and keeps only *which channels fire together*:

```
G[i,j] = Σ_positions  F[i, p] · F[j, p]        (normalized by C·H·W)

L_style = mean over style layers of  Σ (G_generated − G_style)²
```

Summing Gram differences across shallow **and** deep layers is what captures both
fine brush strokes and large compositional motifs.

### 4. Total variation loss

Pixel-space optimization produces high-frequency speckle. TV loss penalizes the
sum of absolute differences between neighbouring pixels, trading a little detail
for a cleaner image.

### 5. The objective

```
L_total = content_weight · L_content
        + style_weight   · L_style
        + tv_weight      · L_tv
```

Only `optimizing_img` is a leaf with `requires_grad=True`. Every VGG parameter is
frozen at construction, so `L_total.backward()` produces gradients with respect
to the pixels and nothing else.

### 6. Optimizers

| Optimizer | Typical iterations | Behaviour |
|---|---|---|
| **L-BFGS** | 300 | Second-order with a `strong_wolfe` line search. Converges in far fewer steps; the paper's choice. Each step is more expensive and may evaluate the loss several times. |
| **Adam** | 1500 | First-order, `lr=1e1`. Slower to converge but very stable and easier to interrupt mid-run. |

### 7. Initialization

| Method | Behaviour |
|---|---|
| `content` | Starts from the photo. Fastest convergence, best structure retention. **Default.** |
| `random` | Gaussian noise. Most stylistic freedom, needs many more iterations. |
| `style` | The style image resized to the content's dimensions. A middle ground. |

---

## Tuning the parameters

What actually matters is the **ratio** `content_weight : style_weight`, not the
absolute numbers.

| Symptom | Fix |
|---|---|
| Output looks like the original photo with a colour cast | Raise `style_weight` (try 10×) |
| Subject is unrecognizable, pure texture | Raise `content_weight` |
| Speckled, noisy, "digital" grain | Raise `tv_weight` to 5–50 |
| Flat and blurry, detail gone | Lower `tv_weight` toward 0.1 |
| Style is applied but at the wrong scale | Resize the style image so its brush strokes are the size you want relative to a 400 px output |
| Converges then degrades | Reduce iterations, or switch L-BFGS → Adam |

Defaults (`content 1e5`, `style 3e4`, `tv 1e0`) keep the subject clearly
recognizable while applying strong stylization. Start there and change one weight
at a time.

---

## HTTP API

Base URL is the server root. All responses are JSON unless noted.

### `GET /api/health`
Liveness check.
```json
{ "status": "ok", "device": "cuda", "active_jobs": 1 }
```

### `GET /api/options`
Everything the frontend needs to build its form — model names, optimizers, init
methods, default iteration counts, and the server's limits.

### `POST /api/transfer`
`multipart/form-data`. Creates a job and returns **202 Accepted** immediately.

| Field | Type | Default |
|---|---|---|
| `content` | file | required |
| `style` | file | required |
| `height` | int | 400 |
| `content_weight` | float | 1e5 |
| `style_weight` | float | 3e4 |
| `tv_weight` | float | 1.0 |
| `model` | string | `vgg19` |
| `optimizer` | string | `lbfgs` |
| `init_method` | string | `content` |
| `iterations` | int | optimizer default |

```bash
curl -X POST http://localhost:8000/api/transfer \
  -F content=@photo.jpg -F style=@painting.jpg \
  -F height=400 -F iterations=200
```

Errors: `400` bad parameters or unreadable image, `413` upload too large.

### `GET /api/jobs/{id}`
Current state. Poll this.
```json
{
  "id": "9f2c1a4b7e80",
  "status": "running",
  "iteration": 128,
  "total_iterations": 300,
  "progress": 0.4267,
  "losses": { "total": 812345.5, "content": 210043.1, "style": 588102.4, "tv": 14200.0 },
  "preview_version": 26,
  "has_result": false,
  "error": null,
  "elapsed": 41.2
}
```
`status` is one of `queued`, `running`, `done`, `failed`, `cancelled`.

### `GET /api/jobs/{id}/preview?v={preview_version}`
The latest in-progress frame as JPEG. Pass `preview_version` as a cache buster.
`404` until the first preview is written.

### `GET /api/jobs/{id}/result`
The final image as a JPEG attachment. `409` until `status` is `done`.

### `POST /api/jobs/{id}/cancel`
Asks the running job to stop at its next iteration.

### `DELETE /api/jobs/{id}`
Cancels the job if running and deletes its files.

Interactive docs are generated by FastAPI at `/docs`.

---

## Project structure

```
neural-style-transfer/
│
├── nst/                     # the library — no web, no CLI, importable anywhere
│   ├── __init__.py          # public API
│   ├── config.py            # TransferConfig dataclass + validation
│   ├── models.py            # Vgg16 / Vgg16Experimental / Vgg19 feature extractors
│   ├── image_utils.py       # load, normalize, denormalize, save
│   ├── losses.py            # gram_matrix, total_variation, build_loss
│   └── engine.py            # the optimization loop, progress + cancellation
│
├── app.py                   # FastAPI server: job queue, uploads, polling API
├── cli.py                   # command line entry point
│
├── static/                  # the frontend, no build step, no framework
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── tests/
│   ├── test_pipeline.py     # models, losses, image I/O, full runs, cancellation
│   ├── test_api.py          # every HTTP endpoint against the real app
│   ├── _fake_weights.py     # lets tests run offline
│   └── run_all.sh
│
├── data/                    # sample inputs and CLI outputs
│   ├── content-images/
│   ├── style-images/
│   └── output-images/
│
├── runs/                    # per-job working directories (created at runtime)
├── requirements.txt
├── Dockerfile
└── README.md
```

The split is deliberate: `nst/` knows nothing about HTTP, so the same code backs
the CLI, the server, and any notebook you import it into.

---

## Configuration

The server reads these environment variables at startup:

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Listen port |
| `HOST` | `0.0.0.0` | Bind address |
| `NST_DEVICE` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `NST_RUNS_DIR` | `runs` | Where job directories are written |
| `NST_MAX_UPLOAD_MB` | `12` | Per-file upload cap |
| `NST_MAX_HEIGHT` | `700` | Largest output height a client may request |
| `NST_MAX_ITERATIONS` | `1000` | Largest iteration count a client may request |
| `NST_JOB_TTL_MINUTES` | `60` | How long finished jobs and their files are kept |

On a small CPU host, lower the ceilings so one user cannot occupy the worker for
half an hour:

```bash
NST_MAX_HEIGHT=384 NST_MAX_ITERATIONS=250 python app.py
```

---

## Deployment

### Docker

```bash
docker build -t nst .
docker run -p 8000:8000 nst
```

The image installs the CPU build of PyTorch (~1 GB instead of ~6 GB) and bakes
the VGG19 weights in at build time, so the first request does not stall on a
download.

With a GPU host:

```bash
docker run --gpus all -p 8000:8000 -e NST_DEVICE=cuda nst
```

### Platform notes

- **Fly.io / Railway / Render.** Works as-is. Give the machine at least **2 GB
  RAM** — VGG19 plus a 400 px image needs it — and raise the request timeout,
  though only the polling endpoints are hit during a run, so the long job never
  sits inside one request.
- **Free CPU tiers.** Usable, but a 400 px / 300-iteration L-BFGS run takes
  several minutes. Set `NST_MAX_HEIGHT=320` and `NST_MAX_ITERATIONS=200`, and
  tell users on the page what to expect. The device badge in the header already
  says `cpu — expect minutes`.
- **Anything serious.** Put a GPU behind it. The same run drops to a few seconds.
- **Behind a reverse proxy.** Only `/api/transfer` sends a large body; set your
  proxy's `client_max_body_size` to at least `NST_MAX_UPLOAD_MB`.

### Scaling past one box

The job registry is an in-memory dict and the worker pool is a single thread, so
this runs correctly on **one process**. Do not start multiple uvicorn workers —
a poll would hit a process that has never heard of the job. To scale out, move
the registry to Redis and the worker to a task queue (Celery, RQ, or arq); the
`nst` package needs no changes for that.

---

## Tests

```bash
bash tests/run_all.sh
```

77 checks covering:

- tap indices and channel widths for all three encoders (this is where a
  refactor breaks first — a wrong slice boundary silently taps the wrong layer)
- VGG19 correctly excluding `conv4_2` from the style set
- image normalize → denormalize round-trip accuracy
- Gram matrix shape and symmetry; TV loss zero on a flat image
- full runs across both optimizers and all three init methods
- loss actually decreasing
- L-BFGS cancellation propagating out of the closure
- config validation rejecting every bad enum and out-of-range value
- every HTTP endpoint: happy path, validation errors, cancel, delete, 404s

Tests patch in randomly-initialized VGG weights so they run offline in seconds —
the plumbing under test does not depend on the learned weights. To run against
the real ImageNet weights:

```bash
NST_TEST_PRETRAINED=1 bash tests/run_all.sh
```

---

## Performance notes

Rough timings for a 400 px output, VGG19, 300 L-BFGS iterations:

| Device | Time |
|---|---|
| Modern NVIDIA GPU | 10–25 s |
| Apple Silicon (MPS) | 1–3 min |
| Laptop CPU | 5–15 min |

Cost scales with the **pixel count**, so doubling the height roughly quadruples
the time. If you are iterating on weights, tune at height 256 with Adam, then do
the final render at full height with L-BFGS.

---

## Troubleshooting

**`CUDA out of memory`** — lower `--height`, or set `NST_DEVICE=cpu`. Memory
scales with the pixel count, not the iteration count.

**First run hangs for minutes** — torchvision is downloading VGG weights. It
caches to `~/.cache/torch` and only happens once. Pre-warm it with:
```bash
python -c "from torchvision.models import vgg19, VGG19_Weights; vgg19(weights=VGG19_Weights.IMAGENET1K_V1)"
```

**Output is just the content photo** — `style_weight` is too low relative to
`content_weight`, or there were too few iterations. Try 10× the style weight.

**Output is noisy static** — `style_weight` far too high, or `random` init with
too few iterations. Switch to `content` init.

**`Result is not ready` (409)** — you requested the result before `status`
became `done`. Poll `/api/jobs/{id}` first.

**Uploads rejected as unreadable** — HEIC from an iPhone is not supported.
Convert to JPEG, or add `pillow-heif` to the requirements and register it.

**Everything is slow and the badge says `cpu`** — that is expected. See
[Performance notes](#performance-notes).

---

## References

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). [*Image Style Transfer Using Convolutional Neural Networks*](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf). CVPR 2016.
- Simonyan, K., & Zisserman, A. (2015). [*Very Deep Convolutional Networks for Large-Scale Image Recognition*](https://arxiv.org/abs/1409.1556). ICLR 2015.
- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). [*Perceptual Losses for Real-Time Style Transfer*](https://arxiv.org/abs/1603.08155). ECCV 2016 — the feed-forward approach, if you want this to run in milliseconds instead of minutes.
- Implementation insights from [Aleksa Gordić's NST series](https://www.youtube.com/playlist?list=PLBoQnSflObcmbfshq9oNs41vODgXG-608).

## License

MIT.
