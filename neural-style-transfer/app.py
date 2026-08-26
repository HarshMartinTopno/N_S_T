"""FastAPI server: upload two images, watch the optimizer run, download the result.

Style transfer is slow (seconds on a GPU, minutes on a CPU), so requests do not
block. Uploading creates a job, the job runs on a single background worker, and
the browser polls for progress and a live preview.
"""

import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from nst import (
    DEFAULT_ITERATIONS,
    INIT_METHODS,
    OPTIMIZERS,
    SUPPORTED_MODELS,
    TransferConfig,
    TransferCancelled,
    resolve_device,
    run_style_transfer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("nst.app")

# ---------------------------------------------------------------------------
# Limits. Override with environment variables when deploying.
# ---------------------------------------------------------------------------
RUNS_DIR = Path(os.getenv("NST_RUNS_DIR", "runs"))
MAX_UPLOAD_BYTES = int(os.getenv("NST_MAX_UPLOAD_MB", "12")) * 1024 * 1024
MAX_HEIGHT = int(os.getenv("NST_MAX_HEIGHT", "700"))
MAX_ITERATIONS = int(os.getenv("NST_MAX_ITERATIONS", "1000"))
JOB_TTL_SECONDS = int(os.getenv("NST_JOB_TTL_MINUTES", "60")) * 60
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}

RUNS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = resolve_device(os.getenv("NST_DEVICE", "auto"))

# One worker: the model is heavy and parallel jobs would just thrash the CPU.
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nst")


@dataclass
class Job:
    id: str
    status: str = "queued"          # queued | running | done | failed | cancelled
    created_at: float = field(default_factory=time.time)
    iteration: int = 0
    total_iterations: int = 0
    losses: Dict[str, float] = field(default_factory=dict)
    preview_version: int = 0
    result_path: Optional[Path] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    settings: Dict = field(default_factory=dict)

    @property
    def directory(self) -> Path:
        return RUNS_DIR / self.id

    def to_dict(self) -> Dict:
        # L-BFGS line search can evaluate the closure more times than max_iter,
        # so the raw counter is clamped before it reaches the client.
        shown = min(self.iteration, self.total_iterations) if self.total_iterations else 0
        progress = (shown / self.total_iterations) if self.total_iterations else 0.0
        if self.status == "done":
            shown, progress = self.total_iterations, 1.0
        return {
            "id": self.id,
            "status": self.status,
            "iteration": shown,
            "total_iterations": self.total_iterations,
            "progress": round(progress, 4),
            "losses": self.losses,
            "preview_version": self.preview_version,
            "has_result": self.result_path is not None and self.result_path.exists(),
            "error": self.error,
            "settings": self.settings,
            "elapsed": round(time.time() - self.created_at, 1),
        }


JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()

app = FastAPI(title="Neural Style Transfer", version="1.0.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_job(job_id: str) -> Job:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job. It may have expired.")
    return job


def save_upload(upload: UploadFile, destination: Path, label: str) -> Path:
    """Stream an upload to disk, enforcing the size cap and verifying it decodes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("wb") as fh:
        while chunk := upload.file.read(1 << 20):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                fh.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"{label} image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                )
            fh.write(chunk)

    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"{label} image is empty.")

    try:
        with Image.open(destination) as img:
            fmt = img.format
            img.verify()
    except (UnidentifiedImageError, OSError):
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"{label} file is not a readable image.")

    if fmt not in ALLOWED_FORMATS:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"{label} image is {fmt}. Use JPEG, PNG, WEBP or BMP.",
        )
    return destination


def purge_expired_jobs() -> None:
    """Drop finished jobs (and their files) once they pass the TTL."""
    cutoff = time.time() - JOB_TTL_SECONDS
    with JOBS_LOCK:
        stale = [j for j in JOBS.values()
                 if j.created_at < cutoff and j.status in {"done", "failed", "cancelled"}]
        for job in stale:
            JOBS.pop(job.id, None)
    for job in stale:
        shutil.rmtree(job.directory, ignore_errors=True)
    if stale:
        logger.info("Purged %d expired job(s)", len(stale))


def execute_job(job: Job, cfg: TransferConfig) -> None:
    """Worker body. Runs on the executor thread, never in the request path."""
    def on_progress(p):
        job.iteration = p.iteration + 1
        job.total_iterations = p.total_iterations
        job.losses = {
            "total": p.total_loss,
            "content": p.content_loss,
            "style": p.style_loss,
            "tv": p.tv_loss,
        }
        if p.preview_path is not None:
            job.preview_version += 1

    try:
        job.status = "running"
        job.result_path = run_style_transfer(
            cfg, on_progress=on_progress,
            should_stop=lambda: job.cancel_requested,
            device=DEVICE,
        )
        job.status = "done"
    except TransferCancelled:
        job.status = "cancelled"
    except Exception as exc:                       # noqa: BLE001 - surfaced to the client
        logger.exception("Job %s failed", job.id)
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    with JOBS_LOCK:
        active = sum(1 for j in JOBS.values() if j.status in {"queued", "running"})
    return {"status": "ok", "device": str(DEVICE), "active_jobs": active}


@app.get("/api/options")
def options():
    """Everything the frontend needs to build its form without hardcoding."""
    return {
        "models": list(SUPPORTED_MODELS),
        "optimizers": list(OPTIMIZERS),
        "init_methods": list(INIT_METHODS),
        "default_iterations": DEFAULT_ITERATIONS,
        "max_height": MAX_HEIGHT,
        "max_iterations": MAX_ITERATIONS,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "device": str(DEVICE),
    }


@app.post("/api/transfer", status_code=202)
def create_transfer(
    content: UploadFile = File(...),
    style: UploadFile = File(...),
    height: int = Form(400),
    content_weight: float = Form(1e5),
    style_weight: float = Form(3e4),
    tv_weight: float = Form(1e0),
    model: str = Form("vgg19"),
    optimizer: str = Form("lbfgs"),
    init_method: str = Form("content"),
    iterations: Optional[int] = Form(None),
):
    purge_expired_jobs()

    if not 64 <= height <= MAX_HEIGHT:
        raise HTTPException(status_code=400,
                            detail=f"Height must be between 64 and {MAX_HEIGHT} pixels.")
    if iterations is not None and not 1 <= iterations <= MAX_ITERATIONS:
        raise HTTPException(status_code=400,
                            detail=f"Iterations must be between 1 and {MAX_ITERATIONS}.")

    job = Job(id=uuid.uuid4().hex[:12])
    job.directory.mkdir(parents=True, exist_ok=True)

    try:
        content_path = save_upload(content, job.directory / f"content{Path(content.filename or '').suffix or '.jpg'}", "Content")
        style_path = save_upload(style, job.directory / f"style{Path(style.filename or '').suffix or '.jpg'}", "Style")

        cfg = TransferConfig(
            content_img=content_path,
            style_img=style_path,
            output_dir=job.directory,
            height=height,
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight,
            model=model,
            optimizer=optimizer,
            init_method=init_method,
            iterations=iterations,
            preview_freq=5,
        )
    except HTTPException:
        shutil.rmtree(job.directory, ignore_errors=True)
        raise
    except ValueError as exc:
        shutil.rmtree(job.directory, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc))

    job.total_iterations = cfg.iterations
    job.settings = {
        "height": cfg.height, "model": cfg.model, "optimizer": cfg.optimizer,
        "init_method": cfg.init_method, "iterations": cfg.iterations,
        "content_weight": cfg.content_weight, "style_weight": cfg.style_weight,
        "tv_weight": cfg.tv_weight,
    }

    with JOBS_LOCK:
        JOBS[job.id] = job
    EXECUTOR.submit(execute_job, job, cfg)

    return JSONResponse(status_code=202, content=job.to_dict())


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return get_job(job_id).to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = get_job(job_id)
    if job.status in {"queued", "running"}:
        job.cancel_requested = True
    return job.to_dict()


@app.get("/api/jobs/{job_id}/preview")
def job_preview(job_id: str, v: int = 0):    # noqa: ARG001 - cache buster only
    job = get_job(job_id)
    preview = job.directory / "preview.jpg"
    if not preview.exists():
        raise HTTPException(status_code=404, detail="No preview yet.")
    return FileResponse(preview, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = get_job(job_id)
    if job.status != "done" or job.result_path is None or not job.result_path.exists():
        raise HTTPException(status_code=409, detail="Result is not ready.")
    return FileResponse(job.result_path, media_type="image/jpeg",
                        filename=job.result_path.name)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = get_job(job_id)
    job.cancel_requested = True
    with JOBS_LOCK:
        JOBS.pop(job_id, None)
    shutil.rmtree(job.directory, ignore_errors=True)
    return {"deleted": job_id}


# Static frontend last, so it does not shadow the API routes.
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=os.getenv("HOST", "0.0.0.0"),
                port=int(os.getenv("PORT", "8000")), reload=False)
