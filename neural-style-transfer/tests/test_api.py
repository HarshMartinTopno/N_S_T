"""API integration test. Patches the pretrained loader, then drives the real app."""

import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
import _fake_weights

_fake_weights.install()

from fastapi.testclient import TestClient  # noqa: E402
import app as server                        # noqa: E402

client = TestClient(server.app)
failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")
    if not cond:
        failures.append(name)


def png_bytes(w=120, h=90, seed=0):
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


# ---- static + meta --------------------------------------------------------
r = client.get("/")
check("GET / serves the page", r.status_code == 200 and "<title>" in r.text)
check("GET /styles.css", client.get("/styles.css").status_code == 200)
check("GET /app.js", client.get("/app.js").status_code == 200)

r = client.get("/api/health")
check("health ok", r.status_code == 200 and r.json()["status"] == "ok")

r = client.get("/api/options")
opts = r.json()
check("options lists models", "vgg19" in opts["models"])
check("options reports device", "device" in opts)

# ---- validation -----------------------------------------------------------
files = {"content": ("c.png", png_bytes(seed=1), "image/png"),
         "style": ("s.png", png_bytes(seed=2), "image/png")}

r = client.post("/api/transfer", files=dict(files), data={"height": "10"})
check("rejects tiny height", r.status_code == 400, r.text[:80])

r = client.post("/api/transfer",
                files={"content": ("c.txt", b"not an image", "text/plain"),
                       "style": ("s.png", png_bytes(seed=2), "image/png")})
check("rejects non-image upload", r.status_code == 400, r.text[:80])

r = client.post("/api/transfer", files=dict(files), data={"model": "resnet"})
check("rejects unknown model", r.status_code == 400, r.text[:80])

r = client.post("/api/transfer", files=dict(files), data={"iterations": "999999"})
check("rejects excessive iterations", r.status_code == 400, r.text[:80])

check("404 on unknown job", client.get("/api/jobs/deadbeef").status_code == 404)

# ---- happy path -----------------------------------------------------------
r = client.post("/api/transfer",
                files={"content": ("c.png", png_bytes(seed=1), "image/png"),
                       "style": ("s.png", png_bytes(seed=2), "image/png")},
                data={"height": "64", "model": "vgg16", "optimizer": "adam",
                      "iterations": "6", "init_method": "content"})
check("accepted with 202", r.status_code == 202, r.text[:120])
job = r.json()
job_id = job["id"]
check("job starts queued/running", job["status"] in {"queued", "running"})

final = None
for _ in range(200):
    time.sleep(0.25)
    final = client.get(f"/api/jobs/{job_id}").json()
    if final["status"] in {"done", "failed", "cancelled"}:
        break

check("job finished", final["status"] == "done", final.get("error") or final["status"])
check("progress reached 1.0", final["progress"] == 1.0, str(final["progress"]))
check("losses reported", set(final["losses"]) == {"total", "content", "style", "tv"})
check("result flagged", final["has_result"] is True)

r = client.get(f"/api/jobs/{job_id}/result")
check("result downloads", r.status_code == 200 and r.headers["content-type"] == "image/jpeg")
with Image.open(io.BytesIO(r.content)) as im:
    check("result height matches request", im.size[1] == 64, str(im.size))

r = client.get(f"/api/jobs/{job_id}/preview?v=1")
check("preview downloads", r.status_code == 200)

# ---- cancel ---------------------------------------------------------------
r = client.post("/api/transfer",
                files={"content": ("c.png", png_bytes(seed=3), "image/png"),
                       "style": ("s.png", png_bytes(seed=4), "image/png")},
                data={"height": "64", "model": "vgg16", "optimizer": "adam",
                      "iterations": "400"})
cancel_id = r.json()["id"]
time.sleep(1.0)
client.post(f"/api/jobs/{cancel_id}/cancel")
for _ in range(120):
    time.sleep(0.25)
    st = client.get(f"/api/jobs/{cancel_id}").json()
    if st["status"] in {"done", "failed", "cancelled"}:
        break
check("cancel stops the run", st["status"] == "cancelled", st["status"])
check("cancelled run has no result", st["has_result"] is False)

# ---- delete ---------------------------------------------------------------
r = client.delete(f"/api/jobs/{job_id}")
check("delete removes job", r.status_code == 200)
check("deleted job is gone", client.get(f"/api/jobs/{job_id}").status_code == 404)
check("job files removed", not (server.RUNS_DIR / job_id).exists())

client.delete(f"/api/jobs/{cancel_id}")

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("All API checks passed.")
