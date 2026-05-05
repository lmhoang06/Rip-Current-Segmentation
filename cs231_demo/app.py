"""
FastAPI app: dual CBAM YOLO-seg streams (two MP4 outputs per upload).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from tqdm import tqdm
from ultralytics import YOLO

from cs231_demo.cbam_runtime import reload_and_inject_cbam_runtime
from cs231_demo.overlay import draw_polygons_overlay, iter_result_polygons

# --- Constants (edit paths to your checkpoints; resolved relative to repo root) ---
_REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_A_NAME = "model_weights/cs231_yolo26n_baseline.pt"
MODEL_B_NAME = "model_weights/cs231_yolo26n_cbam_p3p4.pt"
IMGSZ = 640
CONF = 0.25
IOU = 0.7
DEVICE = ""  # e.g. "0" or "cpu"; empty = Ultralytics default
ALPHA = 0.35
COLOR_A_BGR = (255, 0, 255)  # magenta
COLOR_B_BGR = (0, 255, 0)  # green
MAX_FRAMES = 0  # 0 = all frames

_DEMO_DIR = Path(__file__).resolve().parent
_OUTPUTS_DIR = _DEMO_DIR / "_outputs"

model_a: YOLO | None = None
model_b: YOLO | None = None

jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_render_lock = threading.Lock()


def _predict_kwargs() -> dict:
    kw: dict = {"imgsz": IMGSZ, "conf": CONF, "iou": IOU, "verbose": False}
    if str(DEVICE).strip():
        kw["device"] = str(DEVICE).strip()
    return kw


def _render_job(job_id: str, upload_path: Path, job_dir: Path) -> None:
    global jobs
    cap = None
    wa = wb = None
    try:
        with _render_lock:
            with _jobs_lock:
                jobs[job_id]["status"] = "running"
            cap = cv2.VideoCapture(str(upload_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {upload_path}")

            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w <= 0 or h <= 0:
                raise RuntimeError("Invalid video dimensions.")

            count_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_cap = count_reported if count_reported > 0 else None
            if MAX_FRAMES > 0 and total_cap is not None:
                total_cap = min(total_cap, MAX_FRAMES)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_a = job_dir / "model_a.mp4"
            out_b = job_dir / "model_b.mp4"
            wa = cv2.VideoWriter(str(out_a), fourcc, fps, (w, h))
            wb = cv2.VideoWriter(str(out_b), fourcc, fps, (w, h))
            if not wa.isOpened() or not wb.isOpened():
                raise RuntimeError("Could not open VideoWriter for MP4 output.")

            pk = _predict_kwargs()
            ma, mb = model_a, model_b
            if ma is None or mb is None:
                raise RuntimeError("Models not loaded.")

            outer_total = total_cap
            with tqdm(total=outer_total, desc="render", unit="fr") as pbar:
                idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if MAX_FRAMES > 0 and idx >= MAX_FRAMES:
                        break

                    ra = ma.predict(source=frame, **pk)
                    r0a = ra[0] if ra else None
                    vis_a = draw_polygons_overlay(
                        frame, iter_result_polygons(r0a), alpha=ALPHA, color_bgr=COLOR_A_BGR
                    )
                    wr_a = wa.write(vis_a)
                    if wr_a is False:
                        raise RuntimeError("Failed to write frame to model A video.")

                    rb = mb.predict(source=frame, **pk)
                    r0b = rb[0] if rb else None
                    vis_b = draw_polygons_overlay(
                        frame, iter_result_polygons(r0b), alpha=ALPHA, color_bgr=COLOR_B_BGR
                    )
                    wr_b = wb.write(vis_b)
                    if wr_b is False:
                        raise RuntimeError("Failed to write frame to model B video.")

                    idx += 1
                    pbar.update(1)
                    with _jobs_lock:
                        jobs[job_id]["progress"] = {
                            "current": idx,
                            "total": outer_total if outer_total is not None else idx,
                        }

            if idx == 0:
                raise RuntimeError("No frames decoded from video.")

            # IMPORTANT: finalize MP4s before any post-processing/transcode.
            try:
                wa.release()
            except Exception:
                pass
            try:
                wb.release()
            except Exception:
                pass
            wa = None
            wb = None

            # Transcode to browser-friendly H.264 MP4 (Chrome often rejects mp4v streams).
            out_a_h264 = job_dir / "model_a_h264.mp4"
            out_b_h264 = job_dir / "model_b_h264.mp4"
            for src, dst in [(out_a, out_a_h264), (out_b, out_b_h264)]:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(src),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(dst),
                ]
                p = subprocess.run(cmd, capture_output=True, text=True)
                if p.returncode != 0:
                    raise RuntimeError(f"ffmpeg failed for {src.name}: {(p.stderr or '').strip()}")

            with _jobs_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["progress"] = {"current": idx, "total": idx}
                jobs[job_id]["urls"] = {
                    "a": f"/outputs/{job_id}/{out_a_h264.name if out_a_h264.exists() else out_a.name}",
                    "b": f"/outputs/{job_id}/{out_b_h264.name if out_b_h264.exists() else out_b.name}",
                }
    except Exception as e:
        with _jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
    finally:
        if wa is not None:
            wa.release()
        if wb is not None:
            wb.release()
        if cap is not None:
            cap.release()


def _start_render_worker(job_id: str, upload_path: Path, job_dir: Path) -> None:
    t = threading.Thread(target=_render_job, args=(job_id, upload_path, job_dir), daemon=True)
    t.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_a, model_b
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    reload_and_inject_cbam_runtime()
    path_a = Path(MODEL_A_NAME)
    path_b = Path(MODEL_B_NAME)
    if not path_a.is_absolute():
        path_a = _REPO_ROOT / path_a
    if not path_b.is_absolute():
        path_b = _REPO_ROOT / path_b
    model_a = YOLO(str(path_a))
    model_b = YOLO(str(path_b))
    try:
        model_a.fuse()
    except Exception:
        pass
    try:
        model_b.fuse()
    except Exception:
        pass
    yield


app = FastAPI(title="CS231 CBAM YOLO dual-stream demo", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=str(_OUTPUTS_DIR)), name="outputs")


@app.get("/")
async def index_page():
    return FileResponse(_DEMO_DIR / "static" / "index.html")


@app.post("/api/render")
async def api_render(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    job_id = uuid.uuid4().hex
    job_dir = _OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(video.filename).suffix or ".mp4"
    upload_path = job_dir / f"upload{suffix}"

    with _jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "progress": None,
            "error": None,
            "urls": None,
        }

    try:
        with upload_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)
    except Exception as e:
        with _jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = f"Upload failed: {e}"
        raise HTTPException(status_code=500, detail=str(e)) from e

    _start_render_worker(job_id, upload_path, job_dir)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def api_job_status(job_id: str):
    with _jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")

    body: dict = {"status": job["status"]}
    if job.get("error"):
        body["error"] = job["error"]
    if job.get("progress") is not None:
        body["progress"] = job["progress"]
    if job.get("urls"):
        body["urls"] = job["urls"]
    return body
