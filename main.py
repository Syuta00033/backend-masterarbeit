import os
import subprocess
import tempfile
import uuid
import uvicorn
import threading
import imageio_ffmpeg
import shutil
import json
import mimetypes
from datetime import datetime

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

import cv2
import numpy as np
import mediapipe as mp

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision

from kick_analyzer import KickAnalyzer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATHS = {
    "lite":  "models/pose_landmarker_lite.task",
    "full":  "models/pose_landmarker_full.task",
    "heavy": "models/pose_landmarker_heavy.task",
}

TEMP_DIR = tempfile.gettempdir()
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.join(BACKEND_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Persistenter Speicher für gespeicherte Kicks (Video + Bewertung)
SAVED_DIR = os.path.join(BACKEND_DIR, "saved")
os.makedirs(SAVED_DIR, exist_ok=True)

jobs: dict[str, dict] = {}

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_POSE_LANDMARK_STYLE = drawing_styles.get_default_pose_landmarks_style()
_POSE_CONNECTION_STYLE = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    for pose_landmarks in detection_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=_POSE_LANDMARK_STYLE,
            connection_drawing_spec=_POSE_CONNECTION_STYLE,
        )
    return annotated_image

def build_summary(criteria: list[dict], velocity: float) -> dict:
    if criteria:
        total = 0
        for c in criteria:
            total += c.get("score", 0)
        overall = round(total / len(criteria))
    else:
        overall = 0

    # clamp score stars to 1-5
    stars = round (overall/20)
    if stars < 1:
        stars = 1
    if stars > 5:
        stars = 5

    # check failed criteria to show the tips
    failed = []
    for c in criteria:
        if not c.get("passed", False):
            failed.append(c)
    
    tips = []
    for f in failed:
        tips.append(f.get("feedback", ""))

    return {"overall": overall, "stars": stars, "criteria": criteria, "tips": tips, "velocity": velocity}

def process_video(job_id: str, input_path: str, output_path: str, model_path: str, kick_type: str):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "Konnte Video nicht öffnen."
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)

        jobs[job_id]["total"] = total_frames

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        kick_analyzer = KickAnalyzer(kick_type=kick_type)
        kick_analyzer.kick.fps = fps
        frame_index = 0

        with PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                success, bgr_frame = cap.read()
                if not success:
                    break

                bgr_frame = cv2.resize(bgr_frame, (out_w, out_h))
                timestamp_ms = int((frame_index / fps) * 1000)
                frame_index += 1

                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                annotated = draw_landmarks_on_image(rgb_frame, result)
                annotated = kick_analyzer.process_frame(result, annotated, frame_index)

                out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                jobs[job_id]["progress"] = frame_index

        cap.release()
        out.release()

        # Bewertung
        criteria = kick_analyzer.kick.evaluate()
        jobs[job_id]["analysis"] = build_summary(criteria, kick_analyzer.kick.max_foot_velocity)

        # H264 encoding for better compatibility
        h264_path = output_path.replace(".mp4", "_h264.mp4")
        r = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", output_path,
             "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", h264_path],
            capture_output=True,
        )
        if r.returncode == 0 and os.path.exists(h264_path):
            os.replace(h264_path, output_path)

        jobs[job_id]["status"] = "done"

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    tmp_path = os.path.join(TEMP_DIR, f"{job_id}_input{suffix}")
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    jobs[job_id] = {
        "status": "uploaded",
        "input": tmp_path,
        "output": None,
        "analysis": None,
        "progress": 0,
        "total": 0,
        "error": None,
    }
    return JSONResponse({"job_id": job_id})


@app.post("/process/{job_id}")
async def start_processing(job_id: str, model: str = "lite", kick_type: str = "ap_chagi"):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job nicht gefunden"}, status_code=404)
    if job["status"] != "uploaded":
        return JSONResponse({"error": "Job läuft bereits oder ist fertig"}, status_code=400)
    if model not in MODEL_PATHS:
        return JSONResponse({"error": f"Unbekanntes Modell: {model}"}, status_code=400)

    output_path = os.path.join(TEMP_DIR, f"{job_id}_output.mp4")
    jobs[job_id]["output"] = output_path
    jobs[job_id]["kick_type"] = kick_type
    jobs[job_id]["status"] = "processing"

    threading.Thread(
        target=process_video,
        args=(job_id, job["input"], output_path, MODEL_PATHS[model], kick_type),
        daemon=True,
    ).start()

    return JSONResponse({"status": "processing"})


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Nicht gefunden"}, status_code=404)
    return JSONResponse({
        "status": job["status"],
        "progress": job["progress"],
        "total": job["total"],
        "error": job["error"],
    })


@app.get("/download/{job_id}")
def download_video(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return JSONResponse({"error": "Video noch nicht bereit"}, status_code=404)
    output_path = job["output"]
    if not output_path or not os.path.exists(output_path):
        return JSONResponse({"error": "Ausgabedatei nicht gefunden"}, status_code=404)
    return FileResponse(output_path, media_type="video/mp4", filename="pose_analysis.mp4")

@app.get("/analysis/{job_id}")
def get_analysis(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return JSONResponse({"error": "Video noch nicht bereit"}, status_code=404)
    
    analysis_text = JSONResponse({"analysis": job["analysis"]})

    return analysis_text


@app.post("/save/{job_id}")
def save_kick(job_id: str, client: str = ""):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return JSONResponse({"error": "Kick noch nicht fertig analysiert"}, status_code=404)

    output_path = job["output"]
    if not output_path or not os.path.exists(output_path):
        return JSONResponse({"error": "Ausgabevideo nicht gefunden"}, status_code=404)

    save_id = str(uuid.uuid4())
    shutil.copy(output_path, os.path.join(SAVED_DIR, f"{save_id}.mp4"))

    meta = {
        "id": save_id,
        "client": client,
        "date": datetime.now().isoformat(timespec="seconds"),
        "kick_type": job.get("kick_type", "ap_chagi"),
        "analysis": job["analysis"],
    }
    with open(os.path.join(SAVED_DIR, f"{save_id}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return JSONResponse({"id": save_id})


@app.get("/saved")
def list_saved(client: str = ""):
    items = []
    for name in os.listdir(SAVED_DIR):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(SAVED_DIR, name), encoding="utf-8") as f:
            meta = json.load(f)
        # nur die eigenen Kicks des Geräts anzeigen
        if meta.get("client", "") != client:
            continue
        analysis = meta.get("analysis") or {}
        items.append({
            "id": meta["id"],
            "date": meta["date"],
            "kick_type": meta.get("kick_type"),
            "overall": analysis.get("overall"),
            "stars": analysis.get("stars"),
        })
    items.sort(key=lambda x: x["date"], reverse=True)
    return JSONResponse({"items": items})


@app.get("/saved/{save_id}")
def get_saved(save_id: str):
    save_id = os.path.basename(save_id)
    path = os.path.join(SAVED_DIR, f"{save_id}.json")
    if not os.path.exists(path):
        return JSONResponse({"error": "Nicht gefunden"}, status_code=404)
    with open(path, encoding="utf-8") as f:
        return JSONResponse(json.load(f))


@app.get("/saved/{save_id}/video")
def get_saved_video(save_id: str):
    save_id = os.path.basename(save_id)
    path = os.path.join(SAVED_DIR, f"{save_id}.mp4")
    if not os.path.exists(path):
        return JSONResponse({"error": "Video nicht gefunden"}, status_code=404)
    return FileResponse(path, media_type="video/mp4", filename="kick.mp4")


@app.delete("/saved/{save_id}")
def delete_saved(save_id: str):
    save_id = os.path.basename(save_id)
    removed = False
    for ext in (".mp4", ".json"):
        p = os.path.join(SAVED_DIR, f"{save_id}{ext}")
        if os.path.exists(p):
            os.remove(p)
            removed = True
    if not removed:
        return JSONResponse({"error": "Nicht gefunden"}, status_code=404)
    return JSONResponse({"status": "deleted"})


STATIC_DIR = os.path.join(BACKEND_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(__file__)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=os.path.join(base_dir, "key.pem"),
        ssl_certfile=os.path.join(base_dir, "cert.pem"),
    )
