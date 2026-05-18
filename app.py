import os
import uuid
import sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
import threading

# Add scripts directory to path to import process_video
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts")))
try:
    from process_video_tracking import process_video
except ImportError as e:
    print("Warning: Could not import process_video_tracking from scripts directory", e)

app = FastAPI()

templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mount img directory for logo and static assets
img_dir = Path("img")
img_dir.mkdir(parents=True, exist_ok=True)
app.mount("/img", StaticFiles(directory="img"), name="img")

# In-memory task tracking
# { "task_id": {"status": "processing"|"done"|"error", "progress": 0.0, "csv_path": "path", "error": "msg"} }
tasks = {}

def process_video_task(task_id: str, video_path: Path, frame_stride: int = 50):
    tasks[task_id] = {"status": "processing", "progress": 0.0, "csv_path": None, "error": None}

    def progress_callback(pct: float):
        tasks[task_id]["progress"] = pct

    # Construct paths
    root_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    tag_model_path = root_dir / "weight" / "best-price-tag.pt"
    field_model_path = root_dir / "weight" / "best.pt"
    out_csv = OUTPUT_DIR / f"{task_id}.csv"

    try:
        process_video(
            video_path=video_path,
            tag_model_path=tag_model_path,
            field_model_path=field_model_path,
            out_csv=out_csv,
            conf=0.1,
            imgsz=640,
            frame_stride=frame_stride,
            tag_rotation=270,
            debug_dir=None, # Disable debug images as per instructions
            progress_callback=progress_callback
        )

        # Mark as done
        tasks[task_id]["status"] = "done"
        tasks[task_id]["progress"] = 100.0
        tasks[task_id]["csv_path"] = str(out_csv)
    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)
    finally:
        # Cleanup video if you want, or keep it. Let's clean up to save space.
        pass

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...), frame_stride: int = Form(50)):
    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid video format.")

    task_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{task_id}_{file.filename}"

    with open(video_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    tasks[task_id] = {"status": "pending", "progress": 0.0, "csv_path": None, "error": None}
    background_tasks.add_task(process_video_task, task_id, video_path, frame_stride)

    return JSONResponse({"task_id": task_id})

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return JSONResponse(task)

@app.get("/download/{task_id}")
async def download_csv(task_id: str):
    task = tasks.get(task_id)
    if not task or task["status"] != "done" or not task["csv_path"]:
        raise HTTPException(status_code=404, detail="CSV not found or not ready")

    return FileResponse(
        path=task["csv_path"],
        filename=os.path.basename(task["csv_path"]),
        media_type='text/csv'
    )
