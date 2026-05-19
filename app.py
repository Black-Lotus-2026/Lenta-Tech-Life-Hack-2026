import os
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request

from lenta_shelf_ai.pipeline import PipelineConfig, PriceTagPipeline

app = FastAPI()

templates = Jinja2Templates(directory="templates")

default_runtime_dir = Path("/tmp/lenta_runtime") if os.environ.get("VERCEL") else Path(".")
runtime_dir = Path(os.environ.get("LENTA_RUNTIME_DIR", str(default_runtime_dir)))
UPLOAD_DIR = Path(os.environ.get("LENTA_UPLOAD_DIR", str(runtime_dir / "uploads")))
OUTPUT_DIR = Path(os.environ.get("LENTA_OUTPUT_DIR", str(runtime_dir / "outputs")))
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

    root_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    out_csv = OUTPUT_DIR / f"{task_id}.csv"

    try:
        cfg = PipelineConfig.from_file(root_dir / "configs" / "default.yaml")
        cfg.yolo_weights = str(root_dir / "models" / "price_tag_yolo.pt")
        cfg.field_zone_weights = str(root_dir / "models" / "field_zone_yolo.pt")
        cfg.save_crops = False
        cfg.save_debug_json = True
        cfg.field_zone_save_crops = False
        cfg.max_frames = 0
        # The UI still exposes frame_stride for fast demos. Convert it into an
        # approximate FPS cap instead of skipping arbitrary frames in the app.
        if frame_stride >= 80:
            cfg.sample_fps = min(float(cfg.sample_fps), 1.0)
        elif frame_stride >= 40:
            cfg.sample_fps = min(float(cfg.sample_fps), 2.0)
        else:
            cfg.sample_fps = min(float(cfg.sample_fps), 4.0)

        tasks[task_id]["progress"] = 5.0
        pipe = PriceTagPipeline(cfg)
        task_output_dir = OUTPUT_DIR / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        df = pipe.run_video(video_path, output_dir=task_output_dir, output_csv=out_csv)

        tasks[task_id]["status"] = "done"
        tasks[task_id]["progress"] = 100.0
        tasks[task_id]["csv_path"] = str(out_csv)
        tasks[task_id]["rows"] = int(len(df))
    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)

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

    tasks[task_id] = {"status": "pending", "progress": 0.0, "csv_path": None, "error": None, "rows": 0}
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
