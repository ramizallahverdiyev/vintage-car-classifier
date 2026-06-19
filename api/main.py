import io
import os
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.schemas import PredictResponse, GradCAMResponse, HealthResponse
from api.inference import model_server

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_server.load_model()
    yield


app = FastAPI(
    title="Vintage Car Classifier",
    description="Identify vintage car make, model, and era from photos",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if model_server.loaded else "model not loaded",
        model_loaded=model_server.loaded,
        num_classes=model_server.num_classes(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...)):
    if not model_server.loaded:
        raise HTTPException(503, "Model not loaded yet")
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Invalid image file")
    result = model_server.predict(image)
    return PredictResponse(**result)


@app.post("/gradcam", response_model=GradCAMResponse)
def gradcam(file: UploadFile = File(...)):
    if not model_server.loaded:
        raise HTTPException(503, "Model not loaded yet")
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Invalid image file")
    result = model_server.predict_with_gradcam(image)
    return GradCAMResponse(**result)


@app.get("/style.css", response_class=FileResponse)
def serve_css():
    return FileResponse(os.path.join(PROJECT_DIR, "frontend", "style.css"))


@app.get("/app.js", response_class=FileResponse)
def serve_js():
    return FileResponse(os.path.join(PROJECT_DIR, "frontend", "app.js"))


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(PROJECT_DIR, "frontend", "index.html")
    try:
        with open(html_path) as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Vintage Car Classifier API</h1><p>Frontend not found.</p>"
