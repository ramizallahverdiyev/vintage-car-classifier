<h1 align="center">Vintage Car Classifier</h1>

<p align="center">
  Identify the <strong>make, model, and year</strong> of any car from a single photo.
  <br>
  <strong>196 classes</strong> · <strong>93.43% test accuracy</strong> · Real-time Grad-CAM heatmaps
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/pytorch-2.1+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch 2.1+">
  <img src="https://img.shields.io/badge/fastapi-0.104+-009688?logo=fastapi&logoColor=white" alt="FastAPI 0.104+">
  <img src="https://img.shields.io/badge/tests-10/10-brightgreen" alt="Tests: 10/10 passing">
  <img src="https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker ready">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="MIT License">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Quick Inference (pretrained model included)](#quick-inference-pretrained-model-included)
  - [Dataset Preparation](#dataset-preparation)
  - [Training from Scratch](#training-from-scratch)
- [API Reference](#api-reference)
  - [Health Check](#health-check)
  - [Predict](#predict)
  - [Grad-CAM](#grad-cam)
  - [Web UI](#web-ui)
- [Frontend](#frontend)
- [Model](#model)
  - [Architecture](#model-architecture)
  - [Training Details](#training-details)
  - [Data Augmentation](#data-augmentation)
  - [Export](#model-export)
- [Results](#results)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Project Structure](#project-structure)
- [Colab Training](#colab-training)
- [Troubleshooting](#troubleshooting)
- [Built With](#built-with)
- [License](#license)

---

## Overview

This project is an end-to-end **car make/model/year classifier** that combines deep learning with a production-grade API server. Given a photo of any car, it returns the top-3 predictions with confidence scores and a heatmap overlay showing which parts of the image drove the model's decision.

The model is an **EfficientNet-B0** trained via transfer learning on the full **Stanford Cars dataset** (196 classes, 16,185 images), achieving **93.43% test accuracy**. The trained model is exported to **TorchScript** for fast CPU inference, wrapped in a **FastAPI** server with a drag-and-drop **web UI**, and containerized with **Docker** for one-command deployment.

---

## Features

- **196-class classification** — trained on all Stanford Cars categories (Acura to Volvo, 1990–2015)
- **93.43% test accuracy** — EfficientNet-B0 with transfer learning, data augmentation, and Adam optimizer
- **Grad-CAM heatmaps** — model explainability that highlights the visual regions contributing to each prediction
- **Drag-and-drop web UI** — polished vintage-themed interface with animated confidence gauge and podium-style top-3
- **RESTful API** — clean `/predict` and `/gradcam` endpoints with JSON responses
- **TorchScript export** — model runs without PyTorch source code dependency; ~50ms inference on CPU
- **Dockerized** — single container with all dependencies, ready to deploy anywhere
- **CI/CD pipeline** — GitHub Actions runs tests and builds the Docker image on every push to `main`
- **Colab training** — self-contained notebook for GPU training on Google Colab (free T4)
- **GPU-optional** — trains on CUDA/MPS, runs inference on any device including CPU

---

## Architecture

```
┌──────────┐     ┌────────────────────────────────────────────────┐
│  Browser │────▶│              FastAPI Server                     │
│ (UI)     │     │  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│          │     │  │  /       │  │ /predict │  │  /gradcam    │ │
│ Drag &   │     │  │ HTML UI  │  │ JSON     │  │  JSON + b64  │ │
│ Drop     │     │  └──────────┘  └────┬─────┘  └──────┬───────┘ │
└──────────┘     │                      │               │         │
                 │              ┌───────┴───────┐       │         │
                 │              │ ModelServer   │       │         │
                 │              │ ┌───────────┐ │       │         │
                 │              │ │TorchScript│ │◀──────┘         │
                 │              │ │ Model.pt  │ │                 │
                 │              │ └───────────┘ │                 │
                 │              │ ┌───────────┐ │                 │
                 │              │ │ PyTorch   │ │← Grad-CAM only │
                 │              │ │ Checkpoint│ │                 │
                 │              │ └───────────┘ │                 │
                 │              └───────────────┘                 │
                 └────────────────────────────────────────────────┘
```

The server loads two models on startup:

1. **TorchScript model** (`model_torchscript.pt`) — primary inference engine, loaded for all predictions
2. **PyTorch checkpoint** (`best_model.pth`) — only used for Grad-CAM generation (hooks not supported on TorchScript modules)

If the checkpoint is missing, the Grad-CAM endpoint gracefully falls back to returning prediction results without a heatmap.

---

## Getting Started

### Prerequisites

- Python 3.10+ (3.11 recommended)
- pip
- (Optional) Docker Desktop for containerized deployment
- (Optional) NVIDIA GPU or Apple Silicon for training

### Quick Inference (pretrained model included)

The trained TorchScript model is included in the repository at `models/exported/model_torchscript.pt`. You don't need to train anything to run predictions.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/vintage-car-classifier.git
cd vintage-car-classifier

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser, drag in a car photo, and get the prediction with Grad-CAM heatmap.

Or use curl:

```bash
curl -X POST -F "file=@car.jpg" http://localhost:8000/predict | jq .
```

```json
{
  "class_name": "Ford_Mustang_Convertible_2007",
  "confidence": 0.9642,
  "top_3": [
    {"class_name": "Ford_Mustang_Convertible_2007", "probability": 0.9642},
    {"class_name": "Chevrolet_Camaro_Convertible_2012", "probability": 0.0211},
    {"class_name": "Dodge_Challenger_SRT8_2011", "probability": 0.0083}
  ]
}
```

### Dataset Preparation

Skip this section if you only want inference (pretrained model included). These steps are required only if you want to retrain from scratch.

The project uses the [Stanford Cars dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset). The dataset download uses `kagglehub`:

```bash
# Download and prepare
python -c "from data.dataset import download_dataset; download_dataset()"
python -m data.prepare
```

This will:
1. Download the Stanford Cars dataset (16,185 images, 196 classes)
2. Organize images into class-named subdirectories
3. Split into train/val/test (70/15/15)
4. Save processed data to `data/processed/`

**Dataset statistics:**

| Split | Images | Percentage |
|-------|--------|------------|
| Train | ~11,300 | 70% |
| Validation | ~2,400 | 15% |
| Test | ~2,400 | 15% |
| **Total** | **16,185** | **100%** |

### Training from Scratch

Training on a CPU is slow (~2 minutes per epoch). **Use the Colab notebook** (`train_colab.ipynb`) for GPU training — upload it to Google Colab, run all cells, and download the trained model.

#### Local training (CPU/GPU):

```bash
# Train
python -m models.train

# Evaluate and export
python -m models.evaluate
```

Key training configuration (`configs/config.yaml`):

| Parameter | Value |
|-----------|-------|
| Model | EfficientNet-B0 |
| Epochs | 30 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Weight decay | 0.0001 |
| Image size | 224×224 |
| Early stopping patience | 7 |
| Dropout | 0.2 |

---

## API Reference

### Health Check

```
GET /health
```

Returns the server and model status.

**Response:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "num_classes": 196
}
```

### Predict

```
POST /predict
Content-Type: multipart/form-data

Body: file=<image>
```

Returns the top-3 predictions with confidence scores.

**Response:**

```json
{
  "class_name": "Ford_Mustang_Convertible_2007",
  "confidence": 0.9642,
  "top_3": [
    {"class_name": "Ford_Mustang_Convertible_2007", "probability": 0.9642},
    {"class_name": "Chevrolet_Camaro_Convertible_2012", "probability": 0.0211},
    {"class_name": "Dodge_Challenger_SRT8_2011", "probability": 0.0083}
  ]
}
```

### Grad-CAM

```
POST /gradcam
Content-Type: multipart/form-data

Body: file=<image>
```

Returns the prediction plus a base64-encoded PNG of the heatmap overlay.

**Response:**

```json
{
  "class_name": "Ford_Mustang_Convertible_2007",
  "confidence": 0.9642,
  "heatmap_base64": "iVBORw0KGgo..."
}
```

The heatmap is a 224×224 PNG image overlaid on the original photo with 50% opacity. Decode the base64 string to display or save:

```python
import base64
from PIL import Image
import io

b64 = response["heatmap_base64"]
img = Image.open(io.BytesIO(base64.b64decode(b64)))
img.save("heatmap.png")
```

### Web UI

```
GET /
```

Serves the drag-and-drop web interface.

---

## Frontend

The web UI is a single-page application built with vanilla HTML, CSS, and JavaScript — no framework dependencies.

**Features:**

- **Drag-and-drop upload** with click-to-browse fallback
- **Image preview** showing the uploaded photo
- **Animated circular gauge** displaying confidence percentage via SVG
- **Podium-style top-3** with gold/silver/bronze ranks and confidence mini-bars
- **Side-by-side comparison** — original photo above heatmap overlay below
- **Vintage amber theme** with glass-morphism cards, film grain texture, and subtle glow effects
- **Space Grotesk** typography via Google Fonts
- **Staggered animations** — results slide up sequentially
- **Responsive layout** — two columns on desktop, single column on mobile
- **Graceful degradation** — Grad-CAM failure doesn't block prediction display

**Files:**

- `frontend/index.html` — HTML structure
- `frontend/style.css` — Complete styling with CSS custom properties
- `frontend/app.js` — File handling, API calls, UI updates

---

## Model

### Model Architecture

```python
EfficientNet-B0 (pretrained on ImageNet)
│
├── features           # Feature extractor (frozen initially)
│   ├── 0..8           # 9 MBConv blocks with SE attention
│   └── avgpool        # Adaptive average pooling
│
└── classifier         # Custom classification head
    ├── Dropout(p=0.2)
    └── Linear(1280 → 196)
```

The model uses transfer learning:
1. **EfficientNet-B0** backbone pretrained on ImageNet
2. The original classifier head is replaced with Dropout + Linear layer
3. All layers are fine-tuned during training

### Training Details

| Aspect | Value |
|--------|-------|
| Framework | PyTorch 2.1+ |
| Architecture | EfficientNet-B0 |
| Pretrained weights | ImageNet (via torchvision) |
| Total parameters | ~5.3M |
| Training epochs | 30 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Loss function | Cross-entropy |
| Early stopping | Patience of 7 |
| Learning rate scheduler | None (constant LR) |

### Data Augmentation

Applied during training to improve generalization:

- Random rotation (±10°)
- Random horizontal flip (50% probability)
- Color jitter (brightness ±0.2, contrast ±0.2, saturation ±0.2)
- Random resized crop (scale 0.8–1.0, aspect ratio preserved)
- Resize to 256px, center crop to 224px
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

During inference, only resize + center crop + normalization are applied.

### Model Export

After training, the model is exported in two formats:

1. **TorchScript** (`models/exported/model_torchscript.pt`, 17.9 MB) — optimized for CPU inference
   - No PyTorch source code dependency
   - ~50ms per prediction on CPU
   - Primary inference engine for the API server

2. **PyTorch checkpoint** (`models/checkpoints/best_model.pth`, 51.6 MB) — full state dict
   - Used for Grad-CAM visualization (requires hooks)
   - Can be used to resume training

---

## Results

| Metric | Value |
|--------|-------|
| **Test accuracy** | **93.43%** |
| Training epochs | 30 |
| Training time | ~7.5 minutes (T4 GPU) |
| Inference time | ~50ms (CPU), ~15ms (GPU) |
| Model size (TorchScript) | 17.9 MB |

The confusion matrix is saved at `models/exported/confusion_matrix.png`.

---

## Docker

```bash
# Build the image
docker build -f docker/Dockerfile -t car-classifier .

# Run the container
docker run -p 8000:8000 car-classifier
```

The Docker image:
- Uses `python:3.11-slim` as the base image
- Installs only production dependencies (no test/dev packages)
- Exposes port 8000
- Starts the FastAPI server with uvicorn

The model files are included in the image, so no additional downloads are needed at runtime.

---

## CI/CD

The repository includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs on every push and pull request to `main`:

**Test job:**
- Sets up Python 3.12
- Installs dependencies
- Runs the full pytest suite (10 tests)
- Fails on any test failure

**Docker job** (runs after tests pass):
- Sets up Docker Buildx
- Builds the Docker image
- Does not push to a registry (configure as needed)

The pipeline ensures that:
- All tests pass before merging
- The Docker image builds successfully
- Breaking changes are caught automatically

---

## Project Structure

```
vintage-car-classifier/
│
├── api/                        # FastAPI web server
│   ├── __init__.py
│   ├── main.py                 # Routes: /, /health, /predict, /gradcam
│   ├── inference.py            # ModelServer: load, predict, Grad-CAM
│   └── schemas.py              # Pydantic response models
│
├── configs/
│   └── config.yaml             # All training and API configuration
│
├── data/
│   ├── __init__.py
│   ├── dataset.py              # Dataset download (kagglehub) + PyTorch Dataset
│   ├── prepare.py              # Train/val/test split into class folders
│   ├── raw/                    # Raw dataset (gitignored)
│   └── processed/              # Organized splits (gitignored)
│
├── docker/
│   └── Dockerfile              # Production container definition
│
├── frontend/
│   ├── index.html              # Web UI structure
│   ├── style.css               # Complete styling
│   └── app.js                  # File handling, API calls, UI updates
│
├── models/
│   ├── __init__.py
│   ├── model.py                # build_model() + get_device()
│   ├── train.py                # Training loop with early stopping
│   ├── evaluate.py             # Evaluation, metrics, confusion matrix, export
│   ├── gradcam.py              # Grad-CAM implementation (hooks-based)
│   ├── checkpoints/            # PyTorch checkpoints (gitignored)
│   │   ├── .gitkeep
│   │   └── best_model.pth      # Trained state dict (51.6 MB)
│   └── exported/               # Exported models for inference
│       ├── .gitkeep
│       ├── model_torchscript.pt # TorchScript model (17.9 MB)
│       ├── metadata.json        # Class names and test accuracy
│       └── confusion_matrix.png # 196-class confusion matrix
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   ├── test_api.py             # API endpoint tests (5 tests)
│   └── test_model.py           # Model architecture tests (5 tests)
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
│
├── .dockerignore               # Files excluded from Docker build
├── .gitignore                  # Files excluded from version control
├── Makefile                    # Common commands: test, serve, train, docker
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package definition
└── train_colab.ipynb           # Self-contained Colab notebook for GPU training
```

---

## Colab Training

`train_colab.ipynb` is a self-contained Jupyter notebook designed to run on Google Colab's free T4 GPU:

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `train_colab.ipynb`
3. Set runtime to **T4 GPU** (Runtime → Change runtime type)
4. Run all cells
5. Download the `training_output.zip` containing the trained model
6. Extract into your project root

The notebook handles:
- Installing all dependencies
- Downloading the Stanford Cars dataset via Kaggle API
- Training EfficientNet-B0 for 30 epochs
- Evaluating and exporting to TorchScript
- Downloading the results as a zip file

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install -r requirements.txt` from your virtual environment |
| Model not loading at startup | Ensure `models/exported/model_torchscript.pt` exists (it is tracked in git) |
| Grad-CAM returns 500 | The PyTorch checkpoint (`models/checkpoints/best_model.pth`) is missing. The prediction still works, but heatmaps require the checkpoint. Either retrain or download it from a release |
| Port 8000 already in use | Kill the process: `netstat -ano \| findstr :8000`, then `taskkill /F /PID <PID>` |
| `formData.clone is not a function` | Browser compatibility issue. Refresh with Ctrl+F5 to get the updated frontend JS |
| Slow prediction | The first prediction loads the model. Subsequent calls are faster (~50ms on CPU) |
| Docker build fails on Windows | Ensure Docker Desktop is running and using Linux containers |

---

## Built With

- [PyTorch](https://pytorch.org/) — Deep learning framework for model definition, training, and inference
- [TorchVision](https://pytorch.org/vision/) — Pretrained EfficientNet-B0 weights and image transforms
- [TorchScript](https://pytorch.org/docs/stable/jit.html) — Model export for dependency-free CPU inference
- [FastAPI](https://fastapi.tiangolo.com/) — Async Python web framework with automatic OpenAPI docs
- [Uvicorn](https://www.uvicorn.org/) — ASGI server for FastAPI
- [Pillow](https://python-pillow.org/) — Image loading and manipulation
- [NumPy](https://numpy.org/) — Array operations for heatmap generation
- [Matplotlib](https://matplotlib.org/) — Confusion matrix and heatmap colormaps
- [scikit-learn](https://scikit-learn.org/) — Accuracy metrics and confusion matrix computation
- [Seaborn](https://seaborn.pydata.org/) — Confusion matrix visualization
- [Docker](https://www.docker.com/) — Containerization for reproducible deployment
- [GitHub Actions](https://github.com/features/actions) — CI/CD pipeline with automated testing
- [Google Colab](https://colab.research.google.com/) — Free GPU (T4) for model training

---

## License

MIT — feel free to use, modify, and distribute.
