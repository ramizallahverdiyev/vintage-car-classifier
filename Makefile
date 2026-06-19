.PHONY: setup train evaluate serve test docker-build docker-run clean

PROJECT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
VENV ?= $(PROJECT_DIR).venv
PYTHON := $(VENV)/Scripts/python
PIP := $(VENV)/Scripts/pip

setup:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PYTHON) -m pip install -e .

train:
	$(PYTHON) -m models.train

evaluate:
	$(PYTHON) -m models.evaluate

serve:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

docker-build:
	docker build -f docker/Dockerfile -t vintage-car-classifier .

docker-run:
	docker run -p 8000:8000 vintage-car-classifier

download-data:
	$(PYTHON) -c "from data.dataset import download_dataset; download_dataset()"

clean:
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -rf models/checkpoints/*
	rm -rf models/exported/*
	rm -rf data/processed/*
	rm -rf logs/*
	rm -rf .pytest_cache
