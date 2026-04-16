# PlantVision Web

Local web interface for plant disease diagnosis using the existing Python ML engine.

## Features

- Upload a plant image through the browser
- AI-powered disease classification (4 classes)
- Confidence score and reliability indicator
- Top-3 predictions with probabilities
- GradCAM heatmap visualization

## Requirements

Python 3.12+ with the packages in `requirements.txt`.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Application

Start the FastAPI server:

```bash
uvicorn app:app --reload --port 8000
OU 
python -m uvicorn app:app --reload --port 8000
```

Open your browser: [http://localhost:8000](http://localhost:8000)

## Usage

1. Click "Choose File" and select a plant image (JPG, PNG, etc.)
2. Click "Analyze"
3. Wait a few seconds for the AI to process
4. View diagnosis, confidence, top-3 predictions, and GradCAM heatmap

## Test Image

A sample image is included at `codigo/teste.jpg` for quick testing.

## Project Structure

- `app.py` — FastAPI backend
- `index.html` — Frontend page
- `static/` — Static assets (JS, GradCAM outputs)
- `src/diagnosis/` — Reusable diagnosis service (Phase 1)
- `codigo/` — Original model and scripts (unchanged)

## Notes

- Runs entirely locally — no internet required after dependencies installed
- Model file: `codigo/plant_model.h5` (included)
- GradCAM outputs saved to `static/outputs/`
