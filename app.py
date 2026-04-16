import sys
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add src/ to path for DiagnosisService import
sys.path.insert(0, str(Path(__file__).parent / "src"))

from diagnosis.service import DiagnosisService
from diagnosis.types import AnalysisResult

# ---- Configuration ----
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus"
]
MODEL_PATH = Path(__file__).parent / "codigo" / "plant_model.h5"
OUTPUT_DIR = Path(__file__).parent / "static" / "outputs"

# ---- App Setup ----
app = FastAPI(title="PlantVision Web")

# Mount static files for frontend assets and GradCAM outputs
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create output dir at startup
@app.on_event("startup")
def startup():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Service Singleton ----
# Initialize once at startup, reuse across requests (LOCL-02)
service = DiagnosisService(
    model_path=str(MODEL_PATH),
    class_names=CLASS_NAMES,
    confidence_threshold=0.6
)

# ---- API Endpoint ----
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    """Analyze uploaded plant image and return diagnosis + GradCAM."""
    # 1. Validate content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    # 2. Save upload to temporary file
    suffix = Path(file.filename).suffix if file.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 3. Call service.analyze() with output_dir=OUTPUT_DIR
        result: AnalysisResult = service.analyze(tmp_path, output_dir=str(OUTPUT_DIR))

        # 4. Convert filesystem heatmap_path to URL-relative path
        #    e.g., "C:\...\static\outputs\abc_gradcam.jpg" → "/static/outputs/abc_gradcam.jpg"
        heatmap_url = "/" + Path(result["heatmap_path"]).relative_to(Path(__file__).parent).as_posix()
        result["heatmap_path"] = heatmap_url

        # 5. Cleanup temp file
        Path(tmp_path).unlink(missing_ok=True)

        return result
    except Exception as e:
        # Log server-side
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")

# ---- Frontend Route ----
@app.get("/")
async def serve_index():
    """Serve the frontend HTML page."""
    return FileResponse(Path(__file__).parent / "index.html")
