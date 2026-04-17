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
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite"
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
        diagnosis_raw = result["diagnosis"]
        result["diagnosis"] = CLASS_TRANSLATIONS.get(diagnosis_raw, diagnosis_raw)
        if "top3" in result:
            for item in result["top3"]:
                raw_class = item["class"]
                item["class"] = CLASS_TRANSLATIONS.get(raw_class, raw_class)

        # 4. Convert filesystem heatmap_path to URL-relative path
        #    e.g., "C:\...\static\outputs\abc_gradcam.jpg" → "/static/outputs/abc_gradcam.jpg"
        heatmap_url = "/" + Path(result["heatmap_path"]).relative_to(Path(__file__).parent).as_posix()
        result["heatmap_path"] = heatmap_url

        # 5. Cleanup temp file
        # 5. Gerar interpretação humana
        status, message, recommendation = generate_plant_feedback(result)

        result["condition"] = f"{status} — {message} ({recommendation})"

        # 6. Cleanup temp file
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

CLASS_TRANSLATIONS = {
    "Pepper__bell___Bacterial_spot": "Folha com mancha bacteriana",
    "Pepper__bell___healthy": "Folha saudável",

    "Potato___Early_blight": "Folha com requeima precoce",
    "Potato___healthy": "Folha saudável",
    "Potato___Late_blight": "Folha com requeima tardia",

    "Tomato__Target_Spot": "Folha com mancha alvo",
    "Tomato__Tomato_mosaic_virus": "Folha com vírus do mosaico",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Folha com vírus do enrolamento amarelo",
    "Tomato_Bacterial_spot": "Folha com mancha bacteriana",
    "Tomato_Early_blight": "Folha com requeima precoce",
    "Tomato_healthy": "Folha saudável",
    "Tomato_Late_blight": "Folha com requeima tardia",
    "Tomato_Leaf_Mold": "Folha com mofo",
    "Tomato_Septoria_leaf_spot": "Folha com mancha de septória",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Folha com infestação de ácaros"
}


def generate_plant_feedback(result):
    diagnosis = result["diagnosis"]
    confidence = result["confidence"]

    if "saudável" in diagnosis.lower():
        status = "Saudável"
        message = "A planta aparenta estar saudável."
        recommendation = "Continue monitorando."

    elif confidence < 0.6:
        status = "Inconclusivo"
        message = "Não foi possível determinar com segurança."
        recommendation = "Tente outra imagem."

    else:
        status = "Doente"
        message = f"A planta apresenta sinais de {diagnosis}."
        recommendation = "Recomenda-se tratamento."

    return status, message, recommendation