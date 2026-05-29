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
YOLO_MODEL_PATH = Path(__file__).parent / "runs" / "detect" / "runs" / "plant_yolo_det" / "weights" / "best.pt"
OUTPUT_DIR = Path(__file__).parent / "static" / "outputs"

# ---- Translations ----
CLASS_TRANSLATIONS = {
    "Pepper__bell___Bacterial_spot": "Pimentão com mancha bacteriana",
    "Pepper__bell___healthy": "Pimentão saudável",
    "Potato___Early_blight": "Batata com requeima precoce",
    "Potato___healthy": "Batata saudável",
    "Potato___Late_blight": "Batata com requeima tardia",
    "Tomato__Target_Spot": "Tomate com mancha alvo",
    "Tomato__Tomato_mosaic_virus": "Tomate com vírus do mosaico",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomate com vírus do enrolamento amarelo",
    "Tomato_Bacterial_spot": "Tomate com mancha bacteriana",
    "Tomato_Early_blight": "Tomate com requeima precoce",
    "Tomato_healthy": "Tomate saudável",
    "Tomato_Late_blight": "Tomate com requeima tardia",
    "Tomato_Leaf_Mold": "Tomate com mofo foliar",
    "Tomato_Septoria_leaf_spot": "Tomate com mancha de septória",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomate com infestação de ácaros"
}


def generate_plant_feedback(diagnosis_translated: str, confidence: float):
    """Generate status, message, and recommendation based on translated diagnosis."""
    name_lower = diagnosis_translated.lower()

    if "saudável" in name_lower:
        status = "Saudável"
        message = "A planta aparenta estar saudável."
        recommendation = "Continue monitorando regularmente."
    elif confidence < 0.6:
        status = "Inconclusivo"
        message = "Não foi possível determinar com segurança o estado da planta."
        recommendation = "Tente fotografar a folha com melhor iluminação e foco."
    else:
        status = "Doente"
        message = f"A planta apresenta sinais de {diagnosis_translated}."
        recommendation = "Consulte um agrônomo e inicie o tratamento adequado."

    return status, message, recommendation


def path_to_url(absolute_path: Path, project_root: Path) -> str:
    """Convert an absolute filesystem path to a relative URL for the frontend."""
    try:
        relative = absolute_path.relative_to(project_root)
        return "/" + relative.as_posix()
    except ValueError:
        # Fallback: serve from /static/outputs/ by filename
        return "/static/outputs/" + absolute_path.name


# ---- App Setup ----
app = FastAPI(title="PlantVision Web")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def startup():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- Service Singleton ----
print(f"🔍 MODEL_PATH:      {MODEL_PATH}  (exists={MODEL_PATH.exists()})")
print(f"🔍 YOLO_MODEL_PATH: {YOLO_MODEL_PATH}  (exists={YOLO_MODEL_PATH.exists()})")

service = DiagnosisService(
    model_path=str(MODEL_PATH),
    class_names=CLASS_NAMES,
    confidence_threshold=0.6,
    yolo_model_path=str(YOLO_MODEL_PATH) if YOLO_MODEL_PATH.exists() else None
)

print(f"🔍 YoloService carregado: {service.yolo_service is not None}")


# ---- API Endpoint ----
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    """Analyze uploaded plant image and return diagnosis + GradCAM + YOLO."""

    # 1. Validate content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Tipo de arquivo inválido. Envie uma imagem.")

    # 2. Save upload to temp file
    suffix = Path(file.filename).suffix if file.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        project_root = Path(__file__).parent.resolve()

        # 3. Run full analysis (GradCAM + YOLO)
        result: AnalysisResult = service.analyze(tmp_path, output_dir=str(OUTPUT_DIR))

        print(f"✅ Result keys: {list(result.keys())}")
        print(f"✅ YOLO result: {result.get('yolo')}")

        # 4. Translate diagnosis
        diagnosis_raw = result["diagnosis"]
        diagnosis_translated = CLASS_TRANSLATIONS.get(diagnosis_raw, diagnosis_raw)
        result["diagnosis"] = diagnosis_translated

        # 5. Translate top3
        if "top3" in result:
            for item in result["top3"]:
                raw_class = item["class"]
                item["class"] = CLASS_TRANSLATIONS.get(raw_class, raw_class)

        # 6. Convert GradCAM path → URL
        heatmap_abs = Path(result["heatmap_path"]).resolve()
        result["heatmap_path"] = path_to_url(heatmap_abs, project_root)

        # 7. Convert YOLO annotated path → URL (if present)
        yolo = result.get("yolo")
        if yolo and yolo.get("annotated_path"):
            yolo_abs = Path(yolo["annotated_path"]).resolve()
            yolo["annotated_path"] = path_to_url(yolo_abs, project_root)
            result["yolo"] = yolo

        # 8. Generate feedback fields
        status, message, recommendation = generate_plant_feedback(
            diagnosis_translated, result["confidence"]
        )
        result["condition"] = f"{status} — {message}"
        result["recommendation"] = recommendation

        return result

    except Exception:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Análise falhou. Tente novamente.")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---- Frontend Route ----
@app.get("/")
async def serve_index():
    return FileResponse(Path(__file__).parent / "index.html")