from pathlib import Path
from codigo.plant_analysis import load_and_preprocess, diagnose_plant, CLASS_NAMES
from codigo.plant_yolo import PlantYOLODetector
from codigo.plant_model import build_plant_classifier
import tensorflow as tf


def analyze_with_yolo(img_path: str, model_path: str, output_dir: str = None):
    """
    Análise completa: Classificação TensorFlow + Detecção YOLO
    """

    # 1. Carrega o modelo TensorFlow
    model = tf.keras.models.load_model(model_path)

    # 2. Análise de classificação
    img_original, img_tensor = load_and_preprocess(img_path)
    diagnosis = diagnose_plant(model, img_tensor, CLASS_NAMES)

    # 3. Inicializa detector YOLO
    detector = PlantYOLODetector(model_name="yolov8n.pt")

    # 4. Detecção de áreas afetadas
    yolo_results = detector.detect_affected_areas(img_path, confidence=0.5)

    # 5. Combina resultados
    combined_analysis = detector.compare_with_diagnosis(diagnosis, yolo_results)

    # 6. (Opcional) Desenha bounding boxes
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / "detection_visualized.jpg"
        detector.annotate_image(img_path, str(output_path), confidence=0.5)
        combined_analysis["annotated_image_path"] = str(output_path)

    return combined_analysis


# ---- Execução principal ----
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    test_image = BASE_DIR / "teste.jpg"
    model_file = BASE_DIR / "plant_model.h5"

    if not test_image.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {test_image}")

    if not model_file.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_file}")

    result = analyze_with_yolo(str(test_image), str(model_file), output_dir="outputs")

    print("=" * 60)
    print("DIAGNÓSTICO COMPLETO")
    print("=" * 60)

    print("\nClassificação TensorFlow:")
    print(f"  Diagnóstico: {result['diagnosis']['diagnosis']}")
    print(f"  Confiança: {result['diagnosis']['confidence']:.2%}")

    print("\nDetecção de Áreas (YOLO):")
    print(f"  Áreas afetadas detectadas: {result['affected_areas']['detected']}")
    print(f"  Percentual afetado: {result['affected_areas']['percentage_affected']:.2f}%")
    print(f"  Severidade: {result['severity_assessment']}")