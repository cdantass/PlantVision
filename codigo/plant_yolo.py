import argparse
from ultralytics import YOLO
from pathlib import Path


# =========================
# ✅ VALIDAÇÃO DO DATASET
# =========================
def validate_dataset(data_yaml: str):
    import yaml

    yaml_path = Path(data_yaml)

    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml não encontrado: {data_yaml}")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    base_dir = Path(data.get("path", yaml_path.parent)).resolve()

    train_dir = base_dir / data["train"]
    val_dir = base_dir / data["val"]

    if not train_dir.exists():
        raise FileNotFoundError(f"Train não encontrado: {train_dir}")

    if not val_dir.exists():
        raise FileNotFoundError(f"Val não encontrado: {val_dir}")

    print(f"\n✅ Dataset OK")
    print(f"Train: {train_dir}")
    print(f"Val: {val_dir}\n")


# =========================
# 🚀 TREINO YOLO (ESTÁVEL)
# =========================
def train_yolo_detector(
    data: str,
    model_name: str = "yolov8n.pt",
    epochs: int = 2,
    imgsz: int = 320,
    batch_size: int = 4,
    device: str = "",
):
    # 🔍 valida antes de treinar
    validate_dataset(data)

    print("🚀 Iniciando treino YOLO...\n")

    model = YOLO(model_name)

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=2,
        project="runs",  # padrão YOLO
        name="plant_yolo_det",
        device=device,
        verbose=True
    )

    print("\n✅ Treino finalizado!")

    return results


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Plant Detector")

    parser.add_argument("--data", required=True, help="Caminho do dataset.yaml")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--device", default="")

    return parser.parse_args()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    args = parse_args()

    train_yolo_detector(
        data=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
    )