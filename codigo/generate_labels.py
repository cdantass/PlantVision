from pathlib import Path
import tensorflow as tf
import shutil

from codigo.plant_gradcam import run_full_analysis
from codigo.plant_analysis import CLASS_NAMES

# =========================
# BASE
# =========================
ROOT_DIR = Path(__file__).resolve().parent.parent

IMG_VAL_DIR = ROOT_DIR / "codigo" / "dataset_flat" / "images" / "val"
LBL_VAL_DIR = ROOT_DIR / "codigo" / "dataset_flat" / "labels" / "val"

# 👉 origem das imagens (onde estão as imagens ainda não usadas)
DATASET_SOURCE = ROOT_DIR / "codigo" / "dataset_flat" / "images" / "train"

MODEL_PATH = ROOT_DIR / "codigo" / "plant_model.h5"

# =========================
# CRIAR PASTA VAL
# =========================
IMG_VAL_DIR.mkdir(parents=True, exist_ok=True)
LBL_VAL_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CARREGAR MODELO
# =========================
print("Carregando modelo...")
model = tf.keras.models.load_model(str(MODEL_PATH))

# =========================
# PEGAR IMAGENS PARA VAL
# =========================
images = list(DATASET_SOURCE.glob("*.jpg"))

# 👉 quantidade que vai virar VAL (ex: 20%)
val_size = int(len(images) * 0.2)

val_images = images[:val_size]

print(f"Gerando VAL com {len(val_images)} imagens...\n")

# =========================
# PROCESSAR VAL
# =========================
for img in val_images:
    try:
        print(f"VAL: {img.name}")

        # copiar imagem para VAL
        new_img_path = IMG_VAL_DIR / img.name
        shutil.copy(img, new_img_path)

        # gerar label
        run_full_analysis(
            str(new_img_path),
            model,
            CLASS_NAMES,
            save_label=True,
            label_dir=str(LBL_VAL_DIR)
        )

    except Exception as e:
        print(f"Erro em {img.name}: {e}")

print("\n✅ VAL gerado com sucesso dentro de dataset_flat!")