# ============================================================
# PlantVision AI — Treinamento e Análise de Doenças em Plantas
# ============================================================

import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from plant_gradcam import run_full_analysis
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from plant_model import build_plant_classifier, enable_fine_tuning, diagnose_plant

# ----------------------------------------------------------
# Constantes globais
# ----------------------------------------------------------
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60
BATCH_SIZE = 32
EPOCHS_INITIAL = 10
EPOCHS_FINE_TUNE = 5

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus"
]


# ----------------------------------------------------------
# Carrega e pré-processa imagem
# ----------------------------------------------------------
def load_and_preprocess(img_path: str):
    path = Path(img_path)

    if not path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

    img_original = cv2.imread(str(path))

    if img_original is None:
        raise ValueError(f"Não foi possível carregar: {img_path}")

    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)

    img_array = np.expand_dims(img_resized, axis=0)
    img_tensor = preprocess_input(img_array.astype(np.float32))

    return img_original, img_tensor


# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------
def create_datasets():
    global CLASS_NAMES

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_data = train_gen.flow_from_directory(
        "dataset/train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_data = val_gen.flow_from_directory(
        "dataset/val",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    CLASS_NAMES = list(train_data.class_indices.keys())

    print("\nClasses encontradas:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{i}: {name}")

    return train_data, val_data


# ----------------------------------------------------------
# Treinamento
# ----------------------------------------------------------
def train_model():
    train_data, val_data = create_datasets()

    model, base_model = build_plant_classifier(
        num_classes=len(CLASS_NAMES)
    )

    print("\nIniciando treinamento inicial...")
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_INITIAL
    )

    print("\nAtivando fine-tuning...")
    enable_fine_tuning(model, base_model)

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_FINE_TUNE
    )

    model.save("plant_model.h5")
    print("\nModelo salvo como plant_model.h5")

    return model


# ----------------------------------------------------------
# Teste simples
# ----------------------------------------------------------
def test_single_image(model, img_path):
    _, img_tensor = load_and_preprocess(img_path)

    result = diagnose_plant(
        model,
        img_tensor,
        CLASS_NAMES,
        CONFIDENCE_THRESHOLD
    )

    print("\nResultado da análise:")
    print(f"Diagnóstico: {result['diagnosis']}")
    print(f"Confiança: {result['confidence']:.2%}")
    print(f"Confiável: {result['is_reliable']}")

    return result


# ----------------------------------------------------------
# Execução principal
# ----------------------------------------------------------
if __name__ == "__main__":
    model = tf.keras.models.load_model("plant_model.h5")

    CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus"
] # use exatamente os nomes das suas pastas

    result = run_full_analysis(
        "teste.jpg",
        model,
        CLASS_NAMES
    )

    print("\nResultado:")
    print(result)