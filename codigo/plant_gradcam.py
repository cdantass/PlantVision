import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

from .plant_model import diagnose_plant


# ==============================
# BASE DIR (FIX GLOBAL)
# ==============================
ROOT_DIR = Path(__file__).resolve().parent.parent


# ==============================
# GRAD-CAM
# ==============================
def compute_gradcam(model, img_tensor, class_idx, last_conv_layer="Conv_1"):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_channel = predictions[:, class_idx]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


# ==============================
# PREPROCESSAMENTO
# ==============================
def load_and_preprocess(img_path):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    IMG_SIZE = (224, 224)

    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Erro ao carregar imagem: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)

    img_array = np.expand_dims(img_resized, axis=0)
    img_tensor = preprocess_input(img_array.astype(np.float32))

    return img, img_tensor


# ==============================
# HEATMAP VISUAL
# ==============================
def overlay_heatmap(img_bgr, heatmap, alpha=0.4):
    h, w = img_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)

    return overlay


# ==============================
# HEATMAP → BBOX (REAL)
# ==============================
def heatmap_to_bbox(heatmap, original_shape, threshold=0.5):
    H, W = original_shape[:2]

    heatmap_resized = cv2.resize(heatmap, (W, H))
    mask = heatmap_resized > threshold
    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    return x, y, w, h


# ==============================
# SALVAR LABEL YOLO
# ==============================
def save_yolo_label(img_path, bbox, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    x, y, w, h = bbox

    x_center = (x + w / 2) / W
    y_center = (y + h / 2) / H
    width = w / W
    height = h / H

    label_path = output_dir / (Path(img_path).stem + ".txt")

    with open(label_path, "w") as f:
        f.write(f"0 {x_center} {y_center} {width} {height}")


# ==============================
# PIPELINE COMPLETO
# ==============================
def run_full_analysis(
    img_path: str,
    model: tf.keras.Model,
    class_names: list,
    save_label: bool = False,
    label_dir: str = None
) -> dict:

    # ==========================
    # DEFINIR OUTPUT DIR (FIX)
    # ==========================
    OUTPUT_DIR = ROOT_DIR / "codigo" / "dataset" / "debug"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carregar imagem
    img_original, img_tensor = load_and_preprocess(img_path)

    # 2. Classificação
    result = diagnose_plant(model, img_tensor, class_names)

    # 3. Classe predita
    class_idx = class_names.index(result['diagnosis'])

    # 4. Grad-CAM
    heatmap = compute_gradcam(model, img_tensor, class_idx)

    # 5. Overlay visual
    overlay = overlay_heatmap(img_original, heatmap)

    # ==========================
    # SALVAR HEATMAP NO LUGAR CERTO
    # ==========================
    output_path = OUTPUT_DIR / f"{Path(img_path).stem}_diagnosis.jpg"
    cv2.imwrite(str(output_path), overlay)

    result["heatmap_path"] = str(output_path)

    # ==========================
    # GERAR BBOX + YOLO LABEL
    # ==========================
    bbox = heatmap_to_bbox(heatmap, img_original.shape)

    if bbox:
        result["bbox"] = bbox

        if save_label and label_dir:
            save_yolo_label(img_path, bbox, label_dir)
            result["label_saved"] = True
        else:
            result["label_saved"] = False
    else:
        result["bbox"] = None
        result["label_saved"] = False

    return result