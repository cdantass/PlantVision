import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

from plant_model import diagnose_plant

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

def load_and_preprocess(img_path):
    import cv2
    import numpy as np
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    IMG_SIZE = (224, 224)

    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Erro ao carregar imagem")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)

    img_array = np.expand_dims(img_resized, axis=0)
    img_tensor = preprocess_input(img_array.astype(np.float32))

    return img, img_tensor


def overlay_heatmap(
    img_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:

    h, w = img_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)

    return overlay

def run_full_analysis(
    img_path: str,
    model: tf.keras.Model,
    class_names: list
) -> dict:
    """
    Pipeline completo de análise.
    """

    # 1. Carrega imagem
    img_original, img_tensor = load_and_preprocess(img_path)

    # 2. Diagnóstico
    result = diagnose_plant(model, img_tensor, class_names)

    # 3. Classe predita
    class_idx = class_names.index(result['diagnosis'])

    # 4. GradCAM
    heatmap = compute_gradcam(model, img_tensor, class_idx)

    # 5. Overlay
    overlay = overlay_heatmap(img_original, heatmap)

    # 6. Salvar imagem
    output_path = str(Path(img_path).stem) + "_diagnosis.jpg"
    cv2.imwrite(output_path, overlay)

    result["heatmap_path"] = output_path

    return result