import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf
import cv2

# Add codigo/ to path to import existing modules
CODIGO_DIR = Path(__file__).parent.parent.parent / "codigo"
sys.path.insert(0, str(CODIGO_DIR))

try:
    from plant_model import diagnose_plant
    from plant_gradcam import load_and_preprocess, compute_gradcam, overlay_heatmap
except ImportError as e:
    raise ImportError(f"Failed to import from codigo/: {e}. codigo/ path: {CODIGO_DIR}")

from .types import DiagnosisResult, AnalysisResult
from .yolo_service import YoloService


class DiagnosisService:
    """
    Reusable service wrapper for plant disease diagnosis.

    Wraps existing codigo/plant_model.py and codigo/plant_gradcam.py
    with proper path handling, model caching, and typed interfaces.
    Optionally integrates YoloService for lesion detection.
    """

    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        confidence_threshold: float = 0.6,
        yolo_model_path: Optional[str] = None
    ):
        """
        Initialize the diagnosis service.

        Args:
            model_path: Path to .h5 model file. Can be relative to this file's location.
            class_names: List of class names matching model output order.
            confidence_threshold: Confidence threshold for reliability (default 0.6).
            yolo_model_path: Optional path to YOLO .pt model for lesion detection.

        Raises:
            FileNotFoundError: If model file does not exist.
            ValueError: If class_names is empty.
            RuntimeError: If model fails to load.
        """
        if not class_names:
            raise ValueError("class_names must not be empty.")

        self.class_names = class_names
        self.confidence_threshold = confidence_threshold

        # Resolve model path relative to this file, not CWD
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            model_path_obj = Path(__file__).parent / model_path

        model_path_abs = model_path_obj.resolve()

        if not model_path_abs.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path_abs}. "
                f"Original path: {model_path}"
            )

        try:
            self.model = tf.keras.models.load_model(str(model_path_abs))
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path_abs}: {e}")

        # Optional YOLO service
        self.yolo_service: Optional[YoloService] = None
        if yolo_model_path:
            yolo_path_obj = Path(yolo_model_path)
            if yolo_path_obj.exists():
                try:
                    self.yolo_service = YoloService(str(yolo_path_obj))
                    print(f"✅ YoloService loaded from: {yolo_path_obj}")
                except Exception as e:
                    print(f"⚠️  YoloService failed to load (YOLO disabled): {e}")
            else:
                print(f"⚠️  YOLO model not found at {yolo_path_obj} — YOLO disabled.")

    def diagnose(self, img_tensor: np.ndarray) -> DiagnosisResult:
        """
        Run diagnosis on a preprocessed image tensor.

        Args:
            img_tensor: Preprocessed image tensor of shape (1, 224, 224, 3).

        Returns:
            Typed diagnosis result with diagnosis, confidence, reliability, and top-3.

        Raises:
            ValueError: If img_tensor shape is incorrect.
            RuntimeError: If model inference fails.
        """
        if img_tensor.shape != (1, 224, 224, 3):
            raise ValueError(
                f"Invalid img_tensor shape: {img_tensor.shape}. "
                f"Expected (1, 224, 224, 3)."
            )

        try:
            result = diagnose_plant(
                self.model,
                img_tensor,
                self.class_names,
                self.confidence_threshold
            )
            return result
        except Exception as e:
            raise RuntimeError(f"Diagnosis inference failed: {e}")

    def analyze(self, img_path: str, output_dir: Optional[str] = None) -> AnalysisResult:
        """
        Run full analysis including GradCAM (and optionally YOLO) on an image file.

        Args:
            img_path: Path to input image file.
            output_dir: Directory to save GradCAM and YOLO outputs. If None, saves beside input.

        Returns:
            Full analysis result including heatmap_path and optional yolo detections.

        Raises:
            FileNotFoundError: If input image does not exist.
            ValueError: If image cannot be loaded.
            RuntimeError: If analysis pipeline fails.
        """
        img_path_obj = Path(img_path)

        if not img_path_obj.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Resolve output directory
        if output_dir:
            output_dir_obj = Path(output_dir)
            output_dir_obj.mkdir(parents=True, exist_ok=True)
            gradcam_output_path = output_dir_obj / f"{img_path_obj.stem}_gradcam.jpg"
            yolo_output_path = output_dir_obj / f"{img_path_obj.stem}_yolo.jpg"
        else:
            gradcam_output_path = img_path_obj.parent / f"{img_path_obj.stem}_gradcam.jpg"
            yolo_output_path = img_path_obj.parent / f"{img_path_obj.stem}_yolo.jpg"

        result = self._run_full_analysis_custom(
            str(img_path_obj),
            gradcam_output_path=str(gradcam_output_path),
            yolo_output_path=str(yolo_output_path)
        )

        return result

    def _run_full_analysis_custom(
        self,
        img_path: str,
        gradcam_output_path: str,
        yolo_output_path: str
    ) -> AnalysisResult:
        """
        Full analysis: GradCAM classification + optional YOLO lesion detection.

        Writes GradCAM heatmap to gradcam_output_path.
        If YoloService is available, writes annotated image to yolo_output_path.
        """
        # 1. Load and preprocess
        img_original, img_tensor = load_and_preprocess(img_path)

        # 2. Diagnosis
        result = self.diagnose(img_tensor)

        # 3. Get predicted class index
        class_idx = self.class_names.index(result['diagnosis'])

        # 4. GradCAM
        heatmap = compute_gradcam(self.model, img_tensor, class_idx)

        # 5. Overlay and save
        overlay = overlay_heatmap(img_original, heatmap)
        success = cv2.imwrite(gradcam_output_path, overlay)
        if not success:
            raise RuntimeError(f"Failed to save GradCAM output to {gradcam_output_path}")

        # 6. Build AnalysisResult base
        analysis_result: AnalysisResult = {
            "diagnosis": result["diagnosis"],
            "confidence": result["confidence"],
            "is_reliable": result["is_reliable"],
            "top3": result["top3"],
            "heatmap_path": str(Path(gradcam_output_path).resolve()),
            "yolo": None
        }

        # 7. Optional YOLO detection
        if self.yolo_service is not None:
            try:
                yolo_data = self.yolo_service.detect_and_annotate(
                    img_path=img_path,
                    output_path=yolo_output_path
                )
                # Attach annotated_path from detect_and_annotate result
                analysis_result["yolo"] = {
                    "boxes": yolo_data.get("boxes", []),
                    "confidences": yolo_data.get("confidences", []),
                    "classes": yolo_data.get("classes", []),
                    "num_detections": yolo_data.get("num_detections", 0),
                    "detection_percentage": yolo_data.get("detection_percentage", 0.0),
                    "annotated_path": yolo_data.get("annotated_path"),
                }
                print(f"✅ YOLO: {yolo_data.get('num_detections', 0)} detecção(ões) encontradas.")
            except Exception as e:
                print(f"⚠️  YOLO detection failed (skipped): {e}")
                analysis_result["yolo"] = None

        return analysis_result