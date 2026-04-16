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


class DiagnosisService:
    """
    Reusable service wrapper for plant disease diagnosis.

    Wraps existing codigo/plant_model.py and codigo/plant_gradcam.py
    with proper path handling, model caching, and typed interfaces.
    """

    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        confidence_threshold: float = 0.6
    ):
        """
        Initialize the diagnosis service.

        Args:
            model_path: Path to .h5 model file. Can be relative to this file's location.
            class_names: List of class names matching model output order.
            confidence_threshold: Confidence threshold for reliability (default 0.6).

        Raises:
            FileNotFoundError: If model file does not exist.
            ValueError: If class_names is empty.
            RuntimeError: If model fails to load.
        """
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold

        # Resolve model path relative to this file, not CWD
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            # Resolve relative to the src/diagnosis/ directory
            model_path_obj = Path(__file__).parent / model_path

        # Normalize to absolute path
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
        Run full analysis including GradCAM on an image file.

        This wraps codigo/plant_gradcam.py::run_full_analysis with improved
        output naming (_gradcam.jpg instead of _diagnosis.jpg).

        Args:
            img_path: Path to input image file.
            output_dir: Directory to save GradCAM output. If None, saves beside input.

        Returns:
            Full analysis result including heatmap_path.

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
            output_path = output_dir_obj / f"{img_path_obj.stem}_gradcam.jpg"
        else:
            output_path = img_path_obj.parent / f"{img_path_obj.stem}_gradcam.jpg"

        # Use custom version that respects output_path
        result = self._run_full_analysis_custom(
            str(img_path_obj),
            output_path=str(output_path)
        )

        return result

    def _run_full_analysis_custom(
        self,
        img_path: str,
        output_path: str
    ) -> AnalysisResult:
        """
        Custom full analysis that writes GradCAM to specified output_path.

        This replicates codigo/plant_gradcam.py::run_full_analysis logic but
        uses the provided output_path instead of deriving it from img_path.
        """
        # 1. Load and preprocess
        img_original, img_tensor = load_and_preprocess(img_path)

        # 2. Diagnosis (use self.model and self.class_names)
        result = self.diagnose(img_tensor)

        # 3. Get predicted class index
        class_idx = self.class_names.index(result['diagnosis'])

        # 4. GradCAM
        heatmap = compute_gradcam(self.model, img_tensor, class_idx)

        # 5. Overlay
        overlay = overlay_heatmap(img_original, heatmap)

        # 6. Save to specified output path
        cv2.imwrite(output_path, overlay)

        # 7. Build AnalysisResult
        analysis_result: AnalysisResult = {
            "diagnosis": result["diagnosis"],
            "confidence": result["confidence"],
            "is_reliable": result["is_reliable"],
            "top3": result["top3"],
            "heatmap_path": output_path
        }

        return analysis_result
