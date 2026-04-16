"""
Smoke test — end-to-end verification of DiagnosisService.

Tests that the service can be instantiated and produces valid output
using the repository's existing test assets (codigo/teste.jpg, codigo/plant_model.h5).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure src/ is on path for imports
SRC_DIR = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from diagnosis.service import DiagnosisService
from diagnosis.types import DiagnosisResult, AnalysisResult


# Test asset paths (relative to repo root)
CODIGO_DIR = Path(__file__).parent.parent.parent / "codigo"
TEST_IMAGE = CODIGO_DIR / "teste.jpg"
MODEL_FILE = CODIGO_DIR / "plant_model.h5"

# Class names matching the trained model
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus"
]


class TestDiagnosisService(unittest.TestCase):
    """Smoke tests for DiagnosisService end-to-end functionality."""

    def test_service_initialization(self):
        """Service loads model without error."""
        service = DiagnosisService(
            model_path=str(MODEL_FILE),
            class_names=CLASS_NAMES,
            confidence_threshold=0.6
        )
        self.assertIsNotNone(service.model)
        self.assertEqual(len(service.class_names), 4)

    def test_diagnose_shape_validation(self):
        """Diagnose rejects invalid tensor shapes."""
        service = DiagnosisService(
            model_path=str(MODEL_FILE),
            class_names=CLASS_NAMES
        )
        wrong_shape = np.zeros((1, 224, 224))  # missing channel
        with self.assertRaises(ValueError) as ctx:
            service.diagnose(wrong_shape)
        self.assertIn("Invalid img_tensor shape", str(ctx.exception))

    def test_diagnose_returns_typed_result(self):
        """Diagnose returns properly structured result."""
        service = DiagnosisService(
            model_path=str(MODEL_FILE),
            class_names=CLASS_NAMES
        )
        # Create preprocessed tensor from the test image
        import cv2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        img = cv2.imread(str(TEST_IMAGE))
        self.assertIsNotNone(img, f"Test image not found: {TEST_IMAGE}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0)
        img_tensor = preprocess_input(img_array.astype(np.float32))

        result = service.diagnose(img_tensor)

        # Validate shape
        self.assertIsInstance(result, dict)
        self.assertIn("diagnosis", result)
        self.assertIn("confidence", result)
        self.assertIn("is_reliable", result)
        self.assertIn("top3", result)
        self.assertIsInstance(result["top3"], list)
        self.assertEqual(len(result["top3"]), 3)
        for item in result["top3"]:
            self.assertIn("class", item)
            self.assertIn("prob", item)
        self.assertIn(result["diagnosis"], CLASS_NAMES)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertIsInstance(result["is_reliable"], bool)

    def test_analyze_full_pipeline(self):
        """Full analyze() pipeline produces GradCAM output and complete result."""
        import tempfile

        service = DiagnosisService(
            model_path=str(MODEL_FILE),
            class_names=CLASS_NAMES
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = service.analyze(str(TEST_IMAGE), output_dir=tmpdir)

            # Validate AnalysisResult extended structure
            self.assertIsInstance(result, dict)
            self.assertIn("heatmap_path", result)
            heatmap_path = Path(result["heatmap_path"])
            self.assertTrue(heatmap_path.exists(),
                            f"GradCAM output not created: {heatmap_path}")
            self.assertEqual(heatmap_path.suffix, ".jpg")
            self.assertIn("_gradcam.jpg", heatmap_path.name)

            # Validate core fields
            self.assertIn(result["diagnosis"], CLASS_NAMES)
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)
            self.assertIsInstance(result["is_reliable"], bool)
            self.assertEqual(len(result["top3"]), 3)

    def test_analyze_default_output_location(self):
        """Analyze saves GradCAM beside input when output_dir not provided."""
        import tempfile
        import shutil

        service = DiagnosisService(
            model_path=str(MODEL_FILE),
            class_names=CLASS_NAMES
        )

        # Use a temporary copy of the test image in a temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_img = Path(tmpdir) / "test_copy.jpg"
            shutil.copy(TEST_IMAGE, tmp_img)

            result = service.analyze(str(tmp_img))

            heatmap_path = Path(result["heatmap_path"])
            self.assertEqual(heatmap_path.parent, tmp_img.parent)
            self.assertEqual(heatmap_path.name, "test_copy_gradcam.jpg")
            self.assertTrue(heatmap_path.exists())


if __name__ == '__main__':
    unittest.main()
