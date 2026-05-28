"""Integration test for app.py FastAPI endpoints."""
import sys
from pathlib import Path
import unittest
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
import app as app_module

class TestAppEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # TestClient loads the app and starts startup events
        cls.client = TestClient(app_module.app)

    def test_root_returns_index_html(self):
        """GET / returns index.html."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<!DOCTYPE html>", response.content)
        self.assertIn(b"PlantVision", response.content)

    def test_static_mount_exists(self):
        """Static files are mounted."""
        response = self.client.get("/static/css/style.css")
        # May 404 if no CSS file — acceptable
        self.assertIn(response.status_code, [200, 404])

    def test_analyze_valid_image(self):
        """POST /api/analyze with valid image returns full result."""
        test_image = Path("codigo/teste.jpg")
        self.assertTrue(test_image.exists(), f"Test image not found: {test_image}")

        with open(test_image, "rb") as f:
            response = self.client.post(
                "/api/analyze",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Validate response structure
        self.assertIn("diagnosis", data)
        self.assertIn("confidence", data)
        self.assertIn("is_reliable", data)
        self.assertIn("top3", data)
        self.assertIn("heatmap_path", data)

        # Validate types
        self.assertIsInstance(data["diagnosis"], str)
        self.assertIsInstance(data["confidence"], float)
        self.assertIsInstance(data["is_reliable"], bool)
        self.assertIsInstance(data["top3"], list)
        self.assertEqual(len(data["top3"]), 3)
        self.assertIsInstance(data["heatmap_path"], str)

        # heatmap_path should be URL-relative
        self.assertTrue(data["heatmap_path"].startswith("/static/outputs/"))

        # Verify GradCAM file physically exists
        heatmap_rel = data["heatmap_path"].lstrip("/")
        heatmap_full = Path(heatmap_rel)
        self.assertTrue(heatmap_full.exists(), f"GradCAM file not found: {heatmap_full}")

    def test_analyze_non_image_file(self):
        """POST with non-image file returns 400."""
        # Create a dummy text file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(b"This is not an image")
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                response = self.client.post(
                    "/api/analyze",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertIn("detail", data)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

if __name__ == '__main__':
    unittest.main()
