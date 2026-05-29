from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from ultralytics import YOLO


class YoloService:

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        try:
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect(
        self,
        img_path: str,
        confidence: float = 0.25,
        iou: float = 0.45
    ) -> Dict:

        img_path_obj = Path(img_path)

        if not img_path_obj.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        try:
            results = self.model.predict(
                source=str(img_path),
                conf=confidence,
                iou=iou,
                verbose=False
            )

            if not results:
                return self._empty_result()

            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                return self._empty_result()

            boxes = result.boxes.xywh.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            img = cv2.imread(str(img_path))
            img_height, img_width = img.shape[:2] if img is not None else (0, 0)

            detection_percentage = self._calculate_coverage(
                boxes,
                img_width,
                img_height
            )

            return {
                "boxes": boxes.tolist(),
                "confidences": confidences.tolist(),
                "classes": class_ids.tolist(),
                "num_detections": len(boxes),
                "detection_percentage": detection_percentage,
                "img_width": img_width,
                "img_height": img_height
            }

        except Exception as e:
            raise RuntimeError(f"YOLO detection failed: {e}")

    def detect_and_annotate(
        self,
        img_path: str,
        output_path: str,
        confidence: float = 0.25,
        iou: float = 0.45
    ) -> Dict:
        """
        Run YOLO detection, save annotated image, and return full result dict
        including detection_percentage and annotated_path.
        """
        img_path_obj = Path(img_path)

        if not img_path_obj.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        try:
            results = self.model.predict(
                source=str(img_path),
                conf=confidence,
                iou=iou,
                verbose=False
            )

            if not results:
                return self._empty_result()

            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                return self._empty_result()

            annotated_frame = result.plot()

            success = cv2.imwrite(output_path, annotated_frame)
            if not success:
                raise RuntimeError(f"Failed to save annotated image to {output_path}")

            boxes = result.boxes.xywh.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            img = cv2.imread(str(img_path))
            img_height, img_width = img.shape[:2] if img is not None else (0, 0)

            detection_percentage = self._calculate_coverage(
                boxes,
                img_width,
                img_height
            )

            return {
                "boxes": boxes.tolist(),
                "confidences": confidences.tolist(),
                "classes": class_ids.tolist(),
                "num_detections": len(boxes),
                "detection_percentage": detection_percentage,
                "img_width": img_width,
                "img_height": img_height,
                "annotated_path": str(Path(output_path).resolve())
            }

        except Exception as e:
            raise RuntimeError(f"Failed to annotate image: {e}")

    def _empty_result(self):
        return {
            "boxes": [],
            "confidences": [],
            "classes": [],
            "num_detections": 0,
            "detection_percentage": 0.0,
            "annotated_path": None
        }

    def _calculate_coverage(
        self,
        boxes: np.ndarray,
        img_width: int,
        img_height: int
    ) -> float:

        if len(boxes) == 0 or img_width == 0 or img_height == 0:
            return 0.0

        total_area = img_width * img_height
        covered_area = 0.0

        for box in boxes:
            _, _, w, h = box
            covered_area += (w * h)

        coverage = min(covered_area / total_area, 1.0) * 100
        return round(coverage, 2)