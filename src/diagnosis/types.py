from typing import TypedDict, List, Optional, Dict, Any


Top3Prediction = TypedDict('Top3Prediction', {
    'class': str,
    'prob': float
})


class YoloResult(TypedDict, total=False):
    """Structured YOLO detection result."""
    boxes: List[List[float]]
    confidences: List[float]
    classes: List[float]
    num_detections: int
    detection_percentage: float
    annotated_path: Optional[str]
    img_width: int
    img_height: int


class DiagnosisResult(TypedDict):
    """Structured diagnosis result from plant_model.diagnose_plant."""
    diagnosis: str
    confidence: float
    is_reliable: bool
    top3: List[Top3Prediction]


class AnalysisResult(TypedDict, total=False):
    """Full analysis result including GradCAM output path, YOLO detections, and frontend summary fields."""
    diagnosis: str
    confidence: float
    is_reliable: bool
    top3: List[Top3Prediction]
    heatmap_path: str
    yolo: Optional[YoloResult]
    condition: str
    recommendation: str