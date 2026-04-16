from typing import TypedDict, List


Top3Prediction = TypedDict('Top3Prediction', {
    'class': str,
    'prob': float
})


class DiagnosisResult(TypedDict):
    """Structured diagnosis result from plant_model.diagnose_plant."""
    diagnosis: str
    confidence: float
    is_reliable: bool
    top3: List[Top3Prediction]


class AnalysisResult(DiagnosisResult):
    """Full analysis result including GradCAM output path."""
    heatmap_path: str
