"""
Diagnosis Service — PlantVision Web

Reusable service layer wrapping existing Python diagnosis logic
with typed interfaces, proper path handling, and model caching.

Public API:
- DiagnosisService: Main service class
- DiagnosisResult, AnalysisResult: Typed result structures
- load_image: Helper for image preprocessing
"""

from .service import DiagnosisService
from .types import DiagnosisResult, AnalysisResult, Top3Prediction

__all__ = [
    'DiagnosisService',
    'DiagnosisResult',
    'AnalysisResult',
    'Top3Prediction',
]
