"""
engine – Neuro-Symbolic Medical Expert System Package
=====================================================
Public API re-exported for convenience.
"""

from .config import PROJECT_ROOT, GEMINI_MODEL
from .knowledge_base import KnowledgeBase
from .neural_layer import GeminiNeuralLayer
from .unification import UnificationLayer
from .inference import InferenceEngine
from .explanation import ExplanationFacility
from .orchestrator import run

__all__ = [
    "run",
    "KnowledgeBase",
    "GeminiNeuralLayer",
    "UnificationLayer",
    "InferenceEngine",
    "ExplanationFacility",
    "PROJECT_ROOT",
    "GEMINI_MODEL",
]

