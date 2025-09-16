"""
PersonaLayer WebEval - Profile-Informed Web Personalization Evaluation Tool

This module provides automated evaluation of web interfaces with persona-specific adaptations
for the PersonaLayer research project on enhanced user experience and accessibility.
"""

__version__ = "1.0.0"
__author__ = "PersonaLayer Research Team"

from .data_models import (
    UXProfile,
    TestCase,
    BotDetectionResult,
    HomepageMetrics,
    ExpertAnalysis,
    AdaptationScore,
    VisualComparison,
    LLMMetrics,
    HomepageResult
)

from .evaluator import WebEvaluator
from .llm_client import OpenRouterClient

__all__ = [
    "WebEvaluator",
    "OpenRouterClient",
    "UXProfile",
    "TestCase",
    "BotDetectionResult",
    "HomepageMetrics",
    "ExpertAnalysis",
    "AdaptationScore",
    "VisualComparison",
    "LLMMetrics",
    "HomepageResult"
]