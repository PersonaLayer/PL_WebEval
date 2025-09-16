from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class UXProfile:
    id: str
    name: str
    description: str
    recommended_actions: str
    category: str

@dataclass
class TestCase:
    website: str
    task_goal: str
    task_steps: str
    success_criteria: str
    llm_model: str
    ux_profile: str = ""

@dataclass
class BotDetectionResult:
    is_bot_detected: bool
    confidence_score: float
    detection_type: str
    recommended_action: str
    click_coordinates: Optional[Tuple[int, int]] = None
    alternative_strategy: str = ""
    llm_reasoning: str = ""
    technical_observations: str = ""
    confidence_factors: str = ""
    analysis_timestamp: str = ""

@dataclass
class HomepageMetrics:
    load_time: float = 0.0
    screenshot_size_bytes: int = 0
    html_size_bytes: int = 0
    elements_count: int = 0
    interactive_elements_count: int = 0
    accessibility_score: float = 0.0
    visual_complexity_score: float = 0.0
    color_contrast_issues: int = 0
    text_readability_score: float = 0.0
    adaptation_effectiveness_score: float = 0.0

@dataclass
class ExpertAnalysis:
    accessibility_expert: Dict = field(default_factory=dict)
    wcag_expert: Dict = field(default_factory=dict)
    ux_expert: Dict = field(default_factory=dict)
    visual_critic: Dict = field(default_factory=dict)
    css_analysis: Dict = field(default_factory=dict)
    js_analysis: Dict = field(default_factory=dict)

@dataclass
class AdaptationScore:
    score: int
    reasoning: str
    specific_improvements: List[str]
    missed_opportunities: List[str]
    wcag_compliance_score: float
    accessibility_impact: str
    user_experience_impact: str
    profile_alignment: str
    implementation_quality: str

@dataclass
class VisualComparison:
    baseline_screenshot_path: str
    adapted_screenshot_path: str
    baseline_screenshot_b64: str
    adapted_screenshot_b64: str
    comparison_analysis: str
    visual_differences: List[Dict]
    improvement_score: float
    scientific_observations: str
    expert_analysis: ExpertAnalysis
    adaptation_score: AdaptationScore
    visual_indicators: List[Dict]
    detailed_notes: str

@dataclass
class LLMMetrics:
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    response_time: float = 0.0
    analysis_type: str = ""
    timestamp: str = ""

@dataclass
class HomepageResult:
    test_case: TestCase
    status: str
    execution_time: float
    baseline_screenshot: str
    adapted_screenshot: str
    baseline_snapshot: str
    adapted_snapshot: str
    baseline_metrics: HomepageMetrics
    adapted_metrics: Optional[HomepageMetrics] = None
    ux_adaptations: List[Dict] = field(default_factory=list)
    bot_detection_encountered: bool = False
    bot_detection_result: Optional[BotDetectionResult] = None
    visual_comparison: Optional[VisualComparison] = None
    error_message: Optional[str] = None
    llm_metrics: List[LLMMetrics] = field(default_factory=list)
    test_timestamp: str = ""
    detailed_analysis: Dict = field(default_factory=dict)