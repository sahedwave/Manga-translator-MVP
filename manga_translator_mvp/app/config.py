from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class AppConfig:
    input_path: Path
    output_path: Path
    translator: str = "mock"
    source_lang: str = "auto"
    target_lang: str = "en"
    font_path: Optional[Path] = None
    min_confidence: float = 0.2
    bubble_brightness_threshold: float = 155.0
    bubble_variance_threshold: float = 12000.0
    padding_px: int = 6
    min_bubble_area_ratio: float = 0.002
    max_bubble_area_ratio: float = 0.25
    text_region_min_area_ratio: float = 0.00015
    text_region_max_area_ratio: float = 0.18
    coverage_overlap_threshold: float = 0.35
    coverage_confidence_threshold: float = 0.45
    ai_assist: bool = False
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3-flash-preview"
