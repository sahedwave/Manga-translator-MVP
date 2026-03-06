from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from google import genai
from PIL import Image


def _rect_polygon(bbox: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]:
    x1, y1, x2, y2 = bbox
    return [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]


class GeminiMangaAssistant:
    def __init__(self, api_key: str | None = None, model: str = "gemini-3-flash-preview") -> None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("Gemini API key is required for AI assist.")
        self._client = genai.Client(api_key=resolved_key)
        self._model = model

    def _extract_json(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("Gemini returned an empty response.")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
        raise ValueError(f"Gemini returned non-JSON output: {text[:240]}")

    def review_page(
        self,
        image_path: Path,
        image_size: tuple[int, int],
        local_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        width, height = image_size
        image = Image.open(image_path)
        prompt = (
            "You review manga pages for translation.\n"
            "Detect every dialogue or narration text region that should be translated.\n"
            "Ignore watermarks, publisher logos, support banners, and website addresses.\n"
            "Prefer full text blocks or full speech bubbles, not fragmented lines.\n"
            "Preserve reading order from top-to-bottom then left-to-right.\n"
            f"Image size: width={width}, height={height}.\n"
            "Use the existing OCR records as hints only. You may correct, merge, remove, or add records.\n"
            "Return JSON only with this exact shape:\n"
            "{\n"
            '  "records": [\n'
            "    {\n"
            '      "source_text": "original text",\n'
            '      "translated_text": "natural English dialogue",\n'
            '      "bbox": [x1, y1, x2, y2]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            f"Existing OCR records:\n{json.dumps(local_records, ensure_ascii=False)}"
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=[prompt, image],
            config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "records": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source_text": {"type": "string"},
                                    "translated_text": {"type": "string"},
                                    "bbox": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 4,
                                        "maxItems": 4,
                                    },
                                },
                                "required": ["source_text", "translated_text", "bbox"],
                            },
                        }
                    },
                    "required": ["records"],
                },
            },
        )
        payload = self._extract_json(response.text or "")
        raw_records = payload.get("records", [])
        normalized: list[dict[str, Any]] = []
        for item in raw_records:
            bbox_value = item.get("bbox", [])
            if not isinstance(bbox_value, list) or len(bbox_value) != 4:
                continue
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_value]
            except (TypeError, ValueError):
                continue
            x1 = max(0, min(width, x1))
            y1 = max(0, min(height, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            source_text = str(item.get("source_text", "")).strip()
            translated_text = str(item.get("translated_text", "")).strip()
            if not source_text or not translated_text:
                continue

            bbox = (x1, y1, x2, y2)
            normalized.append(
                {
                    "source_text": source_text,
                    "translated_text": translated_text,
                    "confidence": 1.0,
                    "bbox": bbox,
                    "mask_polygons": _rect_polygon(bbox),
                }
            )
        return normalized
