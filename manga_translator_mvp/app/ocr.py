from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR


Point = Tuple[int, int]


@dataclass(slots=True)
class OCRLine:
    text: str
    confidence: float
    polygon: List[Point]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)


class MangaOCR:
    def __init__(self, languages: Sequence[str] | None = None) -> None:
        self._reader = RapidOCR()

    def _run_reader(self, image_rgb: np.ndarray) -> List[OCRLine]:
        result, _ = self._reader(image_rgb)
        lines: List[OCRLine] = []
        if not result:
            return lines
        for item in result:
            polygon, text, confidence = item[0], item[1], item[2]
            clean = text.strip()
            if not clean:
                continue
            points = [(int(x), int(y)) for x, y in polygon]
            lines.append(OCRLine(text=clean, confidence=float(confidence), polygon=points))
        return lines

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def _prepare_variants(self, image_rgb: np.ndarray, exhaustive: bool) -> List[tuple[np.ndarray, float]]:
        base = image_rgb.astype(np.uint8, copy=False)
        variants: List[tuple[np.ndarray, float]] = [(base, 1.0)]
        if not exhaustive:
            return variants

        gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape[:2]
        scale = 2.0 if max(height, width) < 2200 else 1.5

        upscaled = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append((upscaled, scale))

        contrast = cv2.convertScaleAbs(gray, alpha=1.7, beta=8)
        variants.append((self._ensure_rgb(contrast), 1.0))

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((self._ensure_rgb(otsu), 1.0))

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        variants.append((self._ensure_rgb(adaptive), 1.0))

        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
        variants.append((self._ensure_rgb(sharpened), 1.0))
        return variants

    def _rescale_line(self, line: OCRLine, scale: float) -> OCRLine:
        if scale == 1.0:
            return line
        polygon = [(int(round(x / scale)), int(round(y / scale))) for x, y in line.polygon]
        return OCRLine(text=line.text, confidence=line.confidence, polygon=polygon)

    def _intersection_area(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    def _bbox_area(self, bbox: Tuple[int, int, int, int]) -> int:
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    def _normalize_text_key(self, text: str) -> str:
        return "".join(ch for ch in text.upper() if ch.isalnum())

    def _is_duplicate(self, candidate: OCRLine, accepted: List[OCRLine]) -> bool:
        candidate_key = self._normalize_text_key(candidate.text)
        for existing in accepted:
            existing_key = self._normalize_text_key(existing.text)
            overlap = self._intersection_area(candidate.bbox, existing.bbox)
            if overlap == 0:
                continue
            smaller = min(self._bbox_area(candidate.bbox), self._bbox_area(existing.bbox))
            if smaller == 0:
                continue
            if overlap / smaller < 0.55:
                continue
            if candidate_key == existing_key:
                return True
            if candidate.confidence <= existing.confidence and len(candidate_key) <= len(existing_key):
                return True
        return False

    def extract_lines(self, image_rgb: np.ndarray, exhaustive: bool = False) -> List[OCRLine]:
        variants = self._prepare_variants(image_rgb, exhaustive=exhaustive)
        merged: List[OCRLine] = []
        for variant_image, scale in variants:
            for line in self._run_reader(variant_image):
                adjusted = self._rescale_line(line, scale)
                if not self._is_duplicate(adjusted, merged):
                    merged.append(adjusted)
        merged.sort(key=lambda line: (line.bbox[1], line.bbox[0], -line.confidence))
        return merged
