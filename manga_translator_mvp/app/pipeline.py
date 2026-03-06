from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .config import AppConfig
from .gemini_assist import GeminiMangaAssistant
from .image_ops import (
    crop_image,
    detect_bubble_regions,
    detect_text_regions,
    draw_text_block,
    inpaint_text,
    load_image_rgb,
    polygon_mask,
    region_stats,
    save_image_rgb,
    TextRegionCandidate,
)
from .ocr import MangaOCR, OCRLine
from .serialization import to_builtin
from .translate import build_translator


@dataclass(slots=True)
class TranslationRecord:
    source_text: str
    translated_text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    mask_polygons: list[list[tuple[int, int]]]


@dataclass(slots=True)
class CoverageCandidate:
    source_text: str
    translated_text: str
    confidence: float
    detector_score: float
    bbox: tuple[int, int, int, int]
    mask_polygons: list[list[tuple[int, int]]]


@dataclass(slots=True)
class AnalysisResult:
    records: list[TranslationRecord]
    coverage_candidates: list[CoverageCandidate]


class MangaTranslationPipeline:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._ocr = MangaOCR(["en"])
        self._use_gemini_assist = config.ai_assist and config.translator.lower().strip() == "gemini"
        self._translator = build_translator(
            name=config.translator,
            source_lang=config.source_lang,
            target_lang=config.target_lang,
            api_key=config.gemini_api_key,
            model=config.gemini_model,
        )
        self._ai_assistant = (
            GeminiMangaAssistant(api_key=config.gemini_api_key, model=config.gemini_model)
            if self._use_gemini_assist
            else None
        )

    def _translate_text(self, source_text: str) -> str:
        if self._use_gemini_assist:
            return source_text
        return self._translator.translate(source_text)

    def _is_likely_bubble_text(self, line: OCRLine, image_rgb) -> bool:
        if line.confidence < self._config.min_confidence:
            return False
        mean, variance = region_stats(image_rgb, line.bbox)
        return mean >= self._config.bubble_brightness_threshold and variance <= self._config.bubble_variance_threshold

    def _is_translatable_text(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        alpha_count = sum(ch.isalpha() for ch in stripped)
        return alpha_count >= 2

    def _clean_text(self, text: str) -> str:
        cleaned = " ".join(text.split())
        cleaned = re.sub(r"\s+([,?.!])", r"\1", cleaned)
        cleaned = re.sub(r"([A-Z])([A-Z]{2,})HAL\b", r"\1\2 HAL", cleaned)
        cleaned = cleaned.replace("MENGATAKANHAL", "MENGATAKAN HAL")
        cleaned = cleaned.replace("SEHARLSNYA", "SEHARUSNYA")
        cleaned = cleaned.replace("KONDSIMU", "KONDISIMU")
        cleaned = cleaned.replace("PARKJINYEONG", "PARK JINYEONG")
        cleaned = cleaned.replace("PARKJINYEONG-SSI", "PARK JINYEONG-SSI")
        cleaned = cleaned.replace("COPY-KL", "COPY-KU")
        cleaned = cleaned.replace("BOCAH-BOCAHITU", "BOCAH-BOCAH ITU")
        return cleaned

    def _looks_like_ocr_noise(self, text: str, confidence: float) -> bool:
        stripped = text.strip().upper()
        if not stripped:
            return True

        compact = "".join(ch for ch in stripped if not ch.isspace())
        if not compact:
            return True

        non_alpha_ratio = sum(not ch.isalpha() for ch in compact) / max(len(compact), 1)
        if non_alpha_ratio > 0.18 and confidence < 0.92:
            return True

        tokens = re.findall(r"[A-Z-]+", stripped)
        if any(re.search(r"[BCDFGHJKLMNPQRSTVWXYZ]{4,}", token.replace("-", "")) for token in tokens) and confidence < 0.93:
            return True

        long_tokens = [token.replace("-", "") for token in tokens if len(token.replace("-", "")) >= 4]
        if long_tokens:
            no_vowel_tokens = [token for token in long_tokens if not re.search(r"[AEIOU]", token)]
            if len(no_vowel_tokens) / len(long_tokens) >= 0.6 and confidence < 0.95:
                return True

            low_vowel_tokens = [
                token for token in long_tokens
                if (sum(ch in "AEIOU" for ch in token) / max(len(token), 1)) < 0.26
            ]
            if len(low_vowel_tokens) / len(long_tokens) >= 0.75 and confidence < 0.9:
                return True

        if stripped.startswith(".") and confidence < 0.95:
            return True

        return False

    def _bbox_area(self, bbox: tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _expand_bbox(
        self,
        bbox: tuple[int, int, int, int],
        image_shape: tuple[int, int, int],
        outer_bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        text_height = max(1, y2 - y1)
        pad_x = min(90, max(24, int(text_height * 1.35)))
        pad_y = min(80, max(30, int(text_height * 0.95)))

        img_h, img_w = image_shape[:2]
        min_x, min_y, max_x, max_y = (0, 0, img_w, img_h)
        if outer_bbox is not None:
            min_x = max(0, outer_bbox[0])
            min_y = max(0, outer_bbox[1])
            max_x = min(img_w, outer_bbox[2])
            max_y = min(img_h, outer_bbox[3])

        new_x1 = max(min_x, x1 - pad_x)
        new_y1 = max(min_y, y1 - pad_y)
        new_x2 = min(max_x, x2 + pad_x)
        new_y2 = min(max_y, y2 + pad_y)
        return new_x1, new_y1, new_x2, new_y2

    def _combined_bbox_from_lines(
        self,
        lines: List[OCRLine],
        image_shape: tuple[int, int, int],
        outer_bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[int, int, int, int]:
        x1 = min(line.bbox[0] for line in lines)
        y1 = min(line.bbox[1] for line in lines)
        x2 = max(line.bbox[2] for line in lines)
        y2 = max(line.bbox[3] for line in lines)
        return self._expand_bbox((x1, y1, x2, y2), image_shape, outer_bbox=outer_bbox)

    def _intersection_area(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    def _is_duplicate_record(self, candidate: TranslationRecord, accepted: List[TranslationRecord]) -> bool:
        for existing in accepted:
            overlap = self._intersection_area(candidate.bbox, existing.bbox)
            if overlap == 0:
                continue
            smaller = min(self._bbox_area(candidate.bbox), self._bbox_area(existing.bbox))
            if smaller == 0:
                continue
            overlap_ratio = overlap / smaller
            if overlap_ratio >= 0.75:
                existing_len = len(existing.source_text)
                candidate_len = len(candidate.source_text)
                if candidate_len <= existing_len and candidate.confidence <= existing.confidence:
                    return True
        return False

    def _overlaps_existing_records(
        self,
        candidate_bbox: tuple[int, int, int, int],
        accepted: List[TranslationRecord],
        threshold: float = 0.35,
    ) -> bool:
        for existing in accepted:
            overlap = self._intersection_area(candidate_bbox, existing.bbox)
            if overlap == 0:
                continue
            smaller = min(self._bbox_area(candidate_bbox), self._bbox_area(existing.bbox))
            if smaller == 0:
                continue
            if overlap / smaller >= threshold:
                return True
        return False

    def _overlap_ratio(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        overlap = self._intersection_area(a, b)
        if overlap == 0:
            return 0.0
        smaller = min(self._bbox_area(a), self._bbox_area(b))
        if smaller == 0:
            return 0.0
        return overlap / smaller

    def _line_matches_bbox(self, line: OCRLine, bbox: tuple[int, int, int, int]) -> bool:
        lx1, ly1, lx2, ly2 = line.bbox
        bx1, by1, bx2, by2 = bbox
        line_area = self._bbox_area(line.bbox)
        if line_area == 0:
            return False

        overlap = self._intersection_area(line.bbox, bbox)
        if overlap / line_area >= 0.35:
            return True

        cx = (lx1 + lx2) / 2
        cy = (ly1 + ly2) / 2
        return bx1 <= cx <= bx2 and by1 <= cy <= by2

    def _extract_lines_for_bbox(self, all_lines: List[OCRLine], bbox: tuple[int, int, int, int]) -> List[OCRLine]:
        matched = [
            line
            for line in all_lines
            if self._is_translatable_text(line.text) and self._line_matches_bbox(line, bbox)
        ]
        matched.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
        return matched

    def _fallback_mask_polygon(self, bbox: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]:
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        inset_x = max(8, int(width * 0.18))
        inset_y = max(6, int(height * 0.24))
        mx1 = min(x2 - 1, x1 + inset_x)
        my1 = min(y2 - 1, y1 + inset_y)
        mx2 = max(mx1 + 1, x2 - inset_x)
        my2 = max(my1 + 1, y2 - inset_y)
        return [[(mx1, my1), (mx2, my1), (mx2, my2), (mx1, my2)]]

    def _best_overlap_index(self, candidate_bbox: tuple[int, int, int, int], records: List[TranslationRecord], threshold: float = 0.45) -> int | None:
        best_index: int | None = None
        best_ratio = 0.0
        for index, record in enumerate(records):
            ratio = self._overlap_ratio(candidate_bbox, record.bbox)
            if ratio >= threshold and ratio > best_ratio:
                best_ratio = ratio
                best_index = index
        return best_index

    def _merge_ai_records(
        self,
        local_records: List[TranslationRecord],
        ai_records: List[TranslationRecord],
        image_rgb,
    ) -> List[TranslationRecord]:
        if not ai_records:
            return local_records

        all_lines = self._ocr.extract_lines(image_rgb, exhaustive=True)
        merged = [
            TranslationRecord(
                source_text=record.source_text,
                translated_text=record.translated_text,
                confidence=record.confidence,
                bbox=record.bbox,
                mask_polygons=[list(poly) for poly in record.mask_polygons],
            )
            for record in local_records
        ]

        for ai_record in ai_records:
            overlap_index = self._best_overlap_index(ai_record.bbox, merged)
            if overlap_index is not None:
                local = merged[overlap_index]
                merged[overlap_index] = TranslationRecord(
                    source_text=ai_record.source_text if len(ai_record.source_text) >= len(local.source_text) else local.source_text,
                    translated_text=ai_record.translated_text or local.translated_text,
                    confidence=max(local.confidence, ai_record.confidence),
                    bbox=local.bbox,
                    mask_polygons=local.mask_polygons,
                )
                continue

            region_lines = self._extract_lines_for_bbox(all_lines, ai_record.bbox)
            if region_lines:
                refined_bbox = self._combined_bbox_from_lines(region_lines, image_rgb.shape, outer_bbox=ai_record.bbox)
                refined_polygons = [list(line.polygon) for line in region_lines]
                refined_source = self._clean_text(" ".join(line.text for line in region_lines))
                source_text = refined_source if len(refined_source) >= len(ai_record.source_text) else ai_record.source_text
                confidence = max(ai_record.confidence, sum(line.confidence for line in region_lines) / len(region_lines))
                merged.append(
                    TranslationRecord(
                        source_text=source_text,
                        translated_text=ai_record.translated_text,
                        confidence=confidence,
                        bbox=refined_bbox,
                        mask_polygons=refined_polygons,
                    )
                )
            else:
                merged.append(
                    TranslationRecord(
                        source_text=ai_record.source_text,
                        translated_text=ai_record.translated_text,
                        confidence=ai_record.confidence,
                        bbox=ai_record.bbox,
                        mask_polygons=self._fallback_mask_polygon(ai_record.bbox),
                    )
                )

        return self._dedupe_records(merged)

    def _dedupe_records(self, records: List[TranslationRecord]) -> List[TranslationRecord]:
        records_sorted = sorted(
            records,
            key=lambda record: (
                -(len(record.source_text)),
                -record.confidence,
                record.bbox[1],
                record.bbox[0],
            ),
        )
        accepted: List[TranslationRecord] = []
        for record in records_sorted:
            if not self._is_duplicate_record(record, accepted):
                accepted.append(record)
        accepted.sort(key=lambda record: (record.bbox[1], record.bbox[0]))
        return [
            record
            for record in accepted
            if not self._looks_like_ocr_noise(record.source_text, record.confidence)
        ]

    def _select_lines(self, all_lines: List[OCRLine], image_rgb) -> List[OCRLine]:
        likely_bubbles = [
            line for line in all_lines
            if self._is_translatable_text(line.text) and self._is_likely_bubble_text(line, image_rgb)
        ]
        if likely_bubbles:
            return likely_bubbles
        return [
            line for line in all_lines
            if self._is_translatable_text(line.text) and line.confidence >= self._config.min_confidence
        ]

    def _record_from_lines(
        self,
        lines: List[OCRLine],
        image_shape: tuple[int, int, int],
        outer_bbox: tuple[int, int, int, int] | None = None,
    ) -> TranslationRecord | None:
        if not lines:
            return None
        source_text = self._clean_text(" ".join(line.text for line in lines))
        confidence = sum(line.confidence for line in lines) / len(lines)
        if not source_text or self._looks_like_ocr_noise(source_text, confidence):
            return None
        translated = self._translate_text(source_text)
        return TranslationRecord(
            source_text=source_text,
            translated_text=translated,
            confidence=confidence,
            bbox=self._combined_bbox_from_lines(lines, image_shape, outer_bbox=outer_bbox),
            mask_polygons=[list(line.polygon) for line in lines],
        )

    def _ocr_region_lines(self, image_rgb, bbox: tuple[int, int, int, int]) -> List[OCRLine]:
        crop = crop_image(image_rgb, bbox)
        crop_lines = self._ocr.extract_lines(crop, exhaustive=True)
        adjusted: List[OCRLine] = []
        x1, y1, _, _ = bbox
        for line in crop_lines:
            adjusted_polygon = [(x + x1, y + y1) for x, y in line.polygon]
            adjusted_line = OCRLine(text=line.text, confidence=line.confidence, polygon=adjusted_polygon)
            if self._is_translatable_text(adjusted_line.text):
                adjusted.append(adjusted_line)
        adjusted.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
        return adjusted

    def _candidate_overlaps_regions(
        self,
        candidate_bbox: tuple[int, int, int, int],
        regions: List[TextRegionCandidate],
        threshold: float = 0.4,
    ) -> bool:
        for region in regions:
            if self._overlap_ratio(candidate_bbox, region.bbox) >= threshold:
                return True
        return False

    def _collect_region_proposals(self, image_rgb) -> List[TextRegionCandidate]:
        candidates = detect_text_regions(
            image_rgb,
            min_area_ratio=self._config.text_region_min_area_ratio,
            max_area_ratio=self._config.text_region_max_area_ratio,
        )
        bubbles = detect_bubble_regions(
            image_rgb,
            min_area_ratio=self._config.min_bubble_area_ratio,
            max_area_ratio=self._config.max_bubble_area_ratio,
        )
        for bubble in bubbles:
            if not self._candidate_overlaps_regions(bubble.bbox, candidates):
                candidates.append(
                    TextRegionCandidate(
                        bbox=bubble.bbox,
                        score=0.28,
                        source="bubble",
                    )
                )
        candidates.sort(key=lambda item: (-item.score, item.bbox[1], item.bbox[0]))
        return candidates

    def _build_detected_region_records(
        self,
        image_rgb,
        regions: List[TextRegionCandidate],
    ) -> List[TranslationRecord]:
        records: List[TranslationRecord] = []
        for region in regions:
            region_lines = [
                line
                for line in self._ocr_region_lines(image_rgb, region.bbox)
                if line.confidence >= max(0.1, self._config.min_confidence - 0.1)
            ]
            groups = self._group_fallback_lines(region_lines)
            for group in groups:
                record = self._record_from_lines(group, image_rgb.shape, outer_bbox=region.bbox)
                if record is None:
                    continue
                if self._overlaps_existing_records(record.bbox, records, threshold=0.55):
                    continue
                records.append(record)
        return records

    def _build_page_fallback_records(
        self,
        image_rgb,
        existing_records: List[TranslationRecord],
    ) -> List[TranslationRecord]:
        all_lines = self._ocr.extract_lines(image_rgb, exhaustive=True)
        likely_lines = self._select_lines(all_lines, image_rgb)
        grouped_lines = self._group_fallback_lines(likely_lines)
        fallback_records: List[TranslationRecord] = []
        for group in grouped_lines:
            candidate_bbox = self._combined_bbox_from_lines(group, image_rgb.shape)
            if self._overlaps_existing_records(candidate_bbox, existing_records + fallback_records):
                continue
            record = self._record_from_lines(group, image_rgb.shape)
            if record is None:
                continue
            fallback_records.append(record)
        return fallback_records

    def _build_coverage_candidate(
        self,
        image_rgb,
        region: TextRegionCandidate,
    ) -> CoverageCandidate | None:
        region_lines = self._ocr_region_lines(image_rgb, region.bbox)
        confident_lines = [
            line
            for line in region_lines
            if line.confidence >= max(0.1, self._config.coverage_confidence_threshold - 0.1)
        ]
        source_text = self._clean_text(" ".join(line.text for line in confident_lines)) if confident_lines else ""
        confidence = sum(line.confidence for line in confident_lines) / len(confident_lines) if confident_lines else 0.0
        if source_text and self._looks_like_ocr_noise(source_text, confidence):
            source_text = ""
            confidence = 0.0

        translated_text = self._translate_text(source_text) if source_text else "[REVIEW REGION]"
        polygons = [list(line.polygon) for line in confident_lines] if confident_lines else self._fallback_mask_polygon(region.bbox)
        return CoverageCandidate(
            source_text=source_text,
            translated_text=translated_text,
            confidence=confidence,
            detector_score=region.score,
            bbox=region.bbox,
            mask_polygons=polygons,
        )

    def _coverage_pass(
        self,
        image_rgb,
        existing_records: List[TranslationRecord],
        regions: List[TextRegionCandidate],
    ) -> tuple[List[TranslationRecord], List[CoverageCandidate]]:
        recovered: List[TranslationRecord] = []
        review_candidates: List[CoverageCandidate] = []
        claimed = list(existing_records)
        for region in regions:
            if self._overlaps_existing_records(
                region.bbox,
                claimed + recovered,
                threshold=self._config.coverage_overlap_threshold,
            ):
                continue

            region_lines = [
                line
                for line in self._ocr_region_lines(image_rgb, region.bbox)
                if line.confidence >= self._config.coverage_confidence_threshold
            ]
            groups = self._group_fallback_lines(region_lines)
            added_record = False
            for group in groups:
                record = self._record_from_lines(group, image_rgb.shape, outer_bbox=region.bbox)
                if record is None:
                    continue
                if self._overlaps_existing_records(record.bbox, claimed + recovered, threshold=0.5):
                    continue
                recovered.append(record)
                added_record = True

            if added_record:
                continue

            candidate = self._build_coverage_candidate(image_rgb, region)
            if candidate is None:
                continue
            review_candidates.append(candidate)

        review_candidates.sort(key=lambda item: (-item.detector_score, item.bbox[1], item.bbox[0]))
        return recovered, review_candidates

    def _group_fallback_lines(self, lines: List[OCRLine]) -> List[List[OCRLine]]:
        if not lines:
            return []

        ordered = sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0]))
        groups: List[List[OCRLine]] = []
        for line in ordered:
            placed = False
            line_h = max(1, line.bbox[3] - line.bbox[1])
            line_cx = (line.bbox[0] + line.bbox[2]) / 2
            for group in groups:
                gx1 = min(item.bbox[0] for item in group)
                gy1 = min(item.bbox[1] for item in group)
                gx2 = max(item.bbox[2] for item in group)
                gy2 = max(item.bbox[3] for item in group)
                group_height = max(1, gy2 - gy1)
                vertical_gap = max(0, line.bbox[1] - gy2, gy1 - line.bbox[3])
                if vertical_gap > max(40, int(group_height * 0.8), int(line_h * 2.2)):
                    continue

                group_cx = (gx1 + gx2) / 2
                horizontal_distance = abs(line_cx - group_cx)
                allowed_horizontal = max(120, int(max(gx2 - gx1, line.bbox[2] - line.bbox[0]) * 0.85))
                x_overlap = min(line.bbox[2], gx2) - max(line.bbox[0], gx1)
                if x_overlap < -allowed_horizontal and horizontal_distance > allowed_horizontal:
                    continue

                group.append(line)
                placed = True
                break

            if not placed:
                groups.append([line])

        normalized: List[List[OCRLine]] = []
        for group in groups:
            group.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
            if len(group) == 1:
                only = group[0]
                text_len = len(only.text.strip())
                if text_len < 6 and only.confidence < 0.85:
                    continue
            normalized.append(group)
        return normalized

    def analyze_manual_region(self, bbox: tuple[int, int, int, int]) -> TranslationRecord:
        image_rgb = load_image_rgb(self._config.input_path)
        crop = crop_image(image_rgb, bbox)
        crop_lines = self._ocr.extract_lines(crop, exhaustive=True)

        adjusted: List[OCRLine] = []
        x1, y1, _, _ = bbox
        for line in crop_lines:
            adjusted_polygon = [(x + x1, y + y1) for x, y in line.polygon]
            adjusted_line = OCRLine(text=line.text, confidence=line.confidence, polygon=adjusted_polygon)
            if self._is_translatable_text(adjusted_line.text) and adjusted_line.confidence >= self._config.min_confidence:
                adjusted.append(adjusted_line)

        adjusted.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
        if adjusted:
            source_text = self._clean_text(" ".join(line.text for line in adjusted))
            translated = self._translate_text(source_text)
            record_bbox = self._combined_bbox_from_lines(adjusted, image_rgb.shape, outer_bbox=bbox)
            polygons = [list(line.polygon) for line in adjusted]
            confidence = sum(line.confidence for line in adjusted) / len(adjusted)
        else:
            source_text = "[MANUAL REGION]"
            translated = "[EDIT THIS TRANSLATION]"
            record_bbox = bbox
            bx1, by1, bx2, by2 = bbox
            polygons = [[(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)]]
            confidence = 0.0

        return TranslationRecord(
            source_text=source_text,
            translated_text=translated,
            confidence=confidence,
            bbox=record_bbox,
            mask_polygons=polygons,
        )

    def analyze_with_coverage(self) -> AnalysisResult:
        image_rgb = load_image_rgb(self._config.input_path)
        regions = self._collect_region_proposals(image_rgb)
        records: List[TranslationRecord] = self._build_detected_region_records(image_rgb, regions)
        if not records:
            all_lines = self._ocr.extract_lines(image_rgb)
            kept_lines = self._select_lines(all_lines, image_rgb)
            for line in kept_lines:
                source_text = self._clean_text(line.text)
                translated = self._translate_text(source_text)
                records.append(
                    TranslationRecord(
                        source_text=source_text,
                        translated_text=translated,
                        confidence=line.confidence,
                        bbox=self._expand_bbox(line.bbox, image_rgb.shape),
                        mask_polygons=[list(line.polygon)],
                    )
                )
        records.extend(self._build_page_fallback_records(image_rgb, records))
        records = self._dedupe_records(records)
        if self._ai_assistant is not None:
            ai_records = self._review_with_ai(image_rgb.shape[1], image_rgb.shape[0], records)
            if ai_records:
                records = self._merge_ai_records(records, ai_records, image_rgb)

        recovered_records, coverage_candidates = self._coverage_pass(image_rgb, records, regions)
        if recovered_records:
            records = self._dedupe_records(records + recovered_records)
            _, coverage_candidates = self._coverage_pass(image_rgb, records, regions)
        return AnalysisResult(records=records, coverage_candidates=coverage_candidates)

    def analyze(self) -> List[TranslationRecord]:
        return self.analyze_with_coverage().records

    def _review_with_ai(self, width: int, height: int, local_records: List[TranslationRecord]) -> List[TranslationRecord]:
        if self._ai_assistant is None:
            return []
        ai_payload = self._ai_assistant.review_page(
            image_path=self._config.input_path,
            image_size=(width, height),
            local_records=[to_builtin(record) for record in local_records],
        )
        records: List[TranslationRecord] = []
        for item in ai_payload:
            records.append(
                TranslationRecord(
                    source_text=item["source_text"],
                    translated_text=item["translated_text"],
                    confidence=float(item.get("confidence", 1.0)),
                    bbox=tuple(item["bbox"]),
                    mask_polygons=[
                        [tuple(point) for point in polygon]
                        for polygon in item.get("mask_polygons", [])
                    ],
                )
            )
        return records

    def render(self, records: List[TranslationRecord]) -> None:
        image_rgb = load_image_rgb(self._config.input_path)
        polygons = [polygon for record in records for polygon in record.mask_polygons]
        mask = polygon_mask(
            shape=image_rgb.shape[:2],
            polygons=polygons,
            padding_px=self._config.padding_px,
        )
        translated_image = inpaint_text(image_rgb, mask)
        for record in records:
            translated_image = draw_text_block(
                translated_image,
                record.translated_text,
                record.bbox,
                font_path=str(self._config.font_path) if self._config.font_path else None,
            )
        save_image_rgb(self._config.output_path, translated_image)
        metadata_path = self._config.output_path.with_suffix(".json")
        metadata_path.write_text(json.dumps([to_builtin(record) for record in records], indent=2), encoding="utf-8")

    def run(self) -> List[TranslationRecord]:
        records = self.analyze()
        self.render(records)
        return records

    @property
    def output_path(self) -> Path:
        return self._config.output_path
