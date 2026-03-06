from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


Polygon = Sequence[Tuple[int, int]]


@dataclass(slots=True)
class BubbleRegion:
    bbox: Tuple[int, int, int, int]
    polygon: List[Tuple[int, int]]


@dataclass(slots=True)
class TextRegionCandidate:
    bbox: Tuple[int, int, int, int]
    score: float
    source: str


def load_image_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Could not write image: {path}")


def polygon_mask(shape: Tuple[int, int], polygons: Iterable[Polygon], padding_px: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    if padding_px > 0:
        kernel = np.ones((padding_px, padding_px), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def inpaint_text(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(image_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


def region_stats(image_rgb: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    crop = image_rgb[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
    if crop.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(gray.mean()), float(gray.var())


def crop_image(image_rgb: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return image_rgb[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)].copy()


def _clip_bbox(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int, int] | Tuple[int, int],
) -> Tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def _bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def _merge_bbox(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _expand_bbox(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int, int] | Tuple[int, int],
    pad_x: int,
    pad_y: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return _clip_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), image_shape)


def _boxes_should_merge(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], gap_px: int) -> bool:
    if _intersection_area(a, b) > 0:
        return True
    expanded = _expand_bbox(a, (max(a[3], b[3]) + gap_px, max(a[2], b[2]) + gap_px), gap_px, gap_px)
    return _intersection_area(expanded, b) > 0


def _merge_candidate_boxes(
    boxes: List[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int, int],
    gap_px: int,
) -> List[Tuple[int, int, int, int]]:
    merged = [_clip_bbox(box, image_shape) for box in boxes if _bbox_area(box) > 0]
    changed = True
    while changed:
        changed = False
        next_boxes: List[Tuple[int, int, int, int]] = []
        while merged:
            current = merged.pop()
            keep_merging = True
            while keep_merging:
                keep_merging = False
                remaining: List[Tuple[int, int, int, int]] = []
                for other in merged:
                    if _boxes_should_merge(current, other, gap_px=gap_px):
                        current = _merge_bbox(current, other)
                        keep_merging = True
                        changed = True
                    else:
                        remaining.append(other)
                merged = remaining
            next_boxes.append(_clip_bbox(current, image_shape))
        merged = next_boxes
    return merged


def _collect_component_boxes(
    mask: np.ndarray,
    image_shape: Tuple[int, int, int],
    min_area: int,
    max_area: int,
    expand_px: int,
) -> List[Tuple[int, int, int, int]]:
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: List[Tuple[int, int, int, int]] = []
    for idx in range(1, component_count):
        x, y, width, height, area = stats[idx]
        if area < min_area or area > max_area:
            continue
        if width < 10 or height < 8:
            continue
        aspect_ratio = width / max(height, 1)
        if not 0.2 <= aspect_ratio <= 14.0:
            continue
        boxes.append(_expand_bbox((x, y, x + width, y + height), image_shape, expand_px, expand_px))
    return boxes


def _collect_mser_boxes(
    gray: np.ndarray,
    image_shape: Tuple[int, int, int],
    min_area: int,
    max_area: int,
) -> List[Tuple[int, int, int, int]]:
    detector = cv2.MSER_create(5, max(12, min_area), max(min_area + 1, max_area))
    boxes: List[Tuple[int, int, int, int]] = []
    for source in (gray, cv2.bitwise_not(gray)):
        _, regions = detector.detectRegions(source)
        for x, y, width, height in regions:
            area = width * height
            if area < min_area or area > max_area:
                continue
            if width < 10 or height < 8:
                continue
            boxes.append(_expand_bbox((x, y, x + width, y + height), image_shape, 6, 4))
    return boxes


def _score_text_region(
    gray: np.ndarray,
    binary_mask: np.ndarray,
    edge_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> float:
    x1, y1, x2, y2 = bbox
    gray_crop = gray[y1:y2, x1:x2]
    binary_crop = binary_mask[y1:y2, x1:x2]
    edge_crop = edge_mask[y1:y2, x1:x2]
    if gray_crop.size == 0:
        return 0.0

    area = gray_crop.shape[0] * gray_crop.shape[1]
    dark_ratio = float(np.mean(gray_crop < 190))
    ink_ratio = float(np.count_nonzero(binary_crop)) / max(area, 1)
    edge_ratio = float(np.count_nonzero(edge_crop)) / max(area, 1)
    variance_score = min(float(gray_crop.var()) / 2200.0, 1.0)
    compactness_penalty = min(area / max(gray.shape[0] * gray.shape[1], 1), 0.08) / 0.08

    score = (dark_ratio * 0.22) + (ink_ratio * 0.38) + (edge_ratio * 0.28) + (variance_score * 0.18)
    score -= compactness_penalty * 0.06
    return max(0.0, min(score, 1.0))


def detect_text_regions(
    image_rgb: np.ndarray,
    min_area_ratio: float = 0.00015,
    max_area_ratio: float = 0.18,
) -> List[TextRegionCandidate]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    image_area = image_rgb.shape[0] * image_rgb.shape[1]
    min_area = max(30, int(image_area * min_area_ratio))
    max_area = max(min_area + 1, int(image_area * max_area_ratio))

    adaptive_inv = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)),
    )
    _, blackhat_mask = cv2.threshold(blackhat, 18, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(gray, 60, 180)

    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    adaptive_groups = cv2.morphologyEx(adaptive_inv, cv2.MORPH_CLOSE, line_kernel, iterations=1)
    blackhat_groups = cv2.dilate(blackhat_mask, square_kernel, iterations=1)
    edge_groups = cv2.dilate(canny, square_kernel, iterations=1)

    boxes: List[Tuple[int, int, int, int]] = []
    boxes.extend(_collect_component_boxes(adaptive_groups, image_rgb.shape, min_area, max_area, expand_px=8))
    boxes.extend(_collect_component_boxes(blackhat_groups, image_rgb.shape, min_area, max_area, expand_px=10))
    boxes.extend(_collect_component_boxes(edge_groups, image_rgb.shape, min_area, max_area, expand_px=8))
    boxes.extend(_collect_mser_boxes(gray, image_rgb.shape, min_area=max(20, min_area // 2), max_area=max_area))
    merged = _merge_candidate_boxes(boxes, image_rgb.shape, gap_px=18)

    candidates: List[TextRegionCandidate] = []
    for bbox in merged:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        if area < min_area or area > max_area:
            continue
        if width < 18 or height < 12:
            continue
        aspect_ratio = width / max(height, 1)
        if not 0.18 <= aspect_ratio <= 14.0:
            continue
        score = _score_text_region(gray, adaptive_inv, canny, bbox)
        if score < 0.24:
            continue
        source = "detector"
        candidates.append(TextRegionCandidate(bbox=bbox, score=score, source=source))

    candidates.sort(key=lambda candidate: (-candidate.score, candidate.bbox[1], candidate.bbox[0]))
    return candidates


def detect_bubble_regions(
    image_rgb: np.ndarray,
    min_area_ratio: float,
    max_area_ratio: float,
) -> List[BubbleRegion]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 215, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image_rgb.shape[0] * image_rgb.shape[1]
    min_area = image_area * min_area_ratio
    max_area = image_area * max_area_ratio

    bubbles: List[BubbleRegion] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / max(h, 1)
        if circularity < 0.35:
            continue
        if not 0.45 <= aspect_ratio <= 2.4:
            continue
        mean, variance = region_stats(image_rgb, (x, y, x + w, y + h))
        if mean < 180 or variance > 7000:
            continue
        polygon = [(int(px[0][0]), int(px[0][1])) for px in contour]
        bubbles.append(BubbleRegion(bbox=(x, y, x + w, y + h), polygon=polygon))

    bubbles.sort(key=lambda bubble: (bubble.bbox[1], bubble.bbox[0]))
    return bubbles


def _load_font(font_path: str | None, size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    candidate_paths: List[str] = []
    if font_path:
        candidate_paths.append(font_path)
    candidate_paths.extend(
        [
            "/System/Library/Fonts/Supplemental/Avenir Next Condensed.ttc",
            "/System/Library/Fonts/Supplemental/Arial Narrow.ttf",
            "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        ]
    )
    for candidate in candidate_paths:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _split_long_token(
    draw: ImageDraw.ImageDraw,
    token: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> List[str]:
    if not token:
        return [token]
    pieces: List[str] = []
    current = token[0]
    for ch in token[1:]:
        proposal = f"{current}{ch}"
        width = draw.textbbox((0, 0), proposal, font=font)[2]
        if width <= max_width or len(current) == 1:
            current = proposal
        else:
            pieces.append(current)
            current = ch
    pieces.append(current)
    return pieces


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [text]
    lines: List[str] = []
    current = words[0]
    if draw.textbbox((0, 0), current, font=font)[2] > max_width:
        split_parts = _split_long_token(draw, current, font, max_width)
        lines.extend(split_parts[:-1])
        current = split_parts[-1]
    for word in words[1:]:
        if draw.textbbox((0, 0), word, font=font)[2] > max_width:
            split_parts = _split_long_token(draw, word, font, max_width)
            for part in split_parts[:-1]:
                proposal = f"{current} {part}".strip()
                if draw.textbbox((0, 0), proposal, font=font)[2] <= max_width:
                    lines.append(proposal)
                else:
                    lines.append(current)
                    lines.append(part)
                current = ""
            word = split_parts[-1]
        proposal = f"{current} {word}"
        if draw.textbbox((0, 0), proposal, font=font)[2] <= max_width:
            current = proposal.strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _line_metrics(
    draw: ImageDraw.ImageDraw,
    lines: List[str],
    font: ImageFont.ImageFont,
    stroke_width: int,
    line_spacing: int,
) -> Tuple[List[Tuple[int, int, int, int]], List[int], int, int]:
    line_boxes = [draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width) for line in lines]
    line_heights = [box[3] - box[1] for box in line_boxes]
    total_height = sum(line_heights) + max(0, (len(lines) - 1) * line_spacing)
    max_line_width = max((box[2] - box[0]) for box in line_boxes) if line_boxes else 0
    return line_boxes, line_heights, total_height, max_line_width


def _pick_text_style(region_rgb: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], int]:
    if region_rgb.size == 0:
        return (15, 15, 15), (248, 248, 248), 2

    gray = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    spread = float(gray.std())
    if brightness >= 155:
        return (18, 18, 18), (245, 245, 245), 2 if spread < 36 else 1
    return (245, 245, 245), (18, 18, 18), 2


def draw_text_block(
    image_rgb: np.ndarray,
    text: str,
    bbox: Tuple[int, int, int, int],
    font_path: str | None = None,
    fill: Tuple[int, int, int] | None = None,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    width = max(10, x2 - x1)
    height = max(10, y2 - y1)
    inner_margin_x = max(10, int(width * 0.12))
    inner_margin_y = max(10, int(height * 0.13))
    safe_x1 = x1 + inner_margin_x
    safe_y1 = y1 + inner_margin_y
    safe_x2 = x2 - inner_margin_x
    safe_y2 = y2 - inner_margin_y
    safe_width = max(16, safe_x2 - safe_x1)
    safe_height = max(16, safe_y2 - safe_y1)

    image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image)
    region_rgb = image_rgb[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
    auto_fill, stroke_fill, default_stroke = _pick_text_style(region_rgb)
    fill_color = fill or auto_fill

    text = " ".join(text.split())
    chosen_font = _load_font(font_path, size=18)
    chosen_stroke = default_stroke
    chosen_spacing = 2
    wrapped = [text]
    max_font_size = min(safe_height, max(18, int(width * 0.19), int(height * 0.58)))
    min_font_size = 9

    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = _load_font(font_path, size=font_size)
        stroke_width = max(1, min(default_stroke, font_size // 12))
        line_spacing = max(1, font_size // 10)
        wrapped = _wrap_text(draw, text, font, max_width=max(12, safe_width))
        _, _, total_height, max_line_width = _line_metrics(
            draw,
            wrapped,
            font,
            stroke_width=stroke_width,
            line_spacing=line_spacing,
        )
        if total_height <= safe_height and max_line_width <= safe_width:
            chosen_font = font
            chosen_stroke = stroke_width
            chosen_spacing = line_spacing
            break

    line_boxes, line_heights, total_height, _ = _line_metrics(
        draw,
        wrapped,
        chosen_font,
        stroke_width=chosen_stroke,
        line_spacing=chosen_spacing,
    )
    shadow_fill = tuple(max(0, channel - 70) for channel in stroke_fill)
    shadow_offset = 1 if chosen_stroke >= 2 else 0

    y = safe_y1 + max(1, (safe_height - total_height) // 2)
    for line, box, line_height in zip(wrapped, line_boxes, line_heights):
        line_width = box[2] - box[0]
        x = safe_x1 + max(1, (safe_width - line_width) // 2) - box[0]
        draw_y = y - box[1]
        if shadow_offset:
            draw.text(
                (x + shadow_offset, draw_y + shadow_offset),
                line,
                font=chosen_font,
                fill=shadow_fill,
                stroke_width=chosen_stroke,
                stroke_fill=stroke_fill,
            )
        draw.text(
            (x, draw_y),
            line,
            font=chosen_font,
            fill=fill_color,
            stroke_width=chosen_stroke,
            stroke_fill=stroke_fill,
        )
        y += line_height + chosen_spacing

    return np.array(image)
