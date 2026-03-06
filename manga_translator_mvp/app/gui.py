from __future__ import annotations

from pathlib import Path
import tempfile
import traceback
from typing import Any
from uuid import uuid4

import gradio as gr
import numpy as np

from .config import AppConfig
from .image_ops import save_image_rgb
from .pipeline import CoverageCandidate, MangaTranslationPipeline, TranslationRecord
from .serialization import to_builtin


APP_CSS = """
#app-shell {
  background:
    radial-gradient(circle at top left, #2f1b1b 0%, rgba(47, 27, 27, 0) 34%),
    radial-gradient(circle at top right, #1a2738 0%, rgba(26, 39, 56, 0) 30%),
    linear-gradient(180deg, #11151d 0%, #0a0d12 100%);
  min-height: 100vh;
}

#hero-panel {
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 24px;
  padding: 24px 26px;
  background:
    linear-gradient(90deg, rgba(7, 9, 13, 0.96) 0%, rgba(7, 9, 13, 0.85) 42%, rgba(7, 9, 13, 0.58) 100%),
    url("https://static.wikia.nocookie.net/lookism/images/e/e2/Gun_Anime.png/revision/latest/scale-to-width-down/185?cb=20241030182135") right center / cover no-repeat;
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
}

#hero-panel h1,
#hero-panel p,
#hero-panel a,
#hero-panel strong {
  color: #f5f7fb !important;
}

#hero-panel a {
  text-decoration: none;
  border-bottom: 1px solid rgba(245, 247, 251, 0.45);
}

#hero-panel a:hover {
  border-bottom-color: rgba(245, 247, 251, 0.9);
}

#language-note {
  border: 1px solid rgba(255, 212, 138, 0.38);
  border-radius: 18px;
  padding: 14px 16px;
  background: rgba(255, 244, 216, 0.95);
  color: #2f2012;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.16);
}

#language-note strong {
  color: #7a4612;
}
"""


def _build_output_path(input_path: str, stage: str) -> Path:
    src = Path(input_path).resolve()
    suffix = src.suffix or ".png"
    output_dir = Path(tempfile.gettempdir()) / "manga_translator_mvp"
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_id = uuid4().hex[:8]
    return output_dir / f"{src.stem}_{stage}_{unique_id}{suffix}"


def _persist_source_image(image_value: Any) -> str | None:
    if image_value is None:
        return None
    if isinstance(image_value, (str, Path)):
        return str(Path(image_value).resolve())
    if isinstance(image_value, np.ndarray):
        output_dir = Path(tempfile.gettempdir()) / "manga_translator_mvp"
        output_dir.mkdir(parents=True, exist_ok=True)
        source_path = output_dir / f"source_{uuid4().hex[:8]}.png"
        save_image_rgb(source_path, image_value)
        return str(source_path)
    raise TypeError(f"Unsupported image input type: {type(image_value)!r}")


def _serialize_records(records: list[TranslationRecord]) -> list[dict[str, Any]]:
    return [to_builtin(record) for record in records]


def _serialize_coverage_candidates(candidates: list[CoverageCandidate]) -> list[dict[str, Any]]:
    return [to_builtin(candidate) for candidate in candidates]


def _rows_from_records(records: list[TranslationRecord]) -> list[list[Any]]:
    return [
        [
            True,
            record.source_text,
            record.translated_text,
            round(record.confidence, 3),
            f"{record.bbox[0]},{record.bbox[1]},{record.bbox[2]},{record.bbox[3]}",
        ]
        for record in records
    ]


def _rows_from_coverage_candidates(candidates: list[CoverageCandidate]) -> list[list[Any]]:
    return [
        [
            False,
            candidate.source_text,
            candidate.translated_text,
            round(candidate.confidence, 3),
            round(candidate.detector_score, 3),
            f"{candidate.bbox[0]},{candidate.bbox[1]},{candidate.bbox[2]},{candidate.bbox[3]}",
        ]
        for candidate in candidates
    ]


def _records_from_state(state: list[dict[str, Any]]) -> list[TranslationRecord]:
    records: list[TranslationRecord] = []
    for item in state:
        records.append(
            TranslationRecord(
                source_text=item["source_text"],
                translated_text=item["translated_text"],
                confidence=float(item["confidence"]),
                bbox=tuple(item["bbox"]),
                mask_polygons=[
                    [tuple(point) for point in polygon]
                    for polygon in item.get("mask_polygons", [])
                ],
            )
        )
    return records


def _coverage_from_state(state: list[dict[str, Any]]) -> list[CoverageCandidate]:
    candidates: list[CoverageCandidate] = []
    for item in state:
        candidates.append(
            CoverageCandidate(
                source_text=item.get("source_text", ""),
                translated_text=item.get("translated_text", ""),
                confidence=float(item.get("confidence", 0.0)),
                detector_score=float(item.get("detector_score", 0.0)),
                bbox=tuple(item["bbox"]),
                mask_polygons=[
                    [tuple(point) for point in polygon]
                    for polygon in item.get("mask_polygons", [])
                ],
            )
        )
    return candidates


def _annotations_from_state(
    image_path: str | None,
    record_state: list[dict[str, Any]],
    coverage_state: list[dict[str, Any]],
    pending_point: list[int] | None = None,
):
    if not image_path:
        return None
    records = _records_from_state(record_state)
    coverage_candidates = _coverage_from_state(coverage_state)
    annotations = [((record.bbox[0], record.bbox[1], record.bbox[2], record.bbox[3]), record.source_text[:32]) for record in records]
    annotations.extend(
        (
            (candidate.bbox[0], candidate.bbox[1], candidate.bbox[2], candidate.bbox[3]),
            f"MISS? {(candidate.source_text or 'review')[:24]}",
        )
        for candidate in coverage_candidates
    )
    if pending_point:
        x, y = pending_point
        annotations.append(((x - 10, y - 10, x + 10, y + 10), "start"))
    return (image_path, annotations)


def _normalize_review_rows(review_rows: Any) -> list[list[Any]]:
    if review_rows is None:
        return []
    if hasattr(review_rows, "values") and hasattr(review_rows, "columns"):
        return review_rows.values.tolist()
    if isinstance(review_rows, list):
        return review_rows
    return []


def _coerce_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    return bool(value)


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc)
    upper = text.upper()
    lower = text.lower()
    return "RESOURCE_EXHAUSTED" in upper or "quota exceeded" in lower or "insufficient_quota" in lower


def _run_pipeline_once(
    source_image_path: str,
    translator: str,
    ai_assist: bool,
    gemini_api_key: str,
    gemini_model: str,
):
    output_path = _build_output_path(source_image_path, "draft")
    config = AppConfig(
        input_path=Path(source_image_path).resolve(),
        output_path=output_path,
        translator=translator,
        ai_assist=ai_assist,
        gemini_api_key=gemini_api_key.strip() or None,
        gemini_model=gemini_model.strip() or "gemini-3-flash-preview",
    )
    pipeline = MangaTranslationPipeline(config)
    analysis = pipeline.analyze_with_coverage()
    records = analysis.records
    pipeline.render(records)
    serialized = _serialize_records(records)
    serialized_coverage = _serialize_coverage_candidates(analysis.coverage_candidates)
    rows = _rows_from_records(records)
    coverage_rows = _rows_from_coverage_candidates(analysis.coverage_candidates)
    annotations = _annotations_from_state(source_image_path, serialized, serialized_coverage)
    metadata = {
        "records": serialized,
        "coverage_candidates": serialized_coverage,
    }
    return str(output_path), annotations, rows, serialized, coverage_rows, serialized_coverage, metadata


def process_image(
    image_value: Any,
    translator: str,
    ai_assist: bool,
    gemini_api_key: str,
    gemini_model: str,
) -> tuple[str | None, Any, list[list[Any]], list[dict[str, Any]], list[list[Any]], list[dict[str, Any]], dict[str, Any] | list[dict], str, None, str | None]:
    source_image_path = _persist_source_image(image_value)
    if not source_image_path:
        return None, None, [], [], [], [], [], "Choose an image first.", None, None

    try:
        output_path, annotations, rows, serialized, coverage_rows, coverage_serialized, metadata = _run_pipeline_once(
            source_image_path=source_image_path,
            translator=translator,
            ai_assist=ai_assist,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
        )
        mode_text = "AI-assisted" if ai_assist else "Local"
        record_count = len(serialized)
        coverage_count = len(coverage_serialized)
        return (
            output_path,
            annotations,
            rows,
            serialized,
            coverage_rows,
            coverage_serialized,
            metadata,
            f"{mode_text} pass produced {record_count} translation record(s) and flagged {coverage_count} uncovered region(s). Edit the English text if needed, import any review candidates you trust, then click Render Final. To add a missed bubble manually, click two corners on the original image.",
            None,
            source_image_path,
        )
    except Exception as exc:
        if translator == "gemini" and _is_quota_error(exc):
            try:
                output_path, annotations, rows, serialized, coverage_rows, coverage_serialized, metadata = _run_pipeline_once(
                    source_image_path=source_image_path,
                    translator="google",
                    ai_assist=False,
                    gemini_api_key="",
                    gemini_model="",
                )
                return (
                    output_path,
                    annotations,
                    rows,
                    serialized,
                    coverage_rows,
                    coverage_serialized,
                    metadata,
                    "Gemini quota exhausted. Fell back to local OCR + Google translation.",
                    None,
                    source_image_path,
                )
            except Exception as fallback_exc:
                details = traceback.format_exc(limit=3)
                error_payload = [{"error": str(fallback_exc), "traceback": details}]
                return None, None, [], [], [], [], error_payload, f"Gemini quota exhausted and fallback failed: {fallback_exc}", None, source_image_path
        details = traceback.format_exc(limit=3)
        error_payload = [{"error": str(exc), "traceback": details}]
        return None, None, [], [], [], [], error_payload, f"Translation failed: {exc}", None, source_image_path


def render_final(source_image_path: str | None, state: list[dict[str, Any]], review_rows: list[list[Any]]) -> tuple[str | None, list[dict], str]:
    if not source_image_path:
        return None, [], "Choose an image first."
    if not state:
        return None, [], "Run detection first."

    try:
        output_path = _build_output_path(source_image_path, "final")
        original_records = _records_from_state(state)
        normalized_rows = _normalize_review_rows(review_rows)
        filtered_records: list[TranslationRecord] = []
        for index, row in enumerate(normalized_rows):
            if index >= len(original_records):
                break
            enabled = True if len(row) < 1 else _coerce_enabled(row[0])
            if not enabled:
                continue
            record = original_records[index]
            edited_translation = record.translated_text
            if len(row) >= 3 and row[2] is not None:
                edited_translation = str(row[2]).strip()
            if not edited_translation:
                continue
            record.translated_text = edited_translation
            filtered_records.append(record)

        config = AppConfig(
            input_path=Path(source_image_path).resolve(),
            output_path=output_path,
            translator="mock",
        )
        pipeline = MangaTranslationPipeline(config)
        pipeline.render(filtered_records)
        serialized = _serialize_records(filtered_records)
        return str(output_path), serialized, f"Rendered final image with {len(filtered_records)} reviewed translation(s)."
    except Exception as exc:
        details = traceback.format_exc(limit=3)
        return None, [{"error": str(exc), "traceback": details}], f"Final render failed: {exc}"


def import_selected_coverage(
    source_image_path: str | None,
    record_state: list[dict[str, Any]],
    coverage_state: list[dict[str, Any]],
    coverage_rows: Any,
) -> tuple[Any, list[list[Any]], list[dict[str, Any]], list[list[Any]], list[dict[str, Any]], dict[str, Any], str]:
    if not source_image_path:
        return None, [], [], [], [], {}, "Run detection first."

    records = _records_from_state(record_state)
    coverage_candidates = _coverage_from_state(coverage_state)
    normalized_coverage_rows = _normalize_review_rows(coverage_rows)

    retained_coverage: list[CoverageCandidate] = []
    imported = 0
    for index, candidate in enumerate(coverage_candidates):
        row = normalized_coverage_rows[index] if index < len(normalized_coverage_rows) else None
        enabled = False if row is None or len(row) < 1 else _coerce_enabled(row[0])
        if not enabled:
            retained_coverage.append(candidate)
            continue

        translated_text = candidate.translated_text
        if row is not None and len(row) >= 3 and row[2] is not None:
            translated_text = str(row[2]).strip()
        if not translated_text:
            retained_coverage.append(candidate)
            continue

        records.append(
            TranslationRecord(
                source_text=candidate.source_text or "[UNCERTAIN REGION]",
                translated_text=translated_text,
                confidence=candidate.confidence,
                bbox=candidate.bbox,
                mask_polygons=[
                    [tuple(point) for point in polygon]
                    for polygon in candidate.mask_polygons
                ],
            )
        )
        imported += 1

    serialized_records = _serialize_records(records)
    serialized_coverage = _serialize_coverage_candidates(retained_coverage)
    metadata = {
        "records": serialized_records,
        "coverage_candidates": serialized_coverage,
    }
    return (
        _annotations_from_state(source_image_path, serialized_records, serialized_coverage),
        _rows_from_records(records),
        serialized_records,
        _rows_from_coverage_candidates(retained_coverage),
        serialized_coverage,
        metadata,
        f"Imported {imported} coverage candidate(s) into the review table.",
    )


def handle_manual_click(
    image_value: Any,
    source_image_path: str | None,
    translator: str,
    ai_assist: bool,
    gemini_api_key: str,
    gemini_model: str,
    state: list[dict[str, Any]],
    coverage_state: list[dict[str, Any]],
    review_rows: Any,
    pending_point: list[int] | None,
    evt: gr.SelectData,
) -> tuple[Any, list[list[Any]], list[dict[str, Any]], list[dict], str, list[int] | None, str | None]:
    if not source_image_path:
        source_image_path = _persist_source_image(image_value)
    if not source_image_path:
        return None, [], [], [], "Choose an image first.", None, None

    point = [int(evt.index[0]), int(evt.index[1])] if isinstance(evt.index, (tuple, list)) else None
    if point is None:
        return (
            _annotations_from_state(source_image_path, state, coverage_state, pending_point),
            _normalize_review_rows(review_rows),
            state,
            {"records": state, "coverage_candidates": coverage_state},
            f"Could not read click position: {evt.index!r}",
            pending_point,
            source_image_path,
        )

    if pending_point is None:
        return (
            _annotations_from_state(source_image_path, state, coverage_state, point),
            _normalize_review_rows(review_rows),
            state,
            {"records": state, "coverage_candidates": coverage_state},
            f"Manual region start set at ({point[0]}, {point[1]}). Click the opposite corner.",
            point,
            source_image_path,
        )

    x1, y1 = pending_point
    x2, y2 = point
    bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    if bbox[2] - bbox[0] < 20 or bbox[3] - bbox[1] < 20:
        return (
            _annotations_from_state(source_image_path, state, coverage_state),
            _normalize_review_rows(review_rows),
            state,
            {"records": state, "coverage_candidates": coverage_state},
            "Selected region is too small. Try again.",
            None,
            source_image_path,
        )

    try:
        output_path = _build_output_path(source_image_path, "manual")
        config = AppConfig(
            input_path=Path(source_image_path).resolve(),
            output_path=output_path,
            translator=translator,
            ai_assist=ai_assist,
            gemini_api_key=gemini_api_key.strip() or None,
            gemini_model=gemini_model.strip() or "gemini-3-flash-preview",
        )
        pipeline = MangaTranslationPipeline(config)
        new_record = pipeline.analyze_manual_region(bbox)
        records = _records_from_state(state)
        records.append(new_record)
        serialized = _serialize_records(records)
        rows = _rows_from_records(records)
        annotations = _annotations_from_state(source_image_path, serialized, coverage_state)
        return (
            annotations,
            rows,
            serialized,
            {"records": serialized, "coverage_candidates": coverage_state},
            f"Manual region added at {bbox}. Edit the English text if needed, then click Render Final.",
            None,
            source_image_path,
        )
    except Exception as exc:
        details = traceback.format_exc(limit=3)
        error_payload = [{"error": str(exc), "traceback": details}]
        return (
            _annotations_from_state(source_image_path, state, coverage_state),
            _normalize_review_rows(review_rows),
            state,
            error_payload,
            f"Manual region failed: {exc}",
            None,
            source_image_path,
        )


def reset_for_new_image() -> tuple[None, None, list, list, list, list, list, str, None, None]:
    return None, None, [], [], [], [], [], "Image loaded. Click Detect + Translate.", None, None


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Manga Translator MVP", css=APP_CSS) as demo:
        with gr.Column(elem_id="app-shell"):
            with gr.Row(elem_id="hero-panel"):
                with gr.Column(scale=5):
                    gr.Markdown("# Manga Translator MVP")
                    gr.Markdown("**Created by S Rahman from NE-09**  \n[LinkedIn](https://www.linkedin.com/in/md-shahed-rahman-22732629a/)")
                with gr.Column(scale=3):
                    gr.Markdown(
                        """
<div id="language-note">
  <strong>Language Note</strong><br>
  This app works best with manga text written in English letters or romanized words. It does not reliably translate every language into English yet.
</div>
                        """
                    )
            gr.Markdown("Upload a manga image, choose a translator backend, and generate an English overlay version.")
            gr.Markdown("If automatic detection misses a bubble, click two corners on the original image to add a manual region.")
            gr.Markdown("Uncovered but suspicious regions are listed below so you can import them into the review queue instead of silently missing them.")

            with gr.Row():
                input_image = gr.Image(label="Input manga page", type="numpy", image_mode="RGB")
                output_image = gr.Image(label="Translated output", type="filepath")

            annotated_regions = gr.AnnotatedImage(label="Detected and manual regions")

            with gr.Row():
                translator = gr.Dropdown(choices=["mock", "google", "gemini"], value="mock", label="Translator")
                ai_assist = gr.Checkbox(label="Gemini AI assist for missed text + better OCR", value=False)
            with gr.Row():
                gemini_api_key = gr.Textbox(label="Gemini API key", type="password", placeholder="AIza...", value="")
                gemini_model = gr.Textbox(label="Gemini model", value="gemini-3-flash-preview")
            with gr.Row():
                run_button = gr.Button("Detect + Translate")
                import_coverage_button = gr.Button("Import Selected Coverage")
                render_button = gr.Button("Render Final")

            status = gr.Textbox(label="Status", interactive=False)
            review_table = gr.Dataframe(
                label="Review translations",
                headers=["Use", "Source text", "Editable English", "Confidence", "BBox"],
                datatype=["bool", "str", "str", "number", "str"],
                interactive=True,
                wrap=True,
            )
            coverage_table = gr.Dataframe(
                label="Coverage review",
                headers=["Use", "OCR hint", "Suggested English", "OCR Conf", "Detector", "BBox"],
                datatype=["bool", "str", "str", "number", "number", "str"],
                interactive=True,
                wrap=True,
            )
            records_state = gr.State([])
            coverage_state = gr.State([])
            pending_point_state = gr.State(None)
            source_image_path_state = gr.State(None)
            metadata = gr.JSON(label="OCR and translation records")

            input_image.change(
                fn=reset_for_new_image,
                outputs=[output_image, annotated_regions, review_table, records_state, coverage_table, coverage_state, metadata, status, pending_point_state, source_image_path_state],
            )

            run_button.click(
                fn=process_image,
                inputs=[input_image, translator, ai_assist, gemini_api_key, gemini_model],
                outputs=[output_image, annotated_regions, review_table, records_state, coverage_table, coverage_state, metadata, status, pending_point_state, source_image_path_state],
            )

            import_coverage_button.click(
                fn=import_selected_coverage,
                inputs=[source_image_path_state, records_state, coverage_state, coverage_table],
                outputs=[annotated_regions, review_table, records_state, coverage_table, coverage_state, metadata, status],
            )

            render_button.click(
                fn=render_final,
                inputs=[source_image_path_state, records_state, review_table],
                outputs=[output_image, metadata, status],
            )

            input_image.select(
                fn=handle_manual_click,
                inputs=[input_image, source_image_path_state, translator, ai_assist, gemini_api_key, gemini_model, records_state, coverage_state, review_table, pending_point_state],
                outputs=[annotated_regions, review_table, records_state, metadata, status, pending_point_state, source_image_path_state],
            )

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch(inbrowser=True)
