import argparse
from pathlib import Path

from .config import AppConfig
from .pipeline import MangaTranslationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manga translation MVP")
    parser.add_argument("--input", required=True, help="Input manga image path")
    parser.add_argument("--output", required=True, help="Output translated image path")
    parser.add_argument("--translator", default="mock", choices=["mock", "google"])
    parser.add_argument("--source", default="auto", help="Source language code")
    parser.add_argument("--target", default="en", help="Target language code")
    parser.add_argument("--font", default=None, help="Optional TTF/TTC font path")
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--bubble-brightness-threshold", type=float, default=170.0)
    parser.add_argument("--bubble-variance-threshold", type=float, default=6000.0)
    parser.add_argument("--padding", type=int, default=6, help="Mask dilation padding in pixels")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AppConfig(
        input_path=Path(args.input).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        translator=args.translator,
        source_lang=args.source,
        target_lang=args.target,
        font_path=Path(args.font).expanduser().resolve() if args.font else None,
        min_confidence=args.min_confidence,
        bubble_brightness_threshold=args.bubble_brightness_threshold,
        bubble_variance_threshold=args.bubble_variance_threshold,
        padding_px=args.padding,
    )
    records = MangaTranslationPipeline(config).run()
    print(f"Translated {len(records)} text regions.")
