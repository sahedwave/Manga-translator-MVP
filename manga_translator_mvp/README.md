---
title: Manga Translator MVP
emoji: "🗨️"
colorFrom: slate
colorTo: amber
sdk: gradio
sdk_version: 6.8.0
app_file: app.py
pinned: false
python_version: "3.12"
---

# Manga Translator MVP

Minimal Python MVP for translating romanized manga text into English and drawing the translated text back into the image.

## Creator

**S Rahman from NE-09**  
[LinkedIn](https://www.linkedin.com/in/md-shahed-rahman-22732629a/)

## Folder structure

```text
manga_translator_mvp/
  README.md
  requirements.txt
  app/
    __init__.py
    __main__.py
    cli.py
    config.py
    gui.py
    image_ops.py
    ocr.py
    gemini_assist.py
    pipeline.py
    translate.py
```

## What the MVP does

- Loads a manga image.
- Detects candidate text regions before OCR using image-based heuristics.
- OCRs each detected region and groups lines into translatable blocks.
- Runs a second coverage pass on suspicious uncovered regions.
- Masks and inpaints the original text.
- Draws English text back inside the detected text box.
- Writes an edited image and a JSON sidecar with OCR/translation metadata.
- Exposes a browser UI through Gradio instead of a desktop Tk window.
- Surfaces unresolved text-like regions in a review panel instead of silently dropping them.

## Dependencies

- `rapidocr-onnxruntime` for OCR and text region detection.
- `opencv-python` for image IO, masking, and inpainting.
- `Pillow` for text rendering.
- `numpy` for image operations.
- `deep-translator` for a simple MVP translation backend.
- `google-genai` for optional Gemini-assisted text recovery and stronger translation.
- `gradio` for the browser UI.

## Install

```bash
cd "/Users/VIP/Documents/New project/manga_translator_mvp"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Browser UI:

```bash
python -m app
```

This launches a local Gradio app in the browser. Upload a manga image, then click `Translate`.

## Deploy

Recommended host: Hugging Face Spaces.

1. Create a new Gradio Space on Hugging Face.
2. Push this repository to that Space, or use the included GitHub Actions workflow.
3. Add any needed secrets such as `GEMINI_API_KEY` in the Space settings.

If you want GitHub to deploy automatically, add these GitHub repository secrets:

- `HF_TOKEN`
- `HF_USERNAME`
- `HF_SPACE_NAME`

This repository already includes the Space entrypoint in [app.py](/Users/VIP/Documents/New%20project/manga_translator_mvp/app.py) and the required README metadata.

For stronger automatic detection and translation:

1. Enter a Gemini API key in the UI or export `GEMINI_API_KEY`
2. Choose `gemini` as the translator
3. Enable `Gemini AI assist for missed text + better OCR`

CLI mode is still available:

```bash
python -m app \
  --input "/absolute/path/to/page.png" \
  --output "/absolute/path/to/page_en.png" \
  --translator google \
  --source auto \
  --target en
```

For offline pipeline testing without real translation:

```bash
python -m app \
  --input "/absolute/path/to/page.png" \
  --output "/absolute/path/to/page_en.png" \
  --translator mock
```

## Implementation plan

1. Start with OCR on Latin characters only.
2. Keep only high-confidence text regions that look like speech bubbles.
3. Translate line-by-line through a pluggable translator interface.
4. Inpaint the original text region before drawing English text.
5. Write overlay text with simple wrapping and auto-fit sizing.
6. Save extracted OCR data for manual review and later tuning.

## Notes

- This first version assumes the text is already written using English letters.
- Translation quality depends heavily on the romanized source text and backend.
- The detector is still heuristic-based, not a trained manga-text model.
- `google` mode depends on internet access. Use `mock` mode to test the GUI/pipeline locally first.
- `gemini` mode depends on internet access and a Gemini API key.
- For production use, replace the detector with a trained text detector and keep manual review enabled.

## How to improve accuracy later

1. Add a real speech-bubble segmentation model and use it to constrain OCR.
2. Group OCR boxes by bubble instead of translating line-by-line.
3. Replace `deep-translator` with a better translation model or service.
4. Add a manual correction UI before redrawing text.
5. Use OCR confidence thresholds and glossary overrides for character names.
6. Use a font pack and per-bubble text layout rules to match manga styling better.
