from abc import ABC, abstractmethod
import os

from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator
import google.generativeai as genai


# ===============================
# Utility functions
# ===============================

def _match_source_style(source_text: str, translated_text: str) -> str:
    if source_text.isupper():
        return translated_text.upper()
    return translated_text


def _polish_translation(source_text: str, translated_text: str) -> str:
    source_upper = source_text.upper().strip()
    cleaned = " ".join(translated_text.split())

    exact_source_overrides = {
        "TIDAK APA-APA": "I'm fine.",
        "APA KAU BAIK-BAIK SAJA?": "Are you okay?",
        "OH CHANGSU": "Oh, Changsu.",
        "PARK JINYEONG-SSI": "Park Jinyeong-ssi.",
    }

    if source_upper in exact_source_overrides:
        return _match_source_style(source_text, exact_source_overrides[source_upper])

    phrase_replacements = {
        "THE SAME LIKE": "LIKE",
        "DON'T SAY SO DISAPPOINTING THINGS, DON'T.": "Don't say something so disappointing.",
        "YOUR CONDITION SHOULD STILL BE...": "Your condition should still be...",
    }

    for old, new in phrase_replacements.items():
        cleaned = cleaned.replace(old, new)

    if source_upper.startswith("JANGAN MENGATAKAN HAL MENGECEWAKAN"):
        cleaned = "Don't say something so disappointing."

    return _match_source_style(source_text, cleaned)


# ===============================
# Translator Base
# ===============================

class BaseTranslator(ABC):
    @abstractmethod
    def translate(self, text: str) -> str:
        raise NotImplementedError


class MockTranslator(BaseTranslator):
    def translate(self, text: str) -> str:
        return f"[EN] {text}"


class GoogleWebTranslator(BaseTranslator):
    def __init__(self, source_lang="auto", target_lang="en"):
        self._translator = GoogleTranslator(source=source_lang, target=target_lang)

    def translate(self, text: str) -> str:
        translated = self._translator.translate(text)
        return _polish_translation(text, translated)


class GeminiTranslator(BaseTranslator):
    def __init__(self, source_lang="auto", target_lang="en", api_key=None, model="gemini-1.5-flash"):
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("GEMINI_API_KEY not set")

        genai.configure(api_key=resolved_key)
        self._model = genai.GenerativeModel(model)
        self._source_lang = source_lang
        self._target_lang = target_lang

    def translate(self, text: str) -> str:
        prompt = (
            f"Translate manga dialogue into natural English.\n"
            f"Keep names and honorifics.\n"
            f"Be concise.\n\n"
            f"Text: {text}"
        )

        response = self._model.generate_content(prompt)
        translated = (response.text or "").strip()

        return _polish_translation(text, translated)


# ===============================
# Factory
# ===============================

def build_translator(name, source_lang, target_lang):
    name = name.lower().strip()

    if name == "mock":
        return MockTranslator()

    if name == "google":
        return GoogleWebTranslator(source_lang, target_lang)

    if name == "gemini":
        return GeminiTranslator(source_lang, target_lang)

    raise ValueError(f"Unsupported translator: {name}")


# ===============================
# Flask App
# ===============================

app = Flask(__name__)


@app.route("/")
def home():
    return "Manga Translator API Running 🚀"


@app.route("/translate")
def translate_api():
    text = request.args.get("text", "")
    engine = request.args.get("engine", "google")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        translator = build_translator(engine, "auto", "en")
        result = translator.translate(text)

        return jsonify({
            "engine": engine,
            "original": text,
            "translated": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
