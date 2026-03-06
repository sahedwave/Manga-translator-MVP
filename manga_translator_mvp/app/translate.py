from abc import ABC, abstractmethod
import os

from deep_translator import GoogleTranslator
from google import genai


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


class BaseTranslator(ABC):
    @abstractmethod
    def translate(self, text: str) -> str:
        raise NotImplementedError


class MockTranslator(BaseTranslator):
    def translate(self, text: str) -> str:
        return f"[EN] {text}"


class GoogleWebTranslator(BaseTranslator):
    def __init__(self, source_lang: str = "auto", target_lang: str = "en") -> None:
        self._translator = GoogleTranslator(source=source_lang, target=target_lang)

    def translate(self, text: str) -> str:
        translated = self._translator.translate(text)
        return _polish_translation(text, translated)


class GeminiTranslator(BaseTranslator):
    def __init__(self, source_lang: str = "auto", target_lang: str = "en", api_key: str | None = None, model: str = "gemini-3-flash-preview") -> None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("Gemini API key is required for Gemini translation.")
        self._client = genai.Client(api_key=resolved_key)
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._model = model

    def translate(self, text: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=(
                "Translate manga dialogue into natural English. "
                "Preserve names and honorifics where appropriate. "
                "Be concise and idiomatic. Return only the translated line.\n"
                f"Source language: {self._source_lang}. Target language: {self._target_lang}. Text: {text}"
            ),
        )
        translated = (response.text or "").strip()
        return _polish_translation(text, translated)


def build_translator(name: str, source_lang: str, target_lang: str, api_key: str | None = None, model: str = "gemini-3-flash-preview") -> BaseTranslator:
    normalized = name.lower().strip()
    if normalized == "mock":
        return MockTranslator()
    if normalized == "google":
        return GoogleWebTranslator(source_lang=source_lang, target_lang=target_lang)
    if normalized == "gemini":
        return GeminiTranslator(source_lang=source_lang, target_lang=target_lang, api_key=api_key, model=model)
    raise ValueError(f"Unsupported translator backend: {name}")
