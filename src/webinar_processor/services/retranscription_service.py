"""Retranscription service: Whisper + Qwen3-ASR wrappers."""

import logging

logger = logging.getLogger(__name__)

LANGUAGE_MAP = {
    "ru": "Russian",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}


class RetranscriptionService:
    def __init__(self, language: str = "ru"):
        self.language = language
        self._whisper_model = None
        self._qwen3_model = None

    def _load_whisper(self):
        if self._whisper_model is None:
            import whisper
            logger.info("Loading Whisper large-v3...")
            self._whisper_model = whisper.load_model("large-v3")
            logger.info("Whisper model loaded")
        return self._whisper_model

    def _load_qwen3(self):
        if self._qwen3_model is None:
            from qwen_asr import Qwen3ASRModel
            logger.info("Loading Qwen3-ASR-1.7B...")
            self._qwen3_model = Qwen3ASRModel.from_pretrained(
                "Qwen/Qwen3-ASR-1.7B",
                device_map="cpu",
            )
            logger.info("Qwen3-ASR model loaded")
        return self._qwen3_model

    def transcribe_whisper(self, wav_path: str) -> str:
        model = self._load_whisper()
        result = model.transcribe(wav_path, language=self.language)
        return result["text"].strip()

    def transcribe_qwen3(self, wav_path: str) -> str:
        model = self._load_qwen3()
        lang_name = LANGUAGE_MAP.get(self.language, self.language)

        result = model.transcribe(
            audio=wav_path,
            language=lang_name,
            return_time_stamps=False,
        )
        text = (result[0].text or "").strip() if result else ""
        return text
