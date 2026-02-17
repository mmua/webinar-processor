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
        self._qwen3_processor = None

    def _load_whisper(self):
        if self._whisper_model is None:
            import whisper
            logger.info("Loading Whisper large-v3...")
            self._whisper_model = whisper.load_model("large-v3")
            logger.info("Whisper model loaded")
        return self._whisper_model

    def _load_qwen3(self):
        if self._qwen3_model is None:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            model_id = "Qwen/Qwen3-ASR"
            logger.info("Loading Qwen3-ASR...")
            self._qwen3_processor = AutoProcessor.from_pretrained(model_id)
            self._qwen3_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, device_map="cpu"
            )
            logger.info("Qwen3-ASR model loaded")
        return self._qwen3_model, self._qwen3_processor

    def transcribe_whisper(self, wav_path: str) -> str:
        model = self._load_whisper()
        result = model.transcribe(wav_path, language=self.language)
        return result["text"].strip()

    def transcribe_qwen3(self, wav_path: str) -> str:
        import torch
        import soundfile as sf

        model, processor = self._load_qwen3()
        lang_name = LANGUAGE_MAP.get(self.language, self.language)

        audio_data, sample_rate = sf.read(wav_path)

        prompt = f"<|startoftext|><|{lang_name}|>"
        inputs = processor(
            audio=audio_data,
            sampling_rate=sample_rate,
            text=prompt,
            return_tensors="pt",
        )

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return transcription[0].strip() if transcription else ""
