"""Transcription and diarization service functions.

This module owns heavy ASR/diarization logic so CLI commands stay thin.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union


logger = logging.getLogger(__name__)


DEFAULT_ASR_MODEL = "large-v3"
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
ASR_LANGUAGE_MAP = {
    "ru": "russian",
    "en": "english",
}


def transcribe_wav(wav_filename: str, language: str = "ru") -> Dict[str, Any]:
    """Transcribe WAV audio into Whisper-compatible segment structure."""
    import whisper

    model_id = os.getenv("ASR_WHISPER_MODEL", DEFAULT_ASR_MODEL)
    normalized_language = _normalize_asr_language(language)

    logger.info("Loading Whisper model: %s", model_id)
    model = whisper.load_model(model_id)

    logger.info("Transcribing audio: %s", wav_filename)
    result = model.transcribe(wav_filename, language=normalized_language)

    formatted_result = {
        "text": result.get("text", ""),
        "segments": result.get("segments", []),
        "language": normalized_language,
        "model": model_id,
    }
    logger.info(
        "ASR done: %d segments, %d chars",
        len(formatted_result["segments"]),
        len(formatted_result["text"]),
    )
    return formatted_result


def diarize_wav(
    wav_filename: str,
    transcription_result: Dict[str, Any],
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run speaker diarization and merge it with ASR segments."""
    from pyannote.audio import Pipeline
    from pyannote_whisper.utils import diarize_text

    token = hf_token or os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        raise ValueError("HUGGING_FACE_TOKEN is not set")

    logger.info("Loading diarization model: %s", DEFAULT_DIARIZATION_MODEL)
    pipeline = Pipeline.from_pretrained(
        DEFAULT_DIARIZATION_MODEL,
        token=token,
    )

    diarization_output = pipeline(wav_filename)
    diarization_result = diarization_output.speaker_diarization
    diarized_segments = diarize_text(transcription_result, diarization_result)

    result = []
    for seg, speaker, text in diarized_segments:
        result.append(
            {
                "start": seg.start,
                "end": seg.end,
                "speaker": speaker,
                "text": text,
            }
        )

    logger.info("Diarization done: %d merged segments", len(result))
    return result


def _normalize_asr_language(language: str) -> str:
    normalized = (language or "ru").strip().lower()
    return ASR_LANGUAGE_MAP.get(normalized, normalized)
