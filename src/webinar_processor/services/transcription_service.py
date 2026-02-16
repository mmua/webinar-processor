"""Transcription and diarization service functions.

This module owns heavy ASR/diarization logic so CLI commands stay thin.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union


logger = logging.getLogger(__name__)


DEFAULT_ASR_MODEL = "antony66/whisper-large-v3-russian"
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
ASR_LANGUAGE_MAP = {
    "ru": "russian",
    "en": "english",
}


def transcribe_wav(wav_filename: str, language: str = "ru") -> Dict[str, Any]:
    """Transcribe WAV audio into Whisper-compatible segment structure."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    model_id = os.getenv("ASR_WHISPER_MODEL", DEFAULT_ASR_MODEL)
    normalized_language = _normalize_asr_language(language)

    torch_dtype = torch.float32
    pipeline_device: Union[int, str] = -1

    if torch.cuda.is_available():
        torch_dtype = torch.float16
        pipeline_device = 0
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        pipeline_device = "mps"

    logger.info("Loading ASR model: %s", model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=256,
        chunk_length_s=30,
        batch_size=16 if pipeline_device == 0 else 4,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=pipeline_device,
    )

    asr_output = asr_pipeline(
        wav_filename,
        return_timestamps=True,
        generate_kwargs={
            "language": normalized_language,
            "task": "transcribe",
            "max_new_tokens": 256,
        },
    )

    result = {
        "text": asr_output.get("text", ""),
        "segments": _build_whisper_like_segments(asr_output),
        "language": normalized_language,
        "model": model_id,
    }
    logger.info(
        "ASR done: %d segments, %d chars",
        len(result["segments"]),
        len(result["text"]),
    )
    return result


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
        result.append({
            "start": seg.start,
            "end": seg.end,
            "speaker": speaker,
            "text": text,
        })

    logger.info("Diarization done: %d merged segments", len(result))
    return result


def _normalize_asr_language(language: str) -> str:
    normalized = (language or "ru").strip().lower()
    return ASR_LANGUAGE_MAP.get(normalized, normalized)


def _build_whisper_like_segments(asr_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert HF ASR chunks to pyannote-whisper expected segment format."""
    segments: List[Dict[str, Any]] = []
    chunks = asr_result.get("chunks", [])

    for idx, chunk in enumerate(chunks):
        timestamp = chunk.get("timestamp")
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
            continue

        start, end = timestamp
        if start is None:
            continue

        start_value = float(start)
        end_value = float(end) if end is not None else start_value + 0.01
        if end_value <= start_value:
            end_value = start_value + 0.01

        text = (chunk.get("text") or "").strip()
        if not text:
            continue

        segments.append({
            "id": idx,
            "start": start_value,
            "end": end_value,
            "text": text,
        })

    if not segments and (asr_result.get("text") or "").strip():
        segments.append({
            "id": 0,
            "start": 0.0,
            "end": 0.01,
            "text": asr_result["text"].strip(),
        })

    return segments
