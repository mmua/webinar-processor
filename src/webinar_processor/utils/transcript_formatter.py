"""Format diarized transcripts into structured text for LLM processing."""

import re
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def identify_main_speaker(segments: list) -> Optional[str]:
    """
    Identify the main speaker (lecturer) by total speaking time.

    Uses total duration rather than segment count because the lecturer
    typically has longer utterances than audience members.
    """
    durations: dict = {}
    for seg in segments:
        speaker = seg.get('speaker')
        if speaker is None:
            continue
        duration = seg.get('end', 0) - seg.get('start', 0)
        durations[speaker] = durations.get(speaker, 0) + duration

    if not durations:
        return None

    main = max(durations, key=durations.get)
    total = sum(durations.values())
    main_dur = durations[main]
    logger.info(
        f"Main speaker: {main} ({main_dur:.0f}/{total:.0f}s, "
        f"{main_dur / total * 100:.0f}%)"
    )
    return main


def format_diarized_transcript(
    segments: list,
    main_speaker: Optional[str] = None,
    paragraph_gap: float = 3.0,
) -> str:
    """
    Convert diarized transcript segments into well-structured text.

    - Groups consecutive same-speaker-type segments into paragraphs
    - Inserts paragraph breaks at time gaps > paragraph_gap seconds
    - Marks non-lecturer segments as [ВОПРОС/КОММЕНТАРИЙ]
    - Returns clean text with natural paragraph structure

    Args:
        segments: List of diarized segments with start/end/speaker/text
        main_speaker: Speaker ID of lecturer (auto-detected if None)
        paragraph_gap: Seconds of silence to trigger paragraph break

    Returns:
        Formatted text with paragraph breaks and speaker annotations
    """
    if not segments:
        return ""

    if main_speaker is None:
        main_speaker = identify_main_speaker(segments)

    paragraphs: List[str] = []
    current_parts: List[str] = []
    current_is_main: Optional[bool] = None
    prev_end: float = 0.0

    def flush():
        nonlocal current_parts, current_is_main
        if not current_parts:
            return
        text = ' '.join(current_parts)
        if not text.strip():
            current_parts = []
            return
        if current_is_main:
            paragraphs.append(text)
        else:
            paragraphs.append(f"[ВОПРОС/КОММЕНТАРИЙ]: {text}")
        current_parts = []

    for seg in segments:
        speaker = seg.get('speaker')
        text = seg.get('text', '').strip()
        start = seg.get('start', 0.0)
        end = seg.get('end', start)

        if not text:
            continue

        is_main = (speaker == main_speaker)
        gap = start - prev_end
        speaker_type_changed = (
            current_is_main is not None and is_main != current_is_main
        )

        if speaker_type_changed or (gap > paragraph_gap and current_parts):
            flush()

        current_parts.append(text)
        current_is_main = is_main
        prev_end = end

    flush()

    return '\n\n'.join(paragraphs)


def is_diarized_format(data) -> bool:
    """Check if JSON data is in diarized format (list of segments with timestamps)."""
    if not isinstance(data, list):
        return False
    if not data:
        return False
    first = data[0]
    return isinstance(first, dict) and 'text' in first and 'start' in first


def add_paragraph_breaks(text: str, sentences_per_paragraph: int = 5) -> str:
    """
    Add paragraph breaks to flat text that has no newlines.
    Groups sentences into paragraphs for readability.
    """
    # Split on sentence endings followed by a space and uppercase letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[А-ЯA-Z])', text)

    paragraphs = []
    for i in range(0, len(sentences), sentences_per_paragraph):
        paragraph = ' '.join(sentences[i:i + sentences_per_paragraph])
        paragraphs.append(paragraph)

    return '\n\n'.join(paragraphs)


def split_segments_by_time(
    segments: list,
    chunk_duration_minutes: float = 40.0,
    overlap_minutes: float = 2.0,
) -> List[list]:
    """
    Split diarized segments into time-based chunks for long transcript processing.

    Returns list of segment lists, each covering ~chunk_duration_minutes with overlap.
    """
    if not segments:
        return []

    chunk_duration = chunk_duration_minutes * 60
    overlap_duration = overlap_minutes * 60

    first_start = segments[0].get('start', 0)
    last_end = segments[-1].get('end', 0)
    total_duration = last_end - first_start

    if total_duration <= chunk_duration:
        return [segments]

    chunks: List[list] = []
    chunk_start = first_start

    while chunk_start < last_end:
        chunk_end = chunk_start + chunk_duration

        # Collect segments that overlap with this time range
        chunk_segments = [
            seg for seg in segments
            if seg.get('end', 0) > chunk_start
            and seg.get('start', 0) < chunk_end
        ]

        if chunk_segments:
            chunks.append(chunk_segments)

        # Move forward with overlap
        chunk_start += chunk_duration - overlap_duration

    logger.info(
        f"Split {total_duration / 60:.0f}min transcript into "
        f"{len(chunks)} chunks of ~{chunk_duration_minutes:.0f}min"
    )
    return chunks
