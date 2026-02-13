"""Storytell article generation strategies.

Three strategies for transforming transcripts into educational articles:
1. Outline + per-section (default) — with prompt caching
2. Single-pass — one LLM call
3. Chunked — condense first, then outline + sections
"""

import json
import re
import logging
from typing import Optional

from webinar_processor.llm import LLMError, TOKEN_LIMITS
from webinar_processor.utils.completion import get_completion, get_output_limit
from webinar_processor.utils.io import load_prompt_template
from webinar_processor.utils.package import get_config_path
from webinar_processor.utils.token import count_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_article(
    text: str,
    model: str,
    segments: Optional[list] = None,
    no_appendix: bool = False,
    single_pass: bool = False,
) -> Optional[str]:
    """
    Generate an educational article from transcript text.

    Chooses the best strategy based on flags and text length.

    Args:
        text: Formatted transcript text
        model: LLM model name
        segments: Original diarized segments (for chunked strategy)
        no_appendix: Skip appendix generation
        single_pass: Force single-pass strategy

    Returns:
        Generated article text, or None on failure
    """
    text_tokens = count_tokens(model, text)
    token_limit = TOKEN_LIMITS.get(model, 128000)
    output_limit = get_output_limit(model)
    prompt_overhead = 3000
    available_for_input = token_limit - output_limit - prompt_overhead

    logger.info(
        "Text: %d chars (%d tokens), Model: %s (context=%d, output=%d)",
        len(text), text_tokens, model, token_limit, output_limit,
    )

    # Choose generation strategy
    if single_pass:
        logger.info("Single-pass mode...")
        return _storytell_single_pass(text, model)
    elif text_tokens <= available_for_input:
        logger.info("Outline + per-section mode (cached prefix)...")
        return _storytell_with_outline(text, model, no_appendix)
    else:
        logger.warning(
            "Text exceeds context (%d > %d tokens), condensing first...",
            text_tokens, available_for_input,
        )
        return _storytell_chunked(text, segments, model, no_appendix)


# ---------------------------------------------------------------------------
# Strategy 1: Outline + per-section generation with cached prefix (default)
# ---------------------------------------------------------------------------

def _storytell_with_outline(text: str, model: str, no_appendix: bool = False) -> str:
    """
    Generate article using outline + per-section approach with prompt caching.

    Prompt structure optimized for prefix caching:
    - Call 1 (outline): [transcript] + [outline instructions]
    - Calls 2-N (sections): [transcript + outline + terms] + [section task]
    - Call N+1 (appendix):  [transcript + outline + terms] + [appendix task]

    Calls 2 through N+1 share an identical prefix -> cached at 1/10th cost.
    """
    # Step 1: Generate outline + terminology (one call, full price)
    logger.info("Step 1: Generating outline + terms...")
    outline_prompt = load_prompt_template(
        get_config_path("storytell-outline-prompt.txt")
    )
    outline_response = _get_completion_safe(
        outline_prompt.format(text=text), model
    )
    if not outline_response:
        logger.warning("Outline generation failed, falling back to single-pass")
        return _storytell_single_pass(text, model)

    outline = _extract_json(outline_response)
    if not outline or not outline.get('sections'):
        logger.warning("Could not parse outline, falling back to single-pass")
        return _storytell_single_pass(text, model)

    chapter_title = outline.get('chapter_title', 'Глава')
    sections = outline.get('sections', [])
    terms = outline.get('terms', [])

    logger.info('Outline: "%s" — %d sections, %d terms', chapter_title, len(sections), len(terms))
    for s in sections:
        logger.info("  %s: %s", s.get('id', '?'), s.get('title', '?'))

    # Build shared prefix (identical for all subsequent calls -> cached)
    cached_prefix = _build_cached_prefix(text, sections, terms)
    prefix_tokens = count_tokens(model, cached_prefix)
    logger.info("Cached prefix: %d tokens (cached after first section call)", prefix_tokens)

    # Step 2: Generate each section (prefix cached from 2nd call onward)
    section_task_template = load_prompt_template(
        get_config_path("storytell-section-task.txt")
    )

    parts = [f"# {chapter_title}\n"]

    for i, section in enumerate(sections):
        section_id = section.get('id', f'S{i+1}')
        section_title = section.get('title', f'Раздел {i+1}')
        section_covers = section.get('covers', '')

        if i > 0:
            prev = sections[i - 1]
            prev_info = (
                f"ПРЕДЫДУЩИЙ РАЗДЕЛ (уже написан):\n"
                f"{prev.get('title', '')}: {prev.get('covers', '')}"
            )
        else:
            prev_info = "Это первый раздел статьи. Начни с введения в тему."

        if i + 1 < len(sections):
            nxt = sections[i + 1]
            next_info = (
                f"СЛЕДУЮЩИЙ РАЗДЕЛ (будет написан далее):\n"
                f"{nxt.get('title', '')}: {nxt.get('covers', '')}"
            )
        else:
            next_info = "Это последний раздел. Завершай статью выводами и призывом к действию."

        logger.info("Step 2: Section %d/%d: %s...", i + 1, len(sections), section_title)

        task_suffix = section_task_template.format(
            section_number=i + 1,
            total_sections=len(sections),
            section_title=section_title,
            section_covers=section_covers,
            prev_section_info=prev_info,
            next_section_info=next_info,
        )

        prompt = cached_prefix + task_suffix
        section_text = _get_completion_safe(prompt, model)
        if section_text:
            parts.append(f"\n## {section_title}\n\n{section_text}")
            logger.info("  %s done: %d chars", section_id, len(section_text))
        else:
            logger.error("  %s failed, skipping", section_id)

    # Step 3: Generate appendix (same cached prefix)
    if not no_appendix:
        logger.info("Step 3: Generating appendix...")
        appendix_task = load_prompt_template(
            get_config_path("storytell-appendix-task.txt")
        )
        prompt = cached_prefix + appendix_task
        appendix = _get_completion_safe(prompt, model)
        if appendix and appendix.strip():
            parts.append(f"\n\n---\n\n{appendix.strip()}")
            logger.info("  Appendix done: %d chars", len(appendix))

    return "\n".join(parts)


def _build_cached_prefix(text: str, sections: list, terms: list) -> str:
    """
    Build the shared prompt prefix for all section and appendix calls.

    This string must be IDENTICAL across calls for prompt caching to work.
    Transcript goes first (largest part), then outline and terms.
    """
    outline_text = _format_outline_for_prompt(sections)
    terms_text = _format_terms_for_prompt(terms)

    return (
        f"Основной спикер — лектор. Фрагменты [ВОПРОС/КОММЕНТАРИЙ] — реплики слушателей.\n\n"
        f"ТРАНСКРИПТ:\n---\n{text}\n---\n\n"
        f"ПЛАН СТАТЬИ:\n{outline_text}\n\n"
        f"СЛОВАРЬ ТЕРМИНОВ:\n{terms_text}\n\n"
        f"---\n\n"
    )


def _format_outline_for_prompt(sections: list) -> str:
    """Format outline sections as readable text for inclusion in prompts."""
    lines = []
    for s in sections:
        sid = s.get('id', '?')
        title = s.get('title', '?')
        covers = s.get('covers', '')
        lines.append(f"- {sid}. {title}: {covers}")
    return "\n".join(lines)


def _format_terms_for_prompt(terms: list) -> str:
    """Format terminology dictionary for inclusion in cached prefix."""
    if not terms:
        return "(не определены)"
    lines = []
    for t in terms:
        term = t.get('term', '')
        english = t.get('english', '')
        if term:
            lines.append(f"- {term} ({english})" if english else f"- {term}")
    return "\n".join(lines) if lines else "(не определены)"


# ---------------------------------------------------------------------------
# Strategy 2: Single-pass (--single-pass flag)
# ---------------------------------------------------------------------------

def _storytell_single_pass(text: str, model: str) -> str:
    """Generate article from full transcript in single LLM call.

    Raises:
        LLMError: If LLM generation fails
    """
    prompt_template = load_prompt_template(
        get_config_path("storytell-prompt.txt")
    )
    prompt = prompt_template.format(text=text)
    return get_completion(prompt, model, max_tokens=get_output_limit(model))


# ---------------------------------------------------------------------------
# Strategy 3: Condense + outline + sections (very long transcripts)
# ---------------------------------------------------------------------------

def _storytell_chunked(text: str, segments, model: str, no_appendix: bool = False) -> str:
    """
    Generate article from text that exceeds single-pass context limit.

    1. Split into time-based chunks (diarized) or size-based chunks (flat)
    2. Condense each chunk into structured notes
    3. Generate outline from condensed notes
    4. Write sections from condensed notes with outline context
    """
    from webinar_processor.utils.transcript_formatter import (
        format_diarized_transcript, split_segments_by_time,
    )

    condense_prompt = load_prompt_template(
        get_config_path("storytell-condense-prompt.txt")
    )

    # Create text chunks
    if segments:
        segment_chunks = split_segments_by_time(segments, chunk_duration_minutes=40)
        text_chunks = [
            format_diarized_transcript(chunk) for chunk in segment_chunks
        ]
        logger.info("Split into %d time-based chunks", len(text_chunks))
    else:
        text_chunks = _chunk_text_by_size(text, model)
        logger.info("Split into %d chunks", len(text_chunks))

    # Pass 1: Condense each chunk
    all_notes = []
    total = len(text_chunks)
    for i, chunk in enumerate(text_chunks):
        logger.info("Condensing chunk %d/%d (%d chars)...", i + 1, total, len(chunk))
        prompt = condense_prompt.format(
            text=chunk, chunk_index=i + 1, total_chunks=total
        )
        notes = _get_completion_safe(prompt, model)
        if notes:
            all_notes.append(f"=== Часть {i+1} из {total} ===\n{notes}")

    if not all_notes:
        logger.error("All chunks failed to condense")
        return None

    # Pass 2: Generate outline + sections from condensed notes
    combined_notes = '\n\n'.join(all_notes)
    logger.info("Condensed to %d chars, generating article...", len(combined_notes))

    # Check if condensed notes fit in context for outline+sections approach
    notes_tokens = count_tokens(model, combined_notes)
    token_limit = TOKEN_LIMITS.get(model, 128000)
    output_limit = get_output_limit(model)
    available = token_limit - output_limit - 3000

    if notes_tokens <= available:
        # Notes fit -- use outline + sections with notes as "transcript"
        return _storytell_with_outline(combined_notes, model, no_appendix)
    else:
        # Notes still too long -- fall back to single-pass from notes
        logger.warning("Condensed notes still exceed context, using single-pass from notes")
        write_prompt = load_prompt_template(
            get_config_path("storytell-from-notes-prompt.txt")
        )
        prompt = write_prompt.format(notes=combined_notes)
        return get_completion(prompt, model, max_tokens=output_limit)


def _chunk_text_by_size(text: str, model: str, target_chunk_tokens: int = 50000) -> list:
    """Split flat text into overlapping chunks for condensation."""
    chunk_chars = int(target_chunk_tokens * 3.5)
    overlap_chars = 5000

    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            boundary = text.rfind('\n\n', start + chunk_chars - 10000, end)
            if boundary == -1:
                boundary = text.rfind('. ', start + chunk_chars - 10000, end)
            if boundary != -1:
                end = boundary + 2
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap_chars

    return chunks


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str):
    """Extract JSON object from LLM response, handling markdown code blocks."""
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _get_completion_safe(prompt: str, model: str) -> Optional[str]:
    """Get LLM completion, returning None on error instead of raising."""
    try:
        return get_completion(prompt, model, max_tokens=get_output_limit(model))
    except LLMError as e:
        logger.error("LLM error: %s", e)
        return None
