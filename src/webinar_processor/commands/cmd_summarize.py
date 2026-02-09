import json
import click
from webinar_processor.llm import LLMConfig, LLMError, TOKEN_LIMITS
from webinar_processor.utils.package import get_config_path
from webinar_processor.utils.openai import get_completion, get_output_limit
from webinar_processor.utils.token import count_tokens
from webinar_processor.commands.base_command import BaseCommand


@click.command()
@click.argument('asr_file', type=click.Path(exists=True), nargs=1)
@click.option('--model', default=None, help='LLM model')
@click.option('--output-file', type=click.Path(exists=False))
def summarize(asr_file: str, model: str, output_file: str):
    """
    Create transcript summary (500-1000 words).

    Accepts diarized transcript (JSON array) or ASR output (JSON with "text").
    Shares prompt prefix with storytell for cache reuse.
    """
    from webinar_processor.utils.transcript_formatter import (
        is_diarized_format, format_diarized_transcript, add_paragraph_breaks,
    )

    data = BaseCommand.load_json_file(asr_file)
    model = model or LLMConfig.get_model('summarization')

    # Format transcript (same logic as storytell)
    if is_diarized_format(data):
        click.echo(click.style("Diarized transcript detected, formatting...", fg='blue'))
        text = format_diarized_transcript(data)
    else:
        text = data.get("text", "")
        if '\n' not in text and len(text) > 1000:
            text = add_paragraph_breaks(text)

    if not text.strip():
        click.echo(click.style("No text to process.", fg='red'))
        raise click.Abort()

    click.echo(click.style(
        f"Text: {len(text)} chars, Model: {model}", fg='blue'
    ))

    prompt_template = BaseCommand.load_prompt_template(
        get_config_path("storytell-summary-prompt.txt")
    )
    prompt = prompt_template.format(text=text)

    try:
        summary = get_completion(prompt, model, max_tokens=get_output_limit(model))
    except LLMError as e:
        BaseCommand.handle_llm_error(e, "summarization")

    BaseCommand.write_output(summary, output_file)


# ---------------------------------------------------------------------------
# storytell command — smart article generation from transcripts
# ---------------------------------------------------------------------------

@click.command()
@click.argument('asr_file', type=click.Path(exists=True), nargs=1)
@click.option('--model', default=None, help='LLM model')
@click.option('--output-file', type=click.Path(exists=False))
@click.option('--no-appendix', is_flag=True, help='Skip appendix (key terms + references)')
@click.option('--single-pass', is_flag=True, help='Use single LLM call (faster, may sacrifice quality)')
def storytell(asr_file: str, model: str, output_file: str, no_appendix: bool, single_pass: bool):
    """
    Transform transcript into educational article.

    Accepts two input formats:
    - Diarized transcript: JSON array with start/end/speaker/text
    - ASR output: JSON object with "text" field

    Default: outline + per-section generation (high quality, prompt-cached).
    --single-pass: one LLM call (faster, cheaper).
    """
    from webinar_processor.utils.transcript_formatter import (
        is_diarized_format, format_diarized_transcript, add_paragraph_breaks,
        split_segments_by_time,
    )

    data = BaseCommand.load_json_file(asr_file)
    model = model or LLMConfig.get_model('story')

    # Format transcript
    if is_diarized_format(data):
        click.echo(click.style("Diarized transcript detected, formatting...", fg='blue'))
        text = format_diarized_transcript(data)
        segments = data
    else:
        text = data.get("text", "")
        segments = None
        if '\n' not in text and len(text) > 1000:
            click.echo(click.style("Flat text, adding paragraph breaks...", fg='blue'))
            text = add_paragraph_breaks(text)

    if not text.strip():
        click.echo(click.style("No text to process.", fg='red'))
        raise click.Abort()

    text_tokens = count_tokens(model, text)
    token_limit = TOKEN_LIMITS.get(model, 128000)
    output_limit = get_output_limit(model)
    prompt_overhead = 3000
    available_for_input = token_limit - output_limit - prompt_overhead

    click.echo(click.style(
        f"Text: {len(text)} chars ({text_tokens} tokens), "
        f"Model: {model} (context={token_limit}, output={output_limit})",
        fg='blue'
    ))

    # Choose generation strategy
    if single_pass:
        click.echo(click.style("Single-pass mode...", fg='green'))
        result = _storytell_single_pass(text, model)
    elif text_tokens <= available_for_input:
        click.echo(click.style("Outline + per-section mode (cached prefix)...", fg='green'))
        result = _storytell_with_outline(text, model, no_appendix)
    else:
        click.echo(click.style(
            f"Text exceeds context ({text_tokens} > {available_for_input} tokens), "
            "condensing first...",
            fg='yellow'
        ))
        result = _storytell_chunked(text, segments, model, no_appendix)

    if not result:
        click.echo(click.style("Generation failed", fg='red'))
        raise click.Abort()

    click.echo(click.style(f"Done: {len(result)} chars", fg='green'))
    BaseCommand.write_output(result, output_file)


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

    Calls 2 through N+1 share an identical prefix → cached at 1/10th cost.
    """
    # Step 1: Generate outline + terminology (one call, full price)
    click.echo(click.style("Step 1: Generating outline + terms...", fg='cyan'))
    outline_prompt = BaseCommand.load_prompt_template(
        get_config_path("storytell-outline-prompt.txt")
    )
    outline_response = _get_completion_safe(
        outline_prompt.format(text=text), model
    )
    if not outline_response:
        click.echo(click.style("Outline generation failed, falling back to single-pass", fg='yellow'))
        return _storytell_single_pass(text, model)

    outline = _extract_json(outline_response)
    if not outline or not outline.get('sections'):
        click.echo(click.style("Could not parse outline, falling back to single-pass", fg='yellow'))
        return _storytell_single_pass(text, model)

    chapter_title = outline.get('chapter_title', 'Глава')
    sections = outline.get('sections', [])
    terms = outline.get('terms', [])

    click.echo(click.style(
        f"Outline: \"{chapter_title}\" — {len(sections)} sections, {len(terms)} terms",
        fg='blue'
    ))
    for s in sections:
        click.echo(click.style(f"  {s.get('id', '?')}: {s.get('title', '?')}", fg='blue'))

    # Build shared prefix (identical for all subsequent calls → cached)
    cached_prefix = _build_cached_prefix(text, sections, terms)
    prefix_tokens = count_tokens(model, cached_prefix)
    click.echo(click.style(
        f"Cached prefix: {prefix_tokens} tokens (cached after first section call)",
        fg='blue'
    ))

    # Step 2: Generate each section (prefix cached from 2nd call onward)
    section_task_template = BaseCommand.load_prompt_template(
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

        click.echo(click.style(
            f"Step 2: Section {i+1}/{len(sections)}: {section_title}...",
            fg='cyan'
        ))

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
            click.echo(click.style(
                f"  {section_id} done: {len(section_text)} chars", fg='green'
            ))
        else:
            click.echo(click.style(
                f"  {section_id} failed, skipping", fg='red'
            ))

    # Step 3: Generate appendix (same cached prefix)
    if not no_appendix:
        click.echo(click.style("Step 3: Generating appendix...", fg='cyan'))
        appendix_task = BaseCommand.load_prompt_template(
            get_config_path("storytell-appendix-task.txt")
        )
        prompt = cached_prefix + appendix_task
        appendix = _get_completion_safe(prompt, model)
        if appendix and appendix.strip():
            parts.append(f"\n\n---\n\n{appendix.strip()}")
            click.echo(click.style(
                f"  Appendix done: {len(appendix)} chars", fg='green'
            ))

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
    """Generate article from full transcript in single LLM call."""
    prompt_template = BaseCommand.load_prompt_template(
        get_config_path("storytell-prompt.txt")
    )
    prompt = prompt_template.format(text=text)

    try:
        return get_completion(prompt, model, max_tokens=get_output_limit(model))
    except LLMError as e:
        BaseCommand.handle_llm_error(e, "storytell")


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
        format_diarized_transcript,
    )

    condense_prompt = BaseCommand.load_prompt_template(
        get_config_path("storytell-condense-prompt.txt")
    )

    # Create text chunks
    if segments:
        segment_chunks = split_segments_by_time(segments, chunk_duration_minutes=40)
        text_chunks = [
            format_diarized_transcript(chunk) for chunk in segment_chunks
        ]
        click.echo(click.style(
            f"Split into {len(text_chunks)} time-based chunks", fg='blue'
        ))
    else:
        text_chunks = _chunk_text_by_size(text, model)
        click.echo(click.style(
            f"Split into {len(text_chunks)} chunks", fg='blue'
        ))

    # Pass 1: Condense each chunk
    all_notes = []
    total = len(text_chunks)
    for i, chunk in enumerate(text_chunks):
        click.echo(click.style(
            f"Condensing chunk {i+1}/{total} ({len(chunk)} chars)...", fg='cyan'
        ))
        prompt = condense_prompt.format(
            text=chunk, chunk_index=i + 1, total_chunks=total
        )
        notes = _get_completion_safe(prompt, model)
        if notes:
            all_notes.append(f"=== Часть {i+1} из {total} ===\n{notes}")

    if not all_notes:
        click.echo(click.style("All chunks failed to condense", fg='red'))
        return None

    # Pass 2: Generate outline + sections from condensed notes
    combined_notes = '\n\n'.join(all_notes)
    click.echo(click.style(
        f"Condensed to {len(combined_notes)} chars, generating article...",
        fg='green'
    ))

    # Check if condensed notes fit in context for outline+sections approach
    notes_tokens = count_tokens(model, combined_notes)
    token_limit = TOKEN_LIMITS.get(model, 128000)
    output_limit = get_output_limit(model)
    available = token_limit - output_limit - 3000

    if notes_tokens <= available:
        # Notes fit — use outline + sections with notes as "transcript"
        return _storytell_with_outline(combined_notes, model, no_appendix)
    else:
        # Notes still too long — fall back to single-pass from notes
        click.echo(click.style(
            "Condensed notes still exceed context, using single-pass from notes",
            fg='yellow'
        ))
        write_prompt = BaseCommand.load_prompt_template(
            get_config_path("storytell-from-notes-prompt.txt")
        )
        prompt = write_prompt.format(notes=combined_notes)
        try:
            return get_completion(prompt, model, max_tokens=output_limit)
        except LLMError as e:
            BaseCommand.handle_llm_error(e, "article from notes")


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
    import re as _re
    match = _re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, _re.DOTALL)
    if match:
        text = match.group(1)
    match = _re.search(r'\{.*\}', text, _re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _get_completion_safe(prompt: str, model: str) -> str:
    """Get LLM completion, returning None on error instead of raising."""
    try:
        return get_completion(prompt, model, max_tokens=get_output_limit(model))
    except LLMError as e:
        click.echo(click.style(f"LLM error: {e}", fg='red'))
        return None


@click.command()
@click.argument('asr_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.option('--output-file', type=click.Path(exists=False))
def raw_text(asr_file: click.File, output_file: str):
    """Write raw transcript text."""
    data = json.load(asr_file)
    BaseCommand.write_output(data["text"], output_file)
