import json
import os
from typing import List
import click
from webinar_processor.llm import LLMConfig, LLMError
from webinar_processor.utils.package import get_config_path
from webinar_processor.utils.openai import create_summary_with_context, get_completion, get_output_limit
from webinar_processor.commands.base_command import BaseCommand


# Chunking defaults
DEFAULT_CHUNK_SIZE = 48000    # chars (~12K tokens)
DEFAULT_OVERLAP_SIZE = 3000   # chars overlap between chunks
CONTEXT_PREV_SIZE = 2000      # chars from previous output
CONTEXT_NEXT_SIZE = 1000      # chars lookahead into next chunk


def smart_chunk_text(text: str, target_size: int) -> List[dict]:
    """Split text into chunks at sentence boundaries. No timestamp info available."""
    if len(text) <= target_size:
        return [{"text": text, "start": None, "end": None}]

    chunks = []
    current_chunk = ""
    sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']

    sentences = []
    current_sentence = ""
    for char in text:
        current_sentence += char
        if any(current_sentence.endswith(ending) for ending in sentence_endings):
            sentences.append(current_sentence.strip())
            current_sentence = ""
    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    for sentence in sentences:
        if current_chunk and len(current_chunk) + len(sentence) + 1 > target_size:
            chunks.append({"text": current_chunk.strip(), "start": None, "end": None})
            current_chunk = ""
        current_chunk += (sentence + " ")

    if current_chunk.strip():
        chunks.append({"text": current_chunk.strip(), "start": None, "end": None})

    return chunks


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    if seconds is None:
        return "00:00"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def chunk_from_segments(asr_segments: List[dict], chunk_size: int, fallback_text: str) -> List[dict]:
    """Create chunks from ASR segments, preserving timestamp info."""
    chunks = []
    current_text = ""
    current_start = None
    current_end = None

    for segment in asr_segments:
        text = segment.get("text", "").strip()
        if not text:
            continue

        seg_start = segment.get("start")
        seg_end = segment.get("end")

        if current_text and len(current_text) + len(text) + 1 > chunk_size:
            chunks.append({
                "text": current_text.strip(),
                "start": current_start,
                "end": current_end
            })
            current_text = ""
            current_start = None
            current_end = None

        if current_start is None:
            current_start = seg_start
        current_end = seg_end
        current_text += (text + " ")

    if current_text.strip():
        chunks.append({
            "text": current_text.strip(),
            "start": current_start,
            "end": current_end
        })

    if not chunks and fallback_text.strip():
        chunks = [{"text": fallback_text.strip(), "start": None, "end": None}]

    return chunks


def apply_overlap(chunks: List[dict], overlap: int) -> List[dict]:
    """Add overlap from end of each chunk to beginning of next."""
    if len(chunks) <= 1 or overlap <= 0:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_text = chunks[i - 1]["text"]
        overlap_text = prev_text[-overlap:] if len(prev_text) > overlap else prev_text
        result.append({
            "text": overlap_text + " " + chunks[i]["text"],
            "start": chunks[i]["start"],  # Keep original start time
            "end": chunks[i]["end"]
        })
    return result


def extract_chunks(data, chunk_size: int, overlap: int) -> List[dict]:
    """Extract text chunks from ASR data. Handles both dict and list formats.

    Returns list of dicts: [{"text": str, "start": float|None, "end": float|None}, ...]
    """
    # Handle list format: [{text, start, end, speaker}, ...]
    if isinstance(data, list):
        click.echo(click.style("Using segment list for chunking (with timestamps).", fg='blue'))
        chunks = chunk_from_segments(data, chunk_size, "")
    # Handle dict format: {text: "...", segments: [...]}
    elif isinstance(data, dict):
        segments = data.get("segments")
        full_text = data.get("text", "")
        if segments:
            click.echo(click.style("Using ASR segments for chunking (with timestamps).", fg='blue'))
            chunks = chunk_from_segments(segments, chunk_size, full_text)
        elif full_text.strip():
            click.echo(click.style("Using full text for chunking (no timestamps).", fg='yellow'))
            chunks = smart_chunk_text(full_text, chunk_size)
        else:
            return []
    else:
        return []

    return apply_overlap(chunks, overlap)


def process_chunks(chunks: List[dict], prompt_template: str, model: str, language: str = "ru", topics: str = "") -> str:
    """
    Process chunks with simple placeholder substitution.

    Chunks are dicts: {"text": str, "start": float|None, "end": float|None}

    Prompt template may have: {position}, {time_range}, {topics}, {prev_context}, {next_context}, {text}
    Only {text} is required; others are optional.
    """
    if not chunks:
        return ""

    results = []
    prev_output = ""
    total = len(chunks)

    click.echo(click.style(f"Processing {total} chunks (language: {language})...", fg='green'))

    for i, chunk_data in enumerate(chunks):
        chunk_text = chunk_data["text"]
        chunk_start = chunk_data.get("start")
        chunk_end = chunk_data.get("end")

        if not chunk_text.strip():
            continue

        # Build time range string
        if chunk_start is not None and chunk_end is not None:
            time_range = f"{format_timestamp(chunk_start)} - {format_timestamp(chunk_end)}"
        else:
            time_range = "[время недоступно]"

        click.echo(click.style(f"Chunk {i+1}/{total} ({len(chunk_text)} chars, {time_range})...", fg='cyan'))

        # Build context values
        position = f"Фрагмент {i+1} из {total}"

        if i == 0:
            prev_context = "[НАЧАЛО ДОКУМЕНТА]"
        else:
            prev_context = prev_output[-CONTEXT_PREV_SIZE:] if len(prev_output) > CONTEXT_PREV_SIZE else prev_output

        if i + 1 < total:
            next_text = chunks[i + 1]["text"]
            next_context = next_text[:CONTEXT_NEXT_SIZE] if len(next_text) > CONTEXT_NEXT_SIZE else next_text
        else:
            next_context = "[КОНЕЦ ДОКУМЕНТА]"

        # Substitution with optional placeholders
        try:
            prompt = prompt_template.format(
                position=position,
                time_range=time_range,
                topics=topics or "[НЕТ]",
                prev_context=prev_context,
                next_context=next_context,
                text=chunk_text
            )
        except KeyError:
            # Fallback: only {text} placeholder
            prompt = prompt_template.format(text=chunk_text)

        click.echo(click.style(f"Prompt size: {len(prompt)} chars", fg='yellow'))

        try:
            result = get_completion(prompt, model, max_tokens=get_output_limit(model))
        except LLMError as e:
            click.echo(click.style(f"Chunk {i+1} LLM error: {e}", fg='red'))
            raise click.Abort()

        if result:
            results.append(result)
            prev_output = result
            click.echo(click.style(f"Chunk {i+1} done. Output: {len(result)} chars, starts: {result[:50]!r}", fg='green'))
        else:
            click.echo(click.style(f"Chunk {i+1} failed! Empty result.", fg='red'))
            raise click.Abort()

    click.echo(click.style(f"Total results: {len(results)}, sizes: {[len(r) for r in results]}", fg='blue'))
    return "\n\n".join(results)


def load_topics(topics_file: str, asr_path: str) -> str:
    """Load topics from file or auto-detect next to ASR file."""
    if topics_file:
        with open(topics_file, "r", encoding="utf-8") as f:
            return f.read()

    # Auto-detect
    asr_dir = os.path.dirname(asr_path)
    auto_path = os.path.join(asr_dir, "topics.txt")
    if os.path.exists(auto_path):
        click.echo(click.style(f"Auto-detected: {auto_path}", fg='blue'))
        with open(auto_path, "r", encoding="utf-8") as f:
            return f.read()

    return ""


@click.command()
@click.argument('asr_path', nargs=1)
@click.argument('topics_path', nargs=1, default='')
@click.option('--model', default=None, help='LLM model')
@click.option('--language', default="ru")
@click.option('--prompt-file', type=click.Path(exists=True))
@click.option('--output-file', type=click.Path(exists=False))
def summarize(asr_path: str, topics_path: str, model: str, language: str, prompt_file: str, output_file: str):
    """Create transcript summary."""
    data = BaseCommand.load_json_file(asr_path)
    text = data["text"]

    prompt_file = prompt_file or get_config_path("short-summary-with-context.txt")
    prompt_template = BaseCommand.load_prompt_template(prompt_file)

    if not topics_path:
        asr_dir = os.path.dirname(asr_path)
        topics_path = os.path.join(asr_dir, "topics.txt")

    if not os.path.exists(topics_path):
        click.echo(click.style('Topics file not found', fg='red'))
        raise click.Abort

    with open(topics_path, encoding="utf-8") as f:
        context = f.read()

    model = model or LLMConfig.get_model('summarization')

    try:
        summary = create_summary_with_context(text, context, language, model, prompt_template)
    except LLMError as e:
        BaseCommand.handle_llm_error(e, "summarization")

    BaseCommand.write_output(summary, output_file)


@click.command()
@click.argument('asr_file', type=click.Path(exists=True), nargs=1)
@click.option('--model', default=None, help='LLM model')
@click.option('--language', default="ru", help='Language code (default: ru)')
@click.option('--prompt-file', type=click.Path(exists=True))
@click.option('--output-file', type=click.Path(exists=False))
@click.option('--topics-file', type=click.Path(exists=True))
@click.option('--chunk-size', default=DEFAULT_CHUNK_SIZE)
@click.option('--overlap', default=DEFAULT_OVERLAP_SIZE)
def storytell(asr_file: str, model: str, language: str, prompt_file: str, output_file: str, topics_file: str, chunk_size: int, overlap: int):
    """
    Transform transcript into academic-style text.

    Processes in chunks with overlap for continuity.
    Auto-detects topics.txt next to ASR file.
    """
    data = BaseCommand.load_json_file(asr_file)

    prompt_file = prompt_file or get_config_path("long-story-chunked.txt")
    prompt_template = BaseCommand.load_prompt_template(prompt_file)

    topics = load_topics(topics_file, asr_file)
    chunks = extract_chunks(data, chunk_size, overlap)

    if not chunks:
        click.echo(click.style("No text to process.", fg='yellow'))
        return

    click.echo(click.style(f"Chunks: {len(chunks)}, Size: {chunk_size}, Overlap: {overlap}", fg='blue'))

    model = model or LLMConfig.get_model('story')

    result = process_chunks(chunks, prompt_template, model, language, topics)
    BaseCommand.write_output(result, output_file)


@click.command()
@click.argument('asr_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.option('--output-file', type=click.Path(exists=False))
def raw_text(asr_file: click.File, output_file: str):
    """Write raw transcript text."""
    data = json.load(asr_file)
    BaseCommand.write_output(data["text"], output_file)
