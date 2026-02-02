import json
import os
from typing import List
import click
from dotenv import load_dotenv, find_dotenv

from webinar_processor.llm import LLMConfig
from webinar_processor.utils.package import get_config_path
from webinar_processor.utils.openai import create_summary_with_context, text_transform

_ = load_dotenv(find_dotenv())
NO_PREVIOUS_CONTENT_TOKEN = "[NO_PREVIOUS_CONTENT]"

def smart_chunk_text(text: str, target_size: int, language: str = "ru") -> List[str]:
    """
    Split text into chunks, preferring to break at sentence boundaries.
    """
    if len(text) <= target_size:
        return [text]

    chunks = []
    current_chunk = ""

    # Split by sentences (basic approach for Russian)
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
        # If adding this sentence would exceed target size, finalize current chunk
        if current_chunk and len(current_chunk) + len(sentence) + 1 > target_size:
            chunks.append(current_chunk.strip())
            current_chunk = ""

        current_chunk += (sentence + " ")

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def extract_text_chunks(data: dict, chunk_size: int, language: str) -> List[str]:
    """
    Extract text chunks from ASR data using the best available strategy.
    """
    asr_segments = data.get("segments")
    full_text_from_asr = data.get("text", "")

    if asr_segments:
        click.echo(click.style("Using ASR segments for chunking.", fg='blue'))
        return _chunk_from_segments(asr_segments, chunk_size, full_text_from_asr)
    elif full_text_from_asr.strip():
        click.echo(click.style("No ASR segments found, using smart text chunking.", fg='yellow'))
        return smart_chunk_text(full_text_from_asr, chunk_size, language)
    return []


def _chunk_from_segments(asr_segments: List[dict], chunk_size: int, fallback_text: str) -> List[str]:
    """
    Create chunks from ASR segments, respecting chunk size limits.
    """
    raw_text_chunks = []
    current_chunk_text = ""

    for segment in asr_segments:
        segment_text = segment.get("text", "").strip()
        if not segment_text:
            continue

        # If adding this segment would make the current chunk too long, finalize current chunk
        if current_chunk_text and (len(current_chunk_text) + len(segment_text) + 1 > chunk_size):
            raw_text_chunks.append(current_chunk_text)
            current_chunk_text = ""

        current_chunk_text += (segment_text + " ")

    if current_chunk_text.strip():
        raw_text_chunks.append(current_chunk_text.strip())

    # Fallback if segments were empty or didn't form chunks
    if not raw_text_chunks and fallback_text.strip():
        raw_text_chunks = [fallback_text.strip()]

    return raw_text_chunks


def build_unified_prompt_template(context_snippet: str, core_prompt: str) -> str:
    """
    Build a unified prompt template that handles both first and subsequent chunks.
    """
    return f"""Ты обрабатываешь большой документ по частям. 

ПРЕДЫДУЩИЙ ОБРАБОТАННЫЙ ФРАГМЕНТ (для контекста):
{context_snippet}

Теперь обработай СЛЕДУЮЩИЙ фрагмент, продолжая логично от предыдущего (если предыдущий фрагмент равен {NO_PREVIOUS_CONTENT_TOKEN}, то это начало документа):

{core_prompt}

Убедись, что твой ответ логично продолжает предыдущий обработанный фрагмент (если он есть)."""

def build_contextual_prompt(chunk_index: int, previous_output: str, core_prompt: str) -> str:
    """
    Build an appropriate prompt for the current chunk based on its position and context.
    Uses a unified template with NO_PREVIOUS_CONTENT token for first chunk.
    """
    if chunk_index == 0:
        # For the first chunk, use the token to indicate no previous content
        context_snippet = NO_PREVIOUS_CONTENT_TOKEN
    else:
        # For subsequent chunks, provide context from previous output
        context_snippet = previous_output[-500:] if len(previous_output) > 500 else previous_output
    return build_unified_prompt_template(context_snippet, core_prompt)


def process_chunks(chunks: List[str], core_prompt: str, language: str, model: str) -> str:
    """
    Process all chunks and return the combined result.
    """
    if not chunks:
        click.echo(click.style("Warning: No text chunks to process.", fg='yellow'))
        return ""

    all_processed_parts = []
    previous_output = ""

    click.echo(click.style(f"Processing {len(chunks)} text chunks...", fg='green'))

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        click.echo(click.style(f"Processing chunk {i+1}/{len(chunks)} (length: {len(chunk)} chars)...", fg='cyan'))

        contextual_prompt = build_contextual_prompt(i, previous_output, core_prompt)

        processed_chunk = text_transform(chunk, language, model, contextual_prompt)
        all_processed_parts.append(processed_chunk)
        previous_output = processed_chunk
        click.echo(click.style(f"✓ Chunk {i+1} processed successfully", fg='green'))

    return "\n\n".join(all_processed_parts)


def write_output(content: str, output_file_path: str) -> None:
    """
    Write content to file or console.
    """
    if output_file_path:
        try:
            with open(output_file_path, "w", encoding="utf-8") as of:
                of.write(content)
            click.echo(click.style(f"Successfully generated story to {output_file_path}", fg='green'))
        except IOError as e:
            click.echo(click.style(f'Error writing to output file {output_file_path}: {e}', fg='red'))
            click.echo("\nProcessed story:\n")
            click.echo(content)
    else:
        click.echo("\nProcessed story:\n")
        click.echo(content)


@click.command()
@click.argument('asr_path', nargs=1)
@click.argument('topics_path', nargs=1, default='')
@click.argument('model', nargs=1, default=None)
@click.option('--language', default="ru")
@click.option('--prompt-file', type=click.Path(exists=True), help='Path to a file containing the prompt')
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def summarize(asr_path: str, topics_path: str, model: str, language: str, prompt_file: str, output_file: str):
    """
    Create transcript summary
    """
    with open(asr_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = data["text"]

    if not prompt_file:
        prompt_file = get_config_path("short-summary-with-context.txt")

    with open(prompt_file, "r", encoding="utf-8") as pf:
        prompt_template = pf.read()

    if not topics_path:
        asr_dir = os.path.dirname(asr_path)
        topics_path = os.path.join(asr_dir, "topics.txt")

    if not (topics_path and os.path.exists(topics_path)):
        click.echo(click.style('Topics file not found', fg='red'))
        raise click.Abort

    with open(topics_path, encoding="utf-8") as context_file:
        context = context_file.read()

    model = model or LLMConfig.get_model('summarization')
    try:
        summary = create_summary_with_context(text, context, language, model, prompt_template)
    except Exception as e:
        click.echo(click.style(f'Error generating summary: {e}', fg='red'))
        raise click.Abort

    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as of:
                of.write(summary)
        except IOError as e:
            click.echo(click.style(f'Error writing output file: {e}', fg='red'))
            raise click.Abort
    else:
        click.echo(summary)


@click.command()
@click.argument('asr_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('model', nargs=1, default=None)
@click.option('--language', default="ru")
@click.option('--prompt-file', type=click.Path(exists=True), help='Path to a file containing a prompt template')
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
@click.option('--chunk-size', default=16000, help='Target chunk size in characters')
def storytell(asr_file: click.File, model: str, language: str, prompt_file: str, output_file: str, chunk_size: int):
    """
    Create a story from ASR output, processing in chunks for large texts.

    The function automatically handles context between chunks using a unified prompt template.
    Uses conf/long-story-chunked.txt by default, which is optimized for chunked processing.
    The prompt file should contain a template with a {{text}} placeholder.

    For single-chunk processing, use conf/long-story.txt instead.
    """
    data = json.load(asr_file)

    if not prompt_file:
        prompt_file = get_config_path("long-story-chunked.txt")

    with open(prompt_file, "r", encoding="utf-8") as pf:
        user_core_prompt_template = pf.read()

    raw_text_chunks = extract_text_chunks(data, chunk_size, language)

    if not raw_text_chunks:
        click.echo(click.style("Warning: No text chunks to process.", fg='yellow'))
        write_output("", output_file)
        return

    model = model or LLMConfig.get_model('story')
    try:
        final_story = process_chunks(raw_text_chunks, user_core_prompt_template, language, model)
    except Exception as e:
        click.echo(click.style(f'Error processing story: {e}', fg='red'))
        raise click.Abort
    write_output(final_story, output_file)


@click.command()
@click.argument('asr_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def raw_text(asr_file: click.File, output_file: str):
    """
    Write raw transcript text
    """
    data = json.load(asr_file)
    story = data["text"]

    if output_file:
        with open(output_file, "w", encoding="utf-8") as of:
            of.write(story)
    else:
        click.echo(story)
