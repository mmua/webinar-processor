import json
import os
from typing import List
import click
import openai
from dotenv import load_dotenv, find_dotenv

from webinar_processor.utils.package import get_config_path
from webinar_processor.utils.openai import create_summary_with_context, text_transform

_ = load_dotenv(find_dotenv())

DEFAULT_SUMMARIZATION_MODEL="gpt-4.1-mini"
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
    else:
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
        
        try:
            processed_chunk = text_transform(chunk, language, model, contextual_prompt)
            all_processed_parts.append(processed_chunk)
            previous_output = processed_chunk
            click.echo(click.style(f"✓ Chunk {i+1} processed successfully", fg='green'))
        except Exception as e:
            click.echo(click.style(f"Error processing chunk {i+1}: {str(e)}", fg='red'))
            # Use raw chunk as fallback
            all_processed_parts.append(chunk)
            previous_output = chunk
            click.echo(click.style("Using raw chunk text and continuing.", fg='yellow'))
    
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
            click.echo(click.style(f"Error writing to output file {output_file_path}: {e}", fg='red'))
            click.echo("\nProcessed story:\n")
            click.echo(content)
    else:
        click.echo("\nProcessed story:\n")
        click.echo(content)


@click.command()
@click.argument('asr_path', nargs=1)
@click.argument('topics_path', nargs=1, default='')
@click.argument('model', nargs=1, default=DEFAULT_SUMMARIZATION_MODEL)
@click.option('--language', default="ru")
@click.option('--prompt-file', type=click.Path(exists=True), help='Path to a file containing the prompt')
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def summarize(asr_path: str, topics_path: str, model: str, language: str, prompt_file: str, output_file: str):
    """
    Create transcript summary
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        click.echo(click.style('Error: OpenAI keys is not set', fg='red'))
        raise click.Abort

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

    summary = create_summary_with_context(text, context, language, model, prompt_template)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as of:
            of.write(summary)
    else:
        click.echo(summary)


@click.command()
@click.argument('asr_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('model', nargs=1, default=DEFAULT_SUMMARIZATION_MODEL)
@click.option('--language', default="ru")
@click.option('--prompt-file', type=click.Path(exists=True), help='Path to a file containing the prompt template (e.g., with a {{text}} placeholder). Default: conf/long-story-chunked.txt (optimized for chunked processing)')
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
@click.option('--chunk-size', default=16000, help='Target chunk size in characters (default: 16000)')
def storytell(asr_file: click.File, model: str, language: str, prompt_file: str, output_file: str, chunk_size: int):
    """
    Create a story from ASR output, processing in chunks for large texts.
    
    The function automatically handles context between chunks using a unified prompt template.
    Uses conf/long-story-chunked.txt by default, which is optimized for chunked processing.
    The prompt file should contain a template with a {{text}} placeholder.
    
    For single-chunk processing, use conf/long-story.txt instead.
    """
    # Validate API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        click.echo(click.style('Error: OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.', fg='red'))
        raise click.Abort

    # Load and validate input data
    try:
        data = json.load(asr_file)
    except json.JSONDecodeError:
        click.echo(click.style(f"Error: Could not decode JSON from {asr_file.name}.", fg='red'))
        raise click.Abort

    # Load prompt template
    if not prompt_file:
        # Default to the chunked processing prompt that works better with our context system
        # Alternative: get_config_path("long-story.txt") for the original single-chunk prompt
        prompt_file = get_config_path("long-story-chunked.txt")

    try:
        with open(prompt_file, "r", encoding="utf-8") as pf:
            user_core_prompt_template = pf.read()
    except FileNotFoundError:
        click.echo(click.style(f"Error: Prompt file not found at {prompt_file}", fg='red'))
        raise click.Abort

    # Extract chunks using the best available strategy
    raw_text_chunks = extract_text_chunks(data, chunk_size, language)
    
    if not raw_text_chunks:
        click.echo(click.style("Warning: No text chunks to process.", fg='yellow'))
        write_output("", output_file)
        return

    # Process all chunks and combine results
    final_story = process_chunks(raw_text_chunks, user_core_prompt_template, language, model)

    # Write output
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