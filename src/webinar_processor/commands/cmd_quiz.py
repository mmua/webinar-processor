import click
from webinar_processor.llm import LLMConfig, LLMError
from webinar_processor.utils.openai import get_completion, get_output_limit
from webinar_processor.utils.package import get_config_path
from webinar_processor.commands.base_command import BaseCommand


@click.command()
@click.argument('asr_file', type=click.Path(exists=True), nargs=1)
@click.option('--model', default=None, help='LLM model')
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def quiz(asr_file: str, model: str, output_file: str):
    """
    Create a quiz based on the transcript.

    Accepts diarized transcript (JSON array) or ASR output (JSON with "text").
    Shares prompt prefix with storytell/summarize for cache reuse.
    """
    from webinar_processor.utils.transcript_formatter import (
        is_diarized_format, format_diarized_transcript, add_paragraph_breaks,
    )

    data = BaseCommand.load_json_file(asr_file)
    model = model or LLMConfig.get_model('quiz')

    # Format transcript (same logic as storytell/summarize)
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
        get_config_path("storytell-quiz-prompt.txt")
    )
    prompt = prompt_template.format(text=text)

    try:
        result = get_completion(prompt, model, max_tokens=get_output_limit(model))
    except LLMError as e:
        BaseCommand.handle_llm_error(e, "quiz generation")

    BaseCommand.write_output(result, output_file)
