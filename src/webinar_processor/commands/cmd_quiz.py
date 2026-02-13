import json

import click
from webinar_processor.llm import LLMConfig, LLMError
from webinar_processor.utils.completion import get_completion, get_output_limit
from webinar_processor.utils.package import get_config_path
from webinar_processor.utils.io import load_prompt_template, write_output
from webinar_processor.services.transcript_service import load_and_format_transcript


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
    try:
        text, _segments = load_and_format_transcript(asr_file)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        click.echo(click.style(str(e), fg='red'))
        raise click.Abort()

    model = model or LLMConfig.get_model('quiz')

    click.echo(click.style(
        f"Text: {len(text)} chars, Model: {model}", fg='blue'
    ))

    prompt_template = load_prompt_template(
        get_config_path("storytell-quiz-prompt.txt")
    )
    prompt = prompt_template.format(text=text)

    try:
        result = get_completion(prompt, model, max_tokens=get_output_limit(model))
    except LLMError as e:
        click.echo(click.style(f"Error during quiz generation: {e}", fg='red'))
        raise click.Abort()

    write_output(result, output_file)
