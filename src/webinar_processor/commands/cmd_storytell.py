import json

import click
from webinar_processor.llm import LLMConfig, LLMError
from webinar_processor.utils.io import write_output
from webinar_processor.services.transcript_service import load_and_format_transcript
from webinar_processor.services.storytell_service import generate_article


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
    try:
        text, segments = load_and_format_transcript(asr_file)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        click.echo(click.style(str(e), fg='red'))
        raise click.Abort()

    model = model or LLMConfig.get_model('story')

    try:
        result = generate_article(
            text, model,
            segments=segments,
            no_appendix=no_appendix,
            single_pass=single_pass,
        )
    except LLMError as e:
        click.echo(click.style(f"Error during storytell: {e}", fg='red'))
        raise click.Abort()

    if not result:
        click.echo(click.style("Generation failed", fg='red'))
        raise click.Abort()

    click.echo(click.style(f"Done: {len(result)} chars", fg='green'))
    write_output(result, output_file)
