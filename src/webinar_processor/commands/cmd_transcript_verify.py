import json
import os

import click

from webinar_processor.llm import LLMConfig, LLMError
from webinar_processor.services.transcript_verifier_service import verify_transcript
from webinar_processor.utils.io import write_output


@click.command('transcript-verify')
@click.argument('transcript_path', type=click.Path(exists=True))
@click.option('--media', required=True, type=click.Path(exists=True), help='Path to audio/video file')
@click.option('--language', default='ru', help='Transcript language (ISO 639-1)')
@click.option('--report', default=None, type=click.Path(), help='Output report path')
@click.option('--model', default=None, help='LLM model for verification')
@click.option('--no-llm', is_flag=True, help='Run heuristics only (no LLM verification)')
def transcript_verify(transcript_path, media, language, report, model, no_llm):
    """Verify transcript quality and detect Whisper hallucinations."""
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        click.echo(click.style(f"Error loading transcript: {e}", fg='red'))
        raise click.Abort()

    if not isinstance(segments, list):
        click.echo(click.style("Transcript must be a JSON array of segments", fg='red'))
        raise click.Abort()

    model = model or LLMConfig.get_model('transcript_verification')

    if report is None:
        report = os.path.join(os.path.dirname(os.path.abspath(transcript_path)), 'verify_report.md')

    try:
        result = verify_transcript(
            segments=segments,
            transcript_path=transcript_path,
            media_path=media,
            model=model,
            no_llm=no_llm,
        )
    except ValueError as e:
        click.echo(click.style(f"Validation error: {e}", fg='red'))
        raise click.Abort()
    except LLMError as e:
        click.echo(click.style(f"LLM error: {e}", fg='red'))
        raise click.Abort()

    write_output(result, report)
