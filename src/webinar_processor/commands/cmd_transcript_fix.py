import json
import os

import click

from webinar_processor.llm import LLMConfig, LLMError
from webinar_processor.services.transcript_fixer_service import fix_transcript


@click.command('transcript-fix')
@click.argument('transcript_path', type=click.Path(exists=True))
@click.option('--media', required=True, type=click.Path(exists=True), help='Path to audio/video file')
@click.option('--report', required=True, type=click.Path(exists=True), help='Path to verification report')
@click.option('--language', default='ru', help='Transcript language (ISO 639-1)')
@click.option('--out', default=None, type=click.Path(), help='Output fixed transcript path')
@click.option('--fix-report', default=None, type=click.Path(), help='Output fix report path')
@click.option('--model', default=None, help='LLM model for reconstruction')
@click.option('--include-open', is_flag=True, help='Also process open issues (not just accepted)')
def transcript_fix(transcript_path, media, report, language, out, fix_report, model, include_open):
    """Fix transcript issues identified in verification report."""
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        click.echo(click.style(f"Error loading transcript: {e}", fg='red'))
        raise click.Abort()

    if not isinstance(segments, list):
        click.echo(click.style("Transcript must be a JSON array of segments", fg='red'))
        raise click.Abort()

    try:
        with open(report, 'r', encoding='utf-8') as f:
            report_text = f.read()
    except (FileNotFoundError, IOError) as e:
        click.echo(click.style(f"Error loading report: {e}", fg='red'))
        raise click.Abort()

    model = model or LLMConfig.get_model('transcript_fix')
    transcript_dir = os.path.dirname(os.path.abspath(transcript_path))

    if out is None:
        out = os.path.join(transcript_dir, 'transcript.fixed.json')
    if fix_report is None:
        fix_report = os.path.join(transcript_dir, 'fix_report.md')

    try:
        fixed_segments, fix_report_text = fix_transcript(
            segments=segments,
            transcript_path=transcript_path,
            media_path=media,
            report_text=report_text,
            model=model,
            language=language,
            include_open=include_open,
        )
    except ValueError as e:
        click.echo(click.style(f"Error: {e}", fg='red'))
        raise click.Abort()
    except (LLMError, json.JSONDecodeError) as e:
        click.echo(click.style(f"LLM error: {e}", fg='red'))
        raise click.Abort()

    # Write fixed transcript
    try:
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(fixed_segments, f, ensure_ascii=False, indent=2)
        click.echo(click.style(f"Fixed transcript written to {out}", fg='green'))
    except IOError as e:
        click.echo(click.style(f"Error writing output: {e}", fg='red'))
        raise click.Abort()

    # Write fix report
    try:
        with open(fix_report, 'w', encoding='utf-8') as f:
            f.write(fix_report_text)
        click.echo(click.style(f"Fix report written to {fix_report}", fg='green'))
    except IOError as e:
        click.echo(click.style(f"Error writing fix report: {e}", fg='red'))
        raise click.Abort()
