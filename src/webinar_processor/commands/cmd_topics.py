import os
import click
from dotenv import load_dotenv, find_dotenv
from webinar_processor.llm import LLMConfig
from webinar_processor.utils.openai import create_summary, create_summary_with_context
from webinar_processor.utils.package import get_config_path

_ = load_dotenv(find_dotenv())

@click.command()
@click.argument('text_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.option('--language', default="ru")
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def topics(text_file: click.File, language: str, output_file: str):
    """
    Extract a topics list for a text transcript.
    """
    text = text_file.read()
    model = LLMConfig.get_model('topics')

    intermediate_prompt_path = get_config_path("intermediate-topics-prompt.txt")
    final_prompt_path = get_config_path("final-topics-prompt-onepass.txt")

    try:
        with open(intermediate_prompt_path, "r", encoding="utf-8") as pf:
            intermediate_prompt_template = pf.read()
    except FileNotFoundError:
        click.echo(click.style(f'Error: Intermediate prompt file not found at {intermediate_prompt_path}', fg='red'))
        raise click.Abort

    click.echo("Extracting topics from transcript...")
    try:
        intermediate_topics = create_summary(text, language, model, intermediate_prompt_template)
    except Exception as e:
        click.echo(click.style(f'Error extracting topics: {e}', fg='red'))
        raise click.Abort

    try:
        with open(final_prompt_path, "r", encoding="utf-8") as pf:
            final_prompt_template = pf.read()
    except FileNotFoundError:
        click.echo(click.style(f'Error: Final prompt file not found at {final_prompt_path}', fg='red'))
        raise click.Abort

    click.echo("Refining topics...")
    try:
        final_topics = create_summary_with_context(text, intermediate_topics, language, model, final_prompt_template)
    except Exception as e:
        click.echo(click.style(f'Error refining topics: {e}', fg='red'))
        raise click.Abort

    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as of:
                of.write(final_topics)
            with open(output_file + ".im", "w", encoding="utf-8") as of:
                of.write(intermediate_topics)
            click.echo(f"Topics written to {output_file}")
        except IOError as e:
            click.echo(click.style(f'Error writing output file: {e}', fg='red'))
            raise click.Abort
    else:
        click.echo(final_topics)
