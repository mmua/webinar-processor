import click
from webinar_processor.llm import LLMError
from webinar_processor.services.topic_extraction_service import TopicExtractionService
from webinar_processor.commands.base_command import BaseCommand


@click.command()
@click.argument('text_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.option('--language', default="ru")
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def topics(text_file: click.File, language: str, output_file: str):
    """
    Extract a topics list for a text transcript.
    """
    text = text_file.read()

    service = TopicExtractionService()

    click.echo("Extracting topics from transcript...")
    try:
        intermediate_topics, final_topics = service.extract_topics(text, language=language)
    except LLMError as e:
        BaseCommand.handle_llm_error(e, "topics extraction")

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
