import os
import click
from dotenv import load_dotenv, find_dotenv
from webinar_processor.llm import LLMConfig
from webinar_processor.utils.openai import create_summary_with_context
from webinar_processor.utils.package import get_config_path

_ = load_dotenv(find_dotenv())

@click.command()
@click.argument('text_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('topics_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.option('--language', default="ru")
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def quiz(text_file: click.File, topics_file: click.File, language: str, output_file: str):
    """
    Create a quiz based on the transcript and topics.
    """
    model = LLMConfig.get_model('quiz')

    quiz_prompt_path = get_config_path("quiz-prompt.txt")
    try:
        with open(quiz_prompt_path, "r", encoding="utf-8") as pf:
            quiz_prompt = pf.read()
    except FileNotFoundError:
        click.echo(click.style(f'Error: Quiz prompt file not found at {quiz_prompt_path}', fg='red'))
        raise click.Abort

    text = text_file.read()
    topics = topics_file.read()

    try:
        quiz_md = create_summary_with_context(text, topics, language, model, quiz_prompt)
    except Exception as e:
        click.echo(click.style(f'Error generating quiz: {e}', fg='red'))
        raise click.Abort

    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as of:
                of.write(quiz_md)
        except IOError as e:
            click.echo(click.style(f'Error writing output file: {e}', fg='red'))
            raise click.Abort
    else:
        click.echo(quiz_md)
