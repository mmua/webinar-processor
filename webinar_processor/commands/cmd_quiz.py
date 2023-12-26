import os
import click
import openai
from dotenv import load_dotenv, find_dotenv
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
    Create a quiz
    """
    final_model = "gpt-4-1106-preview"

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        click.echo(click.style('Error: OpenAI keys is not set', fg='red'))
        raise click.Abort

    quiz_prompt_path = get_config_path("quiz-prompt.txt")
    with open(quiz_prompt_path, "r", encoding="utf-8") as pf:
        quiz_prompt = pf.read()

    text = text_file.read()
    topics = topics_file.read()

    quiz_md = create_summary_with_context(text, topics, language, final_model, quiz_prompt)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as of:
            of.write(quiz_md)
    else:
        click.echo(quiz_md)
