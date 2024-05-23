import os
import click
import openai
from dotenv import load_dotenv, find_dotenv
from webinar_processor.utils.openai import text_transform, create_summary_with_context
from webinar_processor.utils.package import get_config_path


_ = load_dotenv(find_dotenv())


@click.command()
@click.argument('text_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.option('--language', default="ru")
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def topics(text_file: click.File, language: str, output_file: str):
    """
    Create a story
    """
    intermediate_model = "gpt-3.5-turbo-16k"
    final_model = "gpt-4o"

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        click.echo(click.style('Error: OpenAI keys is not set', fg='red'))
        raise click.Abort

    text = text_file.read()

    intermediate_prompt_path = get_config_path("intermediate-topics-prompt.txt")
    final_prompt_path = get_config_path("final-topics-prompt-onepass.txt")


    with open(intermediate_prompt_path, "r", encoding="utf-8") as pf:
        intermediate_prompt_template = pf.read()
    intermediate_topics = text_transform(text, language, intermediate_model, intermediate_prompt_template)

    with open(final_prompt_path, "r", encoding="utf-8") as pf:
        final_prompt_template = pf.read()
    final_topics = create_summary_with_context(text, intermediate_topics, language, final_model, final_prompt_template)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as of:
            of.write(final_topics)
    else:
        click.echo(final_topics)
