import os
import click
from webinar_processor.llm import LLMConfig, LLMError
from webinar_processor.utils.openai import create_summary_with_context
from webinar_processor.utils.package import get_config_path
from webinar_processor.commands.base_command import BaseCommand


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
    quiz_prompt = BaseCommand.load_prompt_template(quiz_prompt_path)

    text = text_file.read()
    topics = topics_file.read()

    try:
        quiz_md = create_summary_with_context(text, topics, language, model, quiz_prompt)
    except LLMError as e:
        BaseCommand.handle_llm_error(e, "quiz generation")

    BaseCommand.write_output(quiz_md, output_file)
