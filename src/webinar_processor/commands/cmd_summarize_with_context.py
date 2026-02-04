import click
from webinar_processor.llm import LLMConfig, LLMError
from webinar_processor.utils.openai import create_summary_with_context
from webinar_processor.commands.base_command import BaseCommand


@click.command()
@click.argument('text_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('context_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('prompt_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('model', nargs=1, default=None)
@click.option('--language', default="ru")
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def summarize_with_context(text_file: click.File, context_file: click.File, prompt_file: click.File, model: str, language: str, output_file: str):
    """
    Generates a summary of given text with additional context using an AI model.
    """
    text = text_file.read()
    context = context_file.read()
    prompt_template = prompt_file.read()

    model = model or LLMConfig.get_model('summarization')

    try:
        output = create_summary_with_context(text, context, language, model, prompt_template)
    except LLMError as e:
        BaseCommand.handle_llm_error(e, "summary generation")

    BaseCommand.write_output(output, output_path)
