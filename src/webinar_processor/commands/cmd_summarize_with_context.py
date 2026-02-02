import click
from dotenv import load_dotenv, find_dotenv
from webinar_processor.llm import LLMConfig
from webinar_processor.utils.openai import create_summary_with_context

_ = load_dotenv(find_dotenv())

@click.command()
@click.argument('text_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('context_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('prompt_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('model', nargs=1, default=None)
@click.option('--language', default="ru")
@click.option('--output-path', type=click.Path(exists=False), help='Path to an output file')
def summarize_with_context(text_file: click.File, context_file: click.File, prompt_file: click.File, model: str, language: str, output_path: str):
    """
    Generates a summary of the given text with additional context using an AI model.
    """
    text = text_file.read()
    context = context_file.read()
    prompt_template = prompt_file.read()

    model = model or LLMConfig.get_model('summarization')
    output = create_summary_with_context(text, context, language, model, prompt_template)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as of:
            of.write(output)
    else:
        click.echo(output)
