import click
from dotenv import load_dotenv, find_dotenv
from webinar_processor.utils.openai import create_summary_with_context


@click.command()
@click.argument('text_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('context_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('prompt_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('model', nargs=1, default="gpt-4o")
@click.option('--language', default="ru")
@click.option('--output-path', type=click.Path(exists=False), help='Path to an output file')
def summarize_with_context(text_file: click.File, context_file: click.File, prompt_file: click.File, model: str, language: str, output_path: str):
    """
    Generates a summary of the given text with additional context using an AI model.

    This command reads from a text file, a context file, and a prompt template file. It processes the text 
    by incorporating the additional context and using a specified AI model to generate a summary. The 
    summary can be outputted to a console or written to a specified output file.

    Args:
        text_file: A file object representing the main text to be summarized. The file should be in UTF-8 encoding.
        context_file: A file object containing additional context information that aids in the summarization process.
        prompt_file: A file object containing the prompt template to guide the AI model in summarization.
        model: The name of the AI model to be used for the summarization. Defaults to "gpt-4-1106-preview".
        language: The language code (e.g., 'ru' for Russian) to specify the language of the text. Defaults to 'ru'.
        output_path: Optional. A file path where the generated summary will be written. 
            If not provided, the output is printed to the console.

    Examples:

        python summarize-with-context text.txt context.txt prompt.txt

        python summarize-with-context text.txt context.txt prompt.txt --output-path output.txt
    """
    _ = load_dotenv(find_dotenv())

    text = text_file.read()
    context = context_file.read()
    prompt_template = prompt_file.read()

    output = create_summary_with_context(text, context, language, model, prompt_template)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as of:
            of.write(output)
    else:
        click.echo(output)
