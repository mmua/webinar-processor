import click
from webinar_processor.llm import LLMError
from webinar_processor.utils.openai import text_transform as transform
from webinar_processor.commands.base_command import BaseCommand


@click.command()
@click.argument('text_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('prompt_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('model', nargs=1, default="gpt-4.1-mini")
@click.option('--language', default="ru")
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def text_transform(text_file: click.File, prompt_file: click.File, model: str, language: str, output_path: str):
    """
    Transforms text using a given AI model and a prompt template.

    This command reads from a specified text file and a prompt template file, then processes text
    using an AI model. The transformed text is either outputted to the console or written to a specified
    output file.

    Args:
        text_file: A file object representing text to be transformed. The file should be in UTF-8 encoding.
        prompt_file: A file object containing prompt template to be used in text transformation. The file
                     should be in UTF-8 encoding.
        model: The name of AI model to be used for transformation. Defaults to "gpt-4.1-mini".
        language: The language code (e.g., 'ru' for Russian) to specify the language of text and prompt.
                  Defaults to Russian ('ru').
        output_file: Optional. A file path where transformed text will be written. If not provided,
                     output is printed to the console.

    The `transform` function (not shown in this snippet) is expected to take the text, language, model, and
    prompt as inputs and return the transformed text.

    Examples:

        python text-transform --text_file "path/to/text.txt" --prompt_file "path/to/prompt.txt"

        python text-transform --text_file "path/to/text.txt" --prompt_file "path/to/prompt.txt" --output-file "path/to/output.txt"
    """
    text = text_file.read()
    prompt_template = prompt_file.read()

    try:
        output = transform(text, language, model, prompt_template)
    except LLMError as e:
        BaseCommand.handle_llm_error(e, "text transformation")

    BaseCommand.write_output(output, output_path)
