import json
import click
from webinar_processor.utils.io import write_output


@click.command()
@click.argument('asr_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.option('--output-file', type=click.Path(exists=False))
def raw_text(asr_file: click.File, output_file: str):
    """Write raw transcript text."""
    data = json.load(asr_file)
    write_output(data["text"], output_file)
