import json
import os
from typing import List
import click
import openai
from dotenv import load_dotenv, find_dotenv

from webinar_processor.utils.package import get_config_path
from webinar_processor.utils.openai import create_summary_with_context, text_transform

_ = load_dotenv(find_dotenv())

DEFAULT_SUMMARIZATION_MODEL="gpt-4o-mini"

@click.command()
@click.argument('asr_path', nargs=1)
@click.argument('topics_path', nargs=1, default='')
@click.argument('model', nargs=1, default=DEFAULT_SUMMARIZATION_MODEL)
@click.option('--language', default="ru")
@click.option('--prompt-file', type=click.Path(exists=True), help='Path to a file containing the prompt')
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def summarize(asr_path: str, topics_path: str, model: str, language: str, prompt_file: str, output_file: str):
    """
    Create transcript summary
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        click.echo(click.style('Error: OpenAI keys is not set', fg='red'))
        raise click.Abort

    with open(asr_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = data["text"]

    if not prompt_file:
        prompt_file = get_config_path("short-summary-with-context.txt")

    with open(prompt_file, "r", encoding="utf-8") as pf:
        prompt_template = pf.read()

    if not topics_path:
        asr_dir = os.path.dirname(asr_path)
        topics_path = os.path.join(asr_dir, "topics.txt")

    if not (topics_path and os.path.exists(topics_path)):
        click.echo(click.style('Topics file not found', fg='red'))
        raise click.Abort

    with open(topics_path, encoding="utf-8") as context_file:
        context = context_file.read()

    summary = create_summary_with_context(text, context, language, model, prompt_template)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as of:
            of.write(summary)
    else:
        click.echo(summary)


@click.command()
@click.argument('asr_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('model', nargs=1, default=DEFAULT_SUMMARIZATION_MODEL)
@click.option('--language', default="ru")
@click.option('--prompt-file', type=click.Path(exists=True), help='Path to a file containing the prompt')
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def storytell(asr_file: click.File, model: str, language: str, prompt_file: str, output_file: str):
    """
    Create a story
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        click.echo(click.style('Error: OpenAI keys is not set', fg='red'))
        raise click.Abort

    data = json.load(asr_file)

    if not prompt_file:
        prompt_file = get_config_path("long-story.txt")

    with open(prompt_file, "r", encoding="utf-8") as pf:
        prompt_template = pf.read()

    story = text_transform(data["text"], language, model, prompt_template)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as of:
            of.write(story)
    else:
        click.echo(story)


@click.command()
@click.argument('asr_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def raw_text(asr_file: click.File, output_file: str):
    """
    Write raw transcript text
    """

    data = json.load(asr_file)
    story = data["text"]

    if output_file:
        with open(output_file, "w", encoding="utf-8") as of:
            of.write(story)
    else:
        click.echo(story)
