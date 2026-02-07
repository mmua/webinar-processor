# -*- coding: utf-8 -*-

"""Top-level package for Webinar Processor App"""

import click
from dotenv import find_dotenv, load_dotenv

from webinar_processor import commands
from .commands import download, transcribe, diarize, poster, detect_gender, upload_webinar, summarize, storytell, raw_text, upload_quiz, quiz, tsv_to_transcript, speakers


__author__ = """Maxim Moroz"""
__email__ = 'mimoroz@edu.hse.ru'
__version__ = '0.9.0'


_ = load_dotenv(find_dotenv())


@click.group()
def cli():
    """Process Webinar Data"""


# Add commands
cli.add_command(commands.download)
cli.add_command(commands.transcribe)
cli.add_command(commands.diarize)
cli.add_command(commands.poster)
cli.add_command(commands.detect_gender)
cli.add_command(commands.upload_webinar)
cli.add_command(commands.summarize)
cli.add_command(commands.storytell)
cli.add_command(commands.raw_text)
cli.add_command(commands.upload_quiz)
cli.add_command(commands.quiz)
cli.add_command(commands.tsv_to_transcript)
cli.add_command(commands.speakers)
