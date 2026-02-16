# -*- coding: utf-8 -*-

"""Top-level package for Webinar Processor App"""

import logging

import click
from dotenv import find_dotenv, load_dotenv

from webinar_processor import commands


__author__ = """Maxim Moroz"""
__email__ = 'mmua@users.noreply.github.com'
__version__ = '0.13.0'


_ = load_dotenv(find_dotenv())

# Show service-layer log messages (INFO+) in the terminal.
# format='%(message)s' gives clean output without logger-name prefixes.
logging.basicConfig(level=logging.INFO, format='%(message)s')


@click.group()
def cli():
    """Process Webinar Data"""


# Add commands
cli.add_command(commands.download)
cli.add_command(commands.transcribe)
cli.add_command(commands.diarize)
cli.add_command(commands.upload_webinar)
cli.add_command(commands.summarize)
cli.add_command(commands.storytell)
cli.add_command(commands.raw_text)
cli.add_command(commands.upload_quiz)
cli.add_command(commands.quiz)
cli.add_command(commands.tsv_to_transcript)
cli.add_command(commands.speakers)
