# -*- coding: utf-8 -*-

"""Top-level package for Webinar Processor App"""

import click
from dotenv import find_dotenv, load_dotenv

from webinar_processor import commands


__author__ = """Maxim Moroz"""
__email__ = 'mimoroz@edu.hse.ru'
__version__ = '0.2.0'


_ = load_dotenv(find_dotenv())


@click.group()
def cli():
    """Process Webinar Data"""


# Add commands
cli.add_command(commands.yt_download)
cli.add_command(commands.transcribe)
cli.add_command(commands.diarize)
cli.add_command(commands.poster)
cli.add_command(commands.detect_gender)
cli.add_command(commands.upload_webinar)
cli.add_command(commands.summarize)
cli.add_command(commands.storytell)
cli.add_command(commands.raw_text)
cli.add_command(commands.topics)
cli.add_command(commands.text_transform)
cli.add_command(commands.summarize_with_context)
cli.add_command(commands.upload_quiz)
cli.add_command(commands.quiz)
