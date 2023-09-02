# -*- coding: utf-8 -*-

"""Top-level package for Webinar Processor App"""

import click

from webinar_processor import commands


__author__ = """Maxim Moroz"""
__email__ = 'mimoroz@edu.hse.ru'
__version__ = '0.1.0'


@click.group()
def cli():
    """Download Youtube Video"""


# Add commands
cli.add_command(commands.yt_download)
