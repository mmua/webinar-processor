"""Speaker management CLI group and subcommand registration."""

import click

from .crud import list_speakers, enroll, delete, merge, update, info
from .relabel import relabel
from .analyze import analyze
from .label import label
from .identify import identify
from .apply import apply


@click.group()
def speakers():
    """Manage speaker profiles and identifications."""
    pass


# Register all subcommands
speakers.add_command(list_speakers)
speakers.add_command(enroll)
speakers.add_command(delete)
speakers.add_command(merge)
speakers.add_command(update)
speakers.add_command(info)
speakers.add_command(relabel)
speakers.add_command(analyze)
speakers.add_command(label)
speakers.add_command(identify)
speakers.add_command(apply)
