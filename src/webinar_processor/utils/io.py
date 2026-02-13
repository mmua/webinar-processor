"""Common I/O utilities for CLI commands."""

from typing import Optional
import click


def load_prompt_template(prompt_path: str) -> str:
    """
    Load prompt template from file with error handling.

    Args:
        prompt_path: Path to the prompt template file

    Returns:
        The content of the prompt file

    Raises:
        click.Abort: If file cannot be read
    """
    try:
        with open(prompt_path, "r", encoding="utf-8") as pf:
            return pf.read()
    except FileNotFoundError:
        click.echo(click.style(f'Error: Prompt file not found at {prompt_path}', fg='red'))
        raise click.Abort()
    except IOError as e:
        click.echo(click.style(f'Error reading prompt file: {e}', fg='red'))
        raise click.Abort()


def write_output(content: str, output_file: Optional[str] = None) -> None:
    """
    Write content to file or stdout with error handling.

    Args:
        content: Text content to write
        output_file: Optional path to output file. If None, prints to stdout.

    Raises:
        click.Abort: If file cannot be written
    """
    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            click.echo(click.style(f'Written to {output_file}', fg='green'))
        except IOError as e:
            click.echo(click.style(f'Error writing output file: {e}', fg='red'))
            click.echo(content)
            raise click.Abort()
    else:
        click.echo(content)


