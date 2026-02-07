"""Base command class with common functionality for CLI commands."""

import os
from typing import Optional
import click


class BaseCommand:
    """Base class providing common utilities for CLI commands."""

    @staticmethod
    def handle_llm_error(error: Exception, context: str = "operation") -> None:
        """
        Standard LLM error handling with user-friendly message.

        Args:
            error: The exception that occurred
            context: Description of what operation failed

        Raises:
            click.Abort: Always raises to abort command execution
        """
        from webinar_processor.llm import LLMError

        if isinstance(error, LLMError):
            click.echo(click.style(f'Error during {context}: {error}', fg='red'))
            raise click.Abort()
        raise error

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def validate_env_var(var_name: str, required: bool = True) -> Optional[str]:
        """
        Validate environment variable exists.

        Args:
            var_name: Name of the environment variable
            required: If True, aborts when variable is not set

        Returns:
            The environment variable value or None if not set and not required

        Raises:
            click.Abort: If variable is required but not set
        """
        value = os.getenv(var_name)
        if required and value is None:
            click.echo(click.style(f'Error: {var_name} is not set', fg='red'))
            raise click.Abort()
        return value

    @staticmethod
    def load_json_file(file_path: str) -> dict:
        """
        Load JSON file with error handling.

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed JSON data

        Raises:
            click.Abort: If file cannot be read or parsed
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                import json
                return json.load(f)
        except FileNotFoundError:
            click.echo(click.style(f'Error: File not found at {file_path}', fg='red'))
            raise click.Abort()
        except json.JSONDecodeError as e:
            click.echo(click.style(f'Error parsing JSON file {file_path}: {e}', fg='red'))
            raise click.Abort()
        except IOError as e:
            click.echo(click.style(f'Error reading file {file_path}: {e}', fg='red'))
            raise click.Abort()
