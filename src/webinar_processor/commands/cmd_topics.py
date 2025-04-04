import os
import click
from dotenv import load_dotenv, find_dotenv
from webinar_processor.utils.openai import create_summary, create_summary_with_context, DEFAULT_LONG_CONTEXT_MODEL
from webinar_processor.utils.package import get_config_path


_ = load_dotenv(find_dotenv())


@click.command()
@click.argument('text_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('intermediate_model', nargs=1, default="gpt-4o-mini")
@click.argument('final_model', nargs=1, default=DEFAULT_LONG_CONTEXT_MODEL)
@click.option('--language', default="ru")
@click.option('--output-file', type=click.Path(exists=False), help='Path to an output file')
def topics(text_file: click.File, intermediate_model: str, final_model: str, language: str, output_file: str):
    """
    Extract a topics list for a text transcript.
    """
    # Check for API key - our updated code uses the environment variable directly
    if os.getenv("OPENAI_API_KEY") is None:
        click.echo(click.style('Error: OPENAI_API_KEY environment variable is not set', fg='red'))
        raise click.Abort

    text = text_file.read()

    try:
        # Get the prompt file paths
        intermediate_prompt_path = get_config_path("intermediate-topics-prompt.txt")
        final_prompt_path = get_config_path("final-topics-prompt-onepass.txt")

        # Read the prompt templates
        with open(intermediate_prompt_path, "r", encoding="utf-8") as pf:
            intermediate_prompt_template = pf.read()
        
        # Generate intermediate topics
        click.echo("Extracting topics from transcript...")
        intermediate_topics = create_summary(text, language, intermediate_model, intermediate_prompt_template)

        # Read the final prompt template
        with open(final_prompt_path, "r", encoding="utf-8") as pf:
            final_prompt_template = pf.read()
        
        # Generate final topics
        click.echo("Refining topics...")
        final_topics = create_summary_with_context(text, intermediate_topics, language, final_model, final_prompt_template)

        # Output the results
        if output_file:
            with open(output_file, "w", encoding="utf-8") as of:
                of.write(final_topics)
            with open(output_file + ".im", "w", encoding="utf-8") as of:
                of.write(intermediate_topics)
            click.echo(f"Topics written to {output_file}")
        else:
            click.echo(final_topics)
            
    except Exception as e:
        click.echo(click.style(f'Error: {str(e)}', fg='red'))
        raise click.Abort
