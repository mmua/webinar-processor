import os
import requests
import click
from dotenv import load_dotenv, find_dotenv

@click.command()
@click.argument('quiz_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('slug', nargs=1)
@click.option('--endpoint', help='API endpoint to upload the quiz.', default=None)
def upload_quiz(quiz_file, slug, endpoint):
    """Upload a quiz to a specific webinar.

    Sends quiz markdown content to the learning platform API for the
    webinar identified by SLUG. Requires EDU_PATH_TOKEN in env.
    """
    _ = load_dotenv(find_dotenv(usecwd=True))
    token = os.getenv("EDU_PATH_TOKEN", None)
    if token is None:
        click.echo(click.style('Error: EDU_PATH_TOKEN is not set', fg='red'))
        raise click.Abort()

    if endpoint is None:
        endpoint = os.getenv("EDU_PATH_QUIZ_ENDPOINT", None)

    if endpoint is None:
        click.echo(click.style('Error: No endpoint provided. Set --endpoint or EDU_PATH_QUIZ_ENDPOINT.', fg='red'))
        raise click.Abort()

    headers = {
        'Authorization': f'Bearer {token}',
    }

    content = quiz_file.read()
    data = {
        'content': content,
        'slug': slug,
    }

    response = requests.post(endpoint, headers=headers, data=data, timeout=(3, 10))

    if response.status_code == 201:
        click.echo(click.style('Quiz uploaded successfully!', fg='green'))
    else:
        click.echo(click.style(f'Failed to upload the Quiz!\n\tEndpoint: {endpoint}\n\tResponse: {response.text}', fg='red'))
