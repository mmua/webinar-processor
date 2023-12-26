import os
import requests
import click
from dotenv import load_dotenv, find_dotenv

@click.command()
@click.argument('quiz_file', type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument('slug', nargs=1)
@click.option('--endpoint', help='API endpoint to upload the webinar.', default="https://hse.snap-study.ru/webinars/import_quiz/")
def upload_quiz(quiz_file, slug, endpoint):
    """
    Upload a quiz to a specific webinar.

    This command uploads the contents of a quiz file to a webinar identified by a given slug.
    The quiz content should be in a predefined format (e.g., Markdown, JSON) as required by the endpoint.

    Args:
        quiz_file: A file object representing the quiz to be uploaded. The file should be encoded in UTF-8.
        slug: The unique identifier (slug) for the webinar to which the quiz will be associated. 
        endpoint: The API endpoint URL where the quiz will be uploaded.
                Defaults to "https://hse.snap-study.ru/webinars/import_quiz/" if not provided

    The command requires an access token for authentication, which should be set in the 
    EDU_PATH_TOKEN environment variable. It sends a POST request to the specified endpoint with the 
    quiz content and webinar slug. The command outputs success or failure messages based on the 
    response from the server.

    Examples:

        python upload-quiz --quiz_file "path/to/quiz.md" --slug "webinar-slug"

        python upload-quiz --quiz_file "path/to/quiz.md" --slug "webinar-slug" --endpoint "https://hse.snap-study.ru/webinars/import_quiz/"
    """
    _ = load_dotenv(find_dotenv(usecwd=True))
    token = os.getenv("EDU_PATH_TOKEN", None)
    if token is None:
        click.echo(click.style('Error: Access Token is not set', fg='red'))
        raise click.Abort

    if endpoint is None:
        click.echo(click.style("Error: API Endpoint is not set", fg='red'))
        raise click.Abort

    # Prepare headers
    headers = {
        'Authorization': f'Bearer {token}',
    }

    content = quiz_file.read()
    # Prepare data payload
    data = {
        'content': content,
        'slug': slug,
    }

    # Send POST request to the API endpoint
    response = requests.post(endpoint, headers=headers, data=data, timeout=(3,10))

    # Check the response
    if response.status_code == 201:
        click.echo(click.style('Quiz uploaded successfully!', fg='green'))
    else:
        click.echo(click.style(f'Failed to upload the Quiz!\n\tEndpoint: {endpoint}\n\tResponse: {response.text}', fg='red'))
