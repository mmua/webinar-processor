import os
import requests
import click
from dotenv import load_dotenv, find_dotenv

@click.command()
@click.option('--title', prompt='Title of the webinar', help='Title of the webinar.')
@click.option('--slug', prompt='Slug', help='Unique slug for the webinar.')
@click.option('--video_file', prompt='Path to the video file', help='Path to the video file.', type=click.Path(exists=True))
@click.option('--poster_file', help='Path to the poster file.', type=click.Path(exists=False), default=None)
@click.option('--transcript_file', help='Path to the transcript file.', type=click.Path(exists=False), default=None)
@click.option('--endpoint', help='API endpoint to upload the webinar.', default=None)
def upload_webinar(title, slug, video_file, poster_file, transcript_file, endpoint):
    """Upload a Webinar to the specified API endpoint."""

    _ = load_dotenv(find_dotenv(usecwd=True))
    token = os.getenv("EDU_PATH_TOKEN", None)
    if token is None:
        click.echo(click.style(f'Error: Access Token is not set', fg='red'))
        raise click.Abort

    if endpoint is None:
        endpoint = os.getenv("EDU_PATH_API_ENDPOINT", None)

    if endpoint is None:
        click.echo(click.style(f'Error: API Endpoint is not set', fg='red'))
        raise click.Abort

    if poster_file is None:
        video_dir = os.path.dirname(video_file)
        poster_file = os.path.join(video_dir, "posters", "poster.jpg")

    if transcript_file is None:
        transcript_dir = os.path.dirname(video_file)
        transcript_file = os.path.join(transcript_dir, "transcript.json")

    # Prepare headers
    headers = {
        'Authorization': f'Bearer {token}',
    }

    # Prepare data payload
    data = {
        'title': title,
        'summary': "",
        'long_summary': "",
        'slug': slug,
    }

    # Prepare files payload
    files = {
        'video_file': (os.path.basename(video_file), open(video_file, 'rb')),
        'poster_file': (os.path.basename(poster_file), open(poster_file, 'rb')),
        'transcript_file': (os.path.basename(transcript_file), open(transcript_file, 'rb')),
    }

    # Send POST request to the API endpoint
    response = requests.post(endpoint, headers=headers, files=files, data=data)

    # Check the response
    if response.status_code == 201:
        click.echo(click.style('Webinar successfully uploaded!', fg='green'))
    else:
        click.echo(click.style(f'Failed to upload the Webinar! Response: {response.text}', fg='red'))

if __name__ == '__main__':
    upload_webinar()
