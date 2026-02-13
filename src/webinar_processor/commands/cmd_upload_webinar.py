import os
import requests
import click
from dotenv import load_dotenv, find_dotenv

@click.command()
@click.option('--title', prompt='Title of the webinar', help='Title of the webinar.')
@click.option('--slug', prompt='Slug', help='Unique slug for the webinar.')
@click.option('--poster_file', help='Path to the poster file.', type=click.Path(exists=False), default=None)
@click.option('--transcript_file', help='Path to the transcript file.', type=click.Path(exists=False), default=None)
@click.option('--endpoint', help='API endpoint to upload the webinar.', default=None)
@click.argument('video_file', type=click.Path(exists=True))
def upload_webinar(video_file, title, slug, poster_file, transcript_file, endpoint):
    """Upload a Webinar to the specified API endpoint."""

    _ = load_dotenv(find_dotenv(usecwd=True))
    token = os.getenv("EDU_PATH_TOKEN", None)
    if token is None:
        click.echo(click.style('Error: EDU_PATH_TOKEN is not set', fg='red'))
        raise click.Abort()

    if endpoint is None:
        endpoint = os.getenv("EDU_PATH_API_ENDPOINT", None)

    if endpoint is None:
        click.echo(click.style('Error: No endpoint. Set --endpoint or EDU_PATH_API_ENDPOINT.', fg='red'))
        raise click.Abort()

    video_dir = os.path.dirname(video_file)
    if poster_file is None:
        poster_file = os.path.join(video_dir, "posters", "poster.jpg")

    if transcript_file is None:
        transcript_file = os.path.join(video_dir, "transcript.json")

    # Prepare headers
    headers = {
        'Authorization': f'Bearer {token}',
    }

    summary = ""
    summary_path = os.path.join(video_dir, "summary.txt")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = f.read()

    long_summary = ""
    story_path = os.path.join(video_dir, "story.txt")
    if os.path.exists(story_path):
        with open(story_path, "r", encoding="utf-8") as f:
            long_summary = f.read()

    # Prepare data payload
    data = {
        'title': title,
        'summary': summary,
        'long_summary': long_summary,
        'slug': slug,
    }

    # Prepare and send with proper file handle cleanup
    with open(video_file, 'rb') as vf, \
         open(poster_file, 'rb') as pf, \
         open(transcript_file, 'rb') as tf:
        files = {
            'video_file': (os.path.basename(video_file), vf),
            'poster_file': (os.path.basename(poster_file), pf),
            'transcript_file': (os.path.basename(transcript_file), tf),
        }

        response = requests.post(endpoint, headers=headers, files=files, data=data)

    if response.status_code == 201:
        click.echo(click.style('Webinar successfully uploaded!', fg='green'))
    else:
        click.echo(click.style(f'Failed to upload the Webinar! Response: {response.text}', fg='red'))