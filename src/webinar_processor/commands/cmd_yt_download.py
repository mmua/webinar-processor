"""Video downloader by URL"""

import os
import sys
import click
import requests

from pytube import YouTube


@click.command()
@click.argument('url', nargs=1)
@click.option('--output-dir', '-o', default=None, help='Output directory for downloaded video')
def download(url: str, output_dir: str):
    """
    Downloads YouTube video to specified directory
    """
    try:
        yt = YouTube(url)
    except Exception:
        click.echo("Invalid YouTube URL", err=True)
        sys.exit(1)
        
    video_path = yt.streams \
        .filter(progressive=True, file_extension='mp4') \
        .order_by('resolution') \
        .desc() \
        .first() \
        .download(output_dir)
    
    # download poster
    posters_path = os.path.join(output_dir, "posters") if output_dir else "posters"
    if not os.path.exists(posters_path):
        os.makedirs(posters_path)

    file_name = "poster.jpg"
    file_path = os.path.join(posters_path, file_name)
    response = requests.get(yt.thumbnail_url)

    with open(file_path, 'wb') as file:
        file.write(response.content)

    click.echo(video_path)