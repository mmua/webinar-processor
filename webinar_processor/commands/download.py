"""Video downloader by URL"""

import os
import sys
import click
import requests

from pytube import YouTube


@click.command()
@click.argument('url', nargs=1)
@click.argument('path', default=None)
def yt_download(url: str, path: str):
    """
    Downloads YouTube video to specified directory
    """
    yt = YouTube(url)
    video_path = yt.streams \
        .filter(progressive=True, file_extension='mp4') \
        .order_by('resolution') \
        .desc() \
        .first() \
        .download(path)
    
    # download poster
    posters_path = os.path.join(path, "posters")
    if not os.path.exists(posters_path):
        os.makedirs(posters_path)

    file_name = "poster.jpg"
    file_path = os.path.join(posters_path, file_name)
    response = requests.get(yt.thumbnail_url)

    with open(file_path, 'wb') as file:
        file.write(response.content)

    click.echo(video_path)


if __name__ == "__main__":
    yt_download(sys.argv[1], sys.argv[2])
