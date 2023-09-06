"""Video downloader by URL"""

import sys
import click
from pytube import YouTube


@click.command()
@click.argument('url', nargs=1)
@click.argument('path', default=None)
def yt_download(url: str, path: str):
    """
    Downloads YouTube video to specified directory
    """
    yt = YouTube(url)
    yt.streams \
        .filter(progressive=True, file_extension='mp4') \
        .order_by('resolution') \
        .desc() \
        .first() \
        .download(path)


if __name__ == "__main__":
    yt_download(sys.argv[1], sys.argv[2])
