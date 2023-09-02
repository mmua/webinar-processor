"""Video downloader by URL"""

import sys
import click
from pytube import YouTube


@click.command()
@click.argument('url', nargs=1)
@click.argument('path', default=None)
def yt_download(url: str, path: str):
    """
    Downloads YouTube streams to specified directory
    """
    yt = YouTube(url)
    for stream in yt.streams:
        stream.download(path)


if __name__ == "__main__":
    yt_download(sys.argv[1], sys.argv[2])
