import os
from pathlib import Path

def ensure_dir_exists(directory: str | Path) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory to ensure exists
    """
    if isinstance(directory, str):
        directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
