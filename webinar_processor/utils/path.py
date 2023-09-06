import os

def get_wav_filename(webinar_path: str, directory: str) -> str:
    """
    Given a path to a webinar, get the corresponding .wav filename in a given directory.
    """
    base_name = os.path.basename(webinar_path)  # Get the filename from the path
    name_without_extension = os.path.splitext(base_name)[0]  # Strip the file extension
    return os.path.join(directory, name_without_extension + ".wav")  # Join with the directory and add .wav extension

def get_json_filename(webinar_path: str, directory: str) -> str:
    """
    Given a path to a webinar, get the corresponding .wav filename in a given directory.
    """
    base_name = os.path.basename(webinar_path)  # Get the filename from the path
    name_without_extension = os.path.splitext(base_name)[0]  # Strip the file extension
    return os.path.join(directory, name_without_extension + ".json")  # Join with the directory and add .wav extension
