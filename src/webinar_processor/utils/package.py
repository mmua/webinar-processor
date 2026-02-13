import importlib.resources


def get_config_path(config_name: str) -> str:
    """
    Get the absolute path to a configuration file in package resources.

    Args:
        config_name: The name of the configuration file

    Returns:
        The absolute path to the configuration file as a string
    """
    with importlib.resources.files('webinar_processor').joinpath('resources', 'conf', config_name) as path:
        return str(path)
