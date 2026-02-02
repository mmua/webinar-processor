import os
import importlib.resources
from pathlib import Path
from typing import Optional

def get_config_path(config_name: str) -> str:
    """
    Get the absolute path to a configuration file using modern package resource methods.
    
    Args:
        config_name: The name of the configuration file
        
    Returns:
        The absolute path to the configuration file as a string
    """
    # Try to find the file in the package resources
    try:
        # This is for Python 3.9+ using the newer API
        with importlib.resources.files('webinar_processor').joinpath('resources', 'conf', config_name) as path:
            if path.exists():
                return str(path)
            
            # If still not found, return the resource path anyway (it might not exist)
            return str(path)
    except (ImportError, ModuleNotFoundError, AttributeError):
        # Fallback for older Python versions using deprecation-free API
        try:
            import importlib_resources  # type: ignore
            with importlib_resources.files('webinar_processor').joinpath('resources', 'conf', config_name) as path:
                if path.exists():
                    return str(path)
                
                # If still not found, return the resource path anyway (it might not exist)
                return str(path)
        except (ImportError, ModuleNotFoundError, AttributeError):
            # Last resort fallback to current working directory
            return str(Path(os.getcwd()) / 'conf' / config_name)


def get_model_path(model_name: str) -> str:
    """
    Get the absolute path to a model file using modern package resource methods.
    
    Args:
        model_name: The name of the model file
        
    Returns:
        The absolute path to the model file as a string
    """
    # Try to find the file in the package resources
    try:
        # This is for Python 3.9+ using the newer API
        with importlib.resources.files('webinar_processor').joinpath('resources', 'models', model_name) as path:
            if path.exists():
                return str(path)
            
            # If still not found, return the resource path anyway (it might not exist)
            return str(path)
            
    except (ImportError, ModuleNotFoundError, AttributeError):
        # Fallback for older Python versions using deprecation-free API
        try:
            import importlib_resources  # type: ignore
            with importlib_resources.files('webinar_processor').joinpath('resources', 'models', model_name) as path:
                if path.exists():
                    return str(path)
                                
                # If still not found, return the resource path anyway (it might not exist)
                return str(path)
        except (ImportError, ModuleNotFoundError, AttributeError):
            # Last resort fallback to current working directory
            return str(Path(os.getcwd()) / 'models' / model_name)