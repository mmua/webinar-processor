import os
from pkg_resources import resource_filename

def get_config_path(config_name):
    """
    Get the absolute path to a configuration file.
    
    First tries the pkg_resources approach for installed packages,
    falls back to finding the file relative to the project root.
    
    Args:
        config_name: The name of the configuration file
        
    Returns:
        The absolute path to the configuration file
    """
    # First try the package resource approach
    pkg_path = resource_filename('webinar_processor', f'../conf/{config_name}')
    
    if os.path.exists(pkg_path):
        return pkg_path
    
    # Fallback: Look for the file in the project root directory
    # Get the directory of the current file (package.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to the src/webinar_processor directory
    package_dir = os.path.dirname(current_dir)
    # Go up to the src directory
    src_dir = os.path.dirname(package_dir)
    # Go up to the project root directory
    project_root = os.path.dirname(src_dir)
    # Path to the conf directory in the project root
    conf_path = os.path.join(project_root, 'conf', config_name)
    
    if os.path.exists(conf_path):
        return conf_path
    
    # If all else fails, try relative to current working directory
    cwd_path = os.path.join(os.getcwd(), 'conf', config_name)
    if os.path.exists(cwd_path):
        return cwd_path
        
    # Return the original path, even though it might not exist
    return pkg_path

def get_model_path(model_name):
    """
    Get the absolute path to a model file.
    
    First tries the pkg_resources approach for installed packages,
    falls back to finding the file relative to the project root.
    
    Args:
        model_name: The name of the model file
        
    Returns:
        The absolute path to the model file
    """
    # First try the package resource approach
    pkg_path = resource_filename('webinar_processor', f'../models/{model_name}')
    
    if os.path.exists(pkg_path):
        return pkg_path
    
    # Fallback: Look for the file in the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(current_dir)
    src_dir = os.path.dirname(package_dir)
    project_root = os.path.dirname(src_dir)
    models_path = os.path.join(project_root, 'models', model_name)
    
    if os.path.exists(models_path):
        return models_path
    
    # If all else fails, try relative to current working directory
    cwd_path = os.path.join(os.getcwd(), 'models', model_name)
    if os.path.exists(cwd_path):
        return cwd_path
        
    # Return the original path, even though it might not exist
    return pkg_path