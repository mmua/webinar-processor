from pkg_resources import resource_filename

def get_config_path(config_name):
    return resource_filename('webinar_processor', f'../conf/{config_name}')

def get_model_path(model_name):
    return resource_filename('webinar_processor', f'../models/{model_name}')