"""Configuration module."""

import os
from ppp_libmodule.config import Config as BaseConfig

class Config(BaseConfig):
    config_path_variable = 'PPP_ML_STANDALONE_CONFIG'
    def parse_config(self, data):
        self.data_dir = data['data_dir']

def get_data(filename):
    return os.path.join(Config().data_dir, filename)
