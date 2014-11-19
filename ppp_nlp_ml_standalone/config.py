"""Configuration module."""

from ppp_libmodule.config import Config as BaseConfig

class Config(BaseConfig):
    config_path_variable = 'PPP_NLP_STANDALONE_CONFIG'
    def parse_config(self, data):
        self.data_dir = data['data_dir']
