"""Configuration module."""

import os
import json
from ppp_core.exceptions import InvalidConfig


def get_config_path():
    path = os.environ.get('PPP_ML_STANDALONE_CONFIG', '')
    if not path:
        raise InvalidConfig('Could not find config file, please set '
                            'environment variable PPP_ML_STANDALONE_CONFIG.')
    else:
        try:
            with open(path) as fd:
                    data = json.load(fd)
        except ValueError as exc:
            raise InvalidConfig(*exc.args)

    return data["data"]