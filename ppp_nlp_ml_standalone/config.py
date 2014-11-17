"""Configuration module."""

import ppp_nlp_ml_standalone
import os

from os.path import join


def get_data(filename):
    package_dir = ppp_nlp_ml_standalone.__path__[0]
    dir_name = join(os.path.dirname(package_dir), 'data')
    fullname = join(dir_name, filename)
    return fullname