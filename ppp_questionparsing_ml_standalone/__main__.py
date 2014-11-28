#!/usr/bin/env python3

import sys

from .learn import learn
from .dataset import create_dataset

def bootstrap():
    create_dataset()
    learn()


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'bootstrap':
        bootstrap()
    else:
        print('Syntax: python3 -m ppp_questionparsing_ml_standalone bootstrap.')
