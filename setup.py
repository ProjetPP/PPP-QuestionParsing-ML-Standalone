#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='ppp_nlp_ml_standalone',
    version='0.1.1',
    description='Compute triplets from a question, with an ML approach',
    url='https://github.com/ProjetPP',
    author='Quentin Cormier',
    author_email='quentin.cormier@ens-lyon.fr',
    license='MIT',
    classifiers=[
        'Environment :: No Input/Output (Daemon)',
        'Development Status :: 1 - Planning',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
        'Topic :: Software Development :: Libraries',
    ],
    install_requires=[
        'ppp_datamodel>=0.5',
        'ppp_core>=0.5',
    ],
    packages=[
        'ppp_nlp_ml_standalone',
    ],
)
