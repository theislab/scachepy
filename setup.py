from setuptools import setup

import os

setup(
    name='Scachepy',
    version='0.0.2',
    description='Caching extension for Scanpy',
    url='https://github.com/michalk8/scachepy',
    license='MIT',
    packages=['scachepy'],
    install_requires=list(map(str.strip,
                              open(os.path.abspath('requirements.txt'), 'r').read().split())),

    zip_safe=False
)
