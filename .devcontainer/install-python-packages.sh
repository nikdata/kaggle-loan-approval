#!/usr/bin/env bash

PYTHON_ENV=$1
SETUP_FOLDER=$2

python3 -m venv /opt/$PYTHON_ENV \
    && export PATH=/opt/$PYTHON_ENV/bin:$PATH \
    && echo "source /opt/$PYTHON_ENV/bin/activate" >>  ~/.bashrc

source /opt/$PYTHON_ENV/bin/activate

# source: https://medium.datadriveninvestor.com/boosting-docker-build-speed-slashing-python-build-times-in-half-with-uv-eb7734600cfb
# install uv first to optimize installs of other python packages
pip3 install uv

uv pip install --no-cache -r ./$SETUP_FOLDER/packages.txt