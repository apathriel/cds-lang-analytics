#!/usr/bin/bash

python -m pip install --user virtualenv

python -m virtualenv a03_env

source ./a03_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

deactivate