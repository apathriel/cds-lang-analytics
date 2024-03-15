#!/usr/bin/bash

python -m venv a03_env

source ./a03_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

deactivate