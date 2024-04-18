#!/usr/bin/bash

python -m pip install --user virtualenv

python -m virtualenv la04_env

source ./la04_env/Scripts/activate

pip install --upgrade pip
pip install -r requirements.txt

deactivate