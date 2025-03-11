#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"

# cd user_study
# python3.9 -m venv .env
# source .env/bin/activate
# pip install -U pip
# pip install -U setuptools wheel twine
# pip install -r requirements.txt
# cd ..

cd gpai_study
python3.9 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -U setuptools wheel twine
pip install -r requirements.txt
cd ..
