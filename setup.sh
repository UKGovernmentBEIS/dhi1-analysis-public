#!/bin/bash
# Create a new virtual environment
python -m venv analysis_env

source analysis_env/bin/activate

# Install requirements from the file
pip install -r requirements.txt

python -m ipykernel install --user --name=analysis_env --display-name="Python (analysis_env)"
