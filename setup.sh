#!/bin/bash
# setup.sh - Setup environment for liquidLapse

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. To run the script, activate the virtual environment with:"
echo "source venv/bin/activate"
echo "then run: python liquidLapse.py"