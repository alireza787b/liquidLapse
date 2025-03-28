#!/bin/bash
# setup.sh - Automated setup for liquidLapse
# This script updates the system repositories, checks for and installs
# Python3, pip3, python3-venv, and Google Chrome, creates a virtual environment,
# installs project dependencies from requirements.txt, and gives you clear
# instructions on how to run the main script.

# Exit immediately if a command exits with a non-zero status.
set -e

# Define functions for colored output
info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

info "Starting setup for liquidLapse..."

# 1. Update package repositories
info "Updating package repositories..."
sudo apt update

# 2. Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    error "Python3 is not installed. Installing python3..."
    sudo apt install -y python3
else
    info "Python3 is installed: $(python3 --version)"
fi

# 3. Check if pip3 is installed
if ! command -v pip3 &> /dev/null; then
    error "pip3 is not installed. Installing python3-pip..."
    sudo apt install -y python3-pip
else
    info "pip3 is installed: $(pip3 --version)"
fi

# 4. Check if the venv module is available
if ! python3 -c "import venv" &> /dev/null; then
    error "python3-venv is not available. Installing python3-venv..."
    sudo apt install -y python3-venv
else
    info "python3-venv module is available."
fi

# 5. Check if Google Chrome is installed; install it if missing.
if ! command -v google-chrome &> /dev/null; then
    info "Google Chrome is not installed. Installing Google Chrome..."
    wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -O /tmp/chrome.deb
    sudo apt install -y /tmp/chrome.deb
    rm /tmp/chrome.deb
    success "Google Chrome installed successfully."
else
    info "Google Chrome is installed: $(google-chrome --version)"
fi

# 6. Create and activate a Python virtual environment
info "Creating Python virtual environment..."
python3 -m venv venv

info "Activating virtual environment..."
source venv/bin/activate

# 7. Upgrade pip and install project dependencies
info "Upgrading pip..."
pip install --upgrade pip

if [ -f requirements.txt ]; then
    info "Installing Python dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    error "requirements.txt not found. Please make sure it exists in the project root."
    exit 1
fi

success "Setup complete!"

echo ""
echo "================================================================"
echo "To run liquidLapse:"
echo "1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo "2. Run the main script:"
echo "     python liquidLapse.py"
echo "================================================================"
echo ""
success "All done! Enjoy using liquidLapse."
