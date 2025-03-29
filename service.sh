#!/bin/bash
# service.sh - Manage the liquidLapse service (start, stop, restart, status)
# This script will:
#  - Activate the Python virtual environment.
#  - Read the snapshot interval from config.yaml.
#  - Run liquidLapse.py to capture a snapshot.
#  - Sleep for the configured period.
#  - Repeat the process in an infinite loop.
#
# A PID file (liquidLapseService.pid) is used to manage the background process.
#
# Usage: ./service.sh {start|stop|restart|status}

set -e

# Get the absolute path of the current directory
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Define paths for the virtual environment, main script, and PID file
VENV_DIR="$BASE_DIR/venv"
PYTHON="$VENV_DIR/bin/python"
SCRIPT="$BASE_DIR/liquidLapse.py"
PIDFILE="$BASE_DIR/liquidLapseService.pid"

# Colored output definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Functions for printing messages
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}
success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}
error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to read check_interval from config.yaml using the virtualenv's Python
get_interval() {
    $PYTHON -c "import yaml; print(yaml.safe_load(open('config.yaml'))['check_interval'])"
}

# Function to start the service (periodic snapshot loop)
start_service() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if ps -p "$PID" > /dev/null; then
            success "Service is already running (PID: $PID)."
            exit 0
        else
            info "Found stale PID file. Removing it."
            rm -f "$PIDFILE"
        fi
    fi

    info "Activating virtual environment and starting service..."
    cd "$BASE_DIR"
    if [ ! -d "$VENV_DIR" ]; then
        error "Virtual environment not found. Please run setup.sh first."
        exit 1
    fi

    # Inline the periodic loop in the nohup command.
    nohup bash -c '
        source "'"$VENV_DIR"'/bin/activate"
        while true; do
            PERIOD=$('"$PYTHON"' -c "import yaml; print(yaml.safe_load(open(\"config.yaml\"))[\"check_interval\"])")
            echo -e "\033[0;34m[INFO]\033[0m Running snapshot (interval set to ${PERIOD} seconds)..."
            '"$PYTHON"' liquidLapse.py
            echo -e "\033[0;34m[INFO]\033[0m Snapshot complete. Sleeping for ${PERIOD} seconds..."
            sleep ${PERIOD}
        done
    ' > liquidLapseService.log 2>&1 &

    echo $! > "$PIDFILE"
    success "Service started with PID: $(cat "$PIDFILE")."
}

# Function to stop the service
stop_service() {
    if [ ! -f "$PIDFILE" ]; then
        error "Service is not running (PID file not found)."
        exit 1
    fi
    PID=$(cat "$PIDFILE")
    if ps -p "$PID" > /dev/null; then
        info "Stopping service (PID: $PID)..."
        kill "$PID"
        while ps -p "$PID" > /dev/null; do
            sleep 1
        done
        rm -f "$PIDFILE"
        success "Service stopped."
    else
        error "No process found for PID: $PID. Removing PID file."
        rm -f "$PIDFILE"
    fi
}

# Function to display service status
status_service() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if ps -p "$PID" > /dev/null; then
            success "Service is running (PID: $PID)."
        else
            error "Service is not running, but PID file exists."
        fi
    else
        error "Service is not running."
    fi
}

# Function to restart the service
restart_service() {
    stop_service
    start_service
}

# Main control logic based on command-line argument
case "$1" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        status_service
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
