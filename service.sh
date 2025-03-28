#!/bin/bash
# service.sh - Manage the liquidLapse service (start, stop, restart, status)
# This script will:
#  - Source the Python virtual environment
#  - Start the liquidLapse.py script as a background service
#  - Store the process ID (PID) in a file for later control
#  - Provide informative, colored output

set -e

# Get the absolute path of the current directory
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Define paths for the virtual environment, main script, and PID file
VENV_DIR="$BASE_DIR/venv"
PYTHON="$VENV_DIR/bin/python"
SCRIPT="$BASE_DIR/liquidLapse.py"
PIDFILE="$BASE_DIR/liquidLapse.pid"

# Colored output definitions
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to start the service
start_service() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if ps -p $PID > /dev/null; then
            echo -e "${GREEN}Service is already running (PID: $PID).${NC}"
            exit 0
        else
            echo "Removing stale PID file."
            rm -f "$PIDFILE"
        fi
    fi
    echo "Starting liquidLapse service..."
    cd "$BASE_DIR"
    source "$VENV_DIR/bin/activate"
    nohup $PYTHON "$SCRIPT" > liquidLapse.log 2>&1 &
    echo $! > "$PIDFILE"
    echo -e "${GREEN}Service started with PID: $(cat "$PIDFILE")${NC}"
}

# Function to stop the service
stop_service() {
    if [ ! -f "$PIDFILE" ]; then
        echo -e "${RED}Service is not running (PID file not found).${NC}"
        exit 1
    fi
    PID=$(cat "$PIDFILE")
    if ps -p $PID > /dev/null; then
        echo "Stopping liquidLapse service (PID: $PID)..."
        kill $PID
        while ps -p $PID > /dev/null; do
            sleep 1
        done
        echo "Service stopped."
        rm -f "$PIDFILE"
        echo -e "${GREEN}Service stopped successfully.${NC}"
    else
        echo -e "${RED}No process found for PID: $PID. Removing PID file.${NC}"
        rm -f "$PIDFILE"
    fi
}

# Function to check the service status
status_service() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if ps -p $PID > /dev/null; then
            echo -e "${GREEN}Service is running (PID: $PID).${NC}"
        else
            echo -e "${RED}Service is not running, but PID file exists.${NC}"
        fi
    else
        echo -e "${RED}Service is not running.${NC}"
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
