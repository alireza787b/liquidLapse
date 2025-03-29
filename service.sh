#!/bin/bash
# service.sh - Manage the liquidLapse service with detailed status reporting.
#
# This script will:
#   - Activate the Python virtual environment.
#   - Read the snapshot interval (check_interval) from config.yaml.
#   - Run liquidLapse.py (which should take a single snapshot and exit).
#   - Log execution details and update a status file with:
#         Last execution time,
#         Next scheduled execution time,
#         Interval period,
#         and any error if occurred.
#   - Repeat the process in an infinite loop until the service is stopped.
#
# A PID file (liquidLapseService.pid) is used to manage the background process.
#
# Usage: ./service.sh {start|stop|restart|status}

set -e

# Get the absolute path of the current directory
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Define paths for the virtual environment, main script, PID file, and status file
VENV_DIR="$BASE_DIR/venv"
PYTHON="$VENV_DIR/bin/python"
SCRIPT="$BASE_DIR/liquidLapse.py"
PIDFILE="$BASE_DIR/liquidLapseService.pid"
STATUSFILE="$BASE_DIR/liquidLapseService.status"
LOGFILE="$BASE_DIR/liquidLapseService.log"

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

# Function that runs the periodic snapshot loop and updates status.
run_service() {
    # Infinite loop to take snapshots periodically
    while true; do
        # Get interval from config.yaml
        PERIOD=$(get_interval)
        CURRENT_TIME=$(date "+%Y-%m-%d %H:%M:%S")
        # Calculate next execution time using GNU date (-d "$PERIOD seconds")
        NEXT_TIME=$(date -d "$PERIOD seconds" "+%Y-%m-%d %H:%M:%S")
        
        # Update status file
        echo "Last Execution: $CURRENT_TIME" > "$STATUSFILE"
        echo "Next Execution: $NEXT_TIME" >> "$STATUSFILE"
        echo "Interval: ${PERIOD} seconds" >> "$STATUSFILE"
        
        info "[$CURRENT_TIME] Running snapshot (interval: ${PERIOD} seconds)..."
        
        # Run the snapshot script and append output to log file
        $PYTHON "$SCRIPT" >> "$LOGFILE" 2>&1
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            error "Snapshot failed with exit code $EXIT_CODE at $(date "+%Y-%m-%d %H:%M:%S")."
            echo "Last Execution Error: Exit code $EXIT_CODE at $(date "+%Y-%m-%d %H:%M:%S")" >> "$STATUSFILE"
        else
            success "Snapshot executed successfully at $(date "+%Y-%m-%d %H:%M:%S")."
            echo "Last Execution Successful at $(date "+%Y-%m-%d %H:%M:%S")" >> "$STATUSFILE"
        fi

        info "Sleeping for ${PERIOD} seconds until next execution..."
        sleep "${PERIOD}"
    done
}

# Function to start the service as a background process
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

    # Start the service in the background. We inline the loop.
    nohup bash -c '
        source "'"$VENV_DIR"'/bin/activate"
        run_service() {
            while true; do
                PERIOD=$('"$PYTHON"' -c "import yaml; print(yaml.safe_load(open(\"config.yaml\"))[\"check_interval\"])")
                CURRENT_TIME=$(date "+%Y-%m-%d %H:%M:%S")
                NEXT_TIME=$(date -d "$PERIOD seconds" "+%Y-%m-%d %H:%M:%S")
                echo "Last Execution: $CURRENT_TIME" > "'"$STATUSFILE"'"
                echo "Next Execution: $NEXT_TIME" >> "'"$STATUSFILE"'"
                echo "Interval: ${PERIOD} seconds" >> "'"$STATUSFILE"'"
                echo -e "\033[0;34m[INFO]\033[0m [$CURRENT_TIME] Running snapshot (interval: ${PERIOD} seconds)..."
                '"$PYTHON"' liquidLapse.py >> "'"$LOGFILE"'" 2>&1
                EXIT_CODE=$?
                if [ $EXIT_CODE -ne 0 ]; then
                    echo -e "\033[0;31m[ERROR]\033[0m Snapshot failed with exit code $EXIT_CODE at $(date "+%Y-%m-%d %H:%M:%S")." >> "'"$LOGFILE"'"
                    echo "Last Execution Error: Exit code $EXIT_CODE at $(date "+%Y-%m-%d %H:%M:%S")" >> "'"$STATUSFILE"'"
                else
                    echo -e "\033[0;32m[SUCCESS]\033[0m Snapshot executed successfully at $(date "+%Y-%m-%d %H:%M:%S")." >> "'"$LOGFILE"'"
                    echo "Last Execution Successful at $(date "+%Y-%m-%d %H:%M:%S")" >> "'"$STATUSFILE"'"
                fi
                echo -e "\033[0;34m[INFO]\033[0m Sleeping for ${PERIOD} seconds until next execution..."
                sleep ${PERIOD}
            done
        }
        run_service
    ' > "$LOGFILE" 2>&1 &

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

# Function to display detailed service status
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

    # Display status details if available
    if [ -f "$STATUSFILE" ]; then
        echo -e "${BLUE}--- Service Status ---${NC}"
        cat "$STATUSFILE"
        echo -e "${BLUE}----------------------${NC}"
    else
        info "No status file available."
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
