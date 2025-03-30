#!/bin/bash
# service.sh - Manage the liquidLapse service with detailed status reporting.
#
# This script will:
#   - Activate the Python virtual environment.
#   - Read the snapshot interval (check_interval) from config.yaml.
#   - Run liquidLapse.py (which should take a single snapshot and exit).
#   - Log execution details and update a status file with:
#         Last snapshot execution time,
#         Interval period,
#         and any errors.
#   - Repeat the process in an infinite loop until the service is stopped.
#
# A PID file (liquidLapseService.pid) is used to manage the background process.
#
# Usage: ./service.sh {start|stop|restart|status}

set -e

# Get the absolute path of the current directory
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Define paths for the virtual environment, main script, PID file, status file, and log file
VENV_DIR="$BASE_DIR/venv"
PYTHON="$VENV_DIR/bin/python"
SCRIPT="$BASE_DIR/liquidLapse.py"
PIDFILE="$BASE_DIR/liquidLapseService.pid"
STATUSFILE="$BASE_DIR/liquidLapseService.status"
LOGFILE="$BASE_DIR/liquidLapseService.log"

# Colored output for console messages only
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Functions for printing console messages
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

# Function that runs the periodic snapshot loop and updates the status file.
run_service() {
    while true; do
        # Get interval from config.yaml
        PERIOD=$("$PYTHON" -c "import yaml; print(yaml.safe_load(open('config.yaml'))['check_interval'])")
        CURRENT_TIME=$(date "+%Y-%m-%d %H:%M:%S")
        
        # Update status file with a clear marker (plain text)
        {
            echo "SNAPSHOT_START: $CURRENT_TIME"
            echo "INTERVAL: ${PERIOD} seconds"
        } > "$STATUSFILE"
        
        echo "[INFO] [$CURRENT_TIME] Running snapshot (interval: ${PERIOD} seconds)..."
        "$PYTHON" "$SCRIPT" >> "$LOGFILE" 2>&1
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            ERR_TIME=$(date "+%Y-%m-%d %H:%M:%S")
            echo "[ERROR] Snapshot failed with exit code $EXIT_CODE at $ERR_TIME." >> "$LOGFILE"
            echo "SNAPSHOT_ERROR: Exit code $EXIT_CODE at $ERR_TIME" >> "$STATUSFILE"
        else
            SUCCESS_TIME=$(date "+%Y-%m-%d %H:%M:%S")
            echo "[SUCCESS] Snapshot executed successfully at $SUCCESS_TIME." >> "$LOGFILE"
            echo "SNAPSHOT_SUCCESS: $SUCCESS_TIME" >> "$STATUSFILE"
        fi

        echo "[INFO] Sleeping for ${PERIOD} seconds until next execution..."
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

    # Start the service in the background. Inline the periodic loop without ANSI codes for logging.
    nohup bash -c '
        source "'"$VENV_DIR"'/bin/activate"
        run_service() {
            while true; do
                PERIOD=$('"$PYTHON"' -c "import yaml; print(yaml.safe_load(open(\"config.yaml\"))[\"check_interval\"])")
                CURRENT_TIME=$(date "+%Y-%m-%d %H:%M:%S")
                {
                    echo "SNAPSHOT_START: $CURRENT_TIME"
                    echo "INTERVAL: ${PERIOD} seconds"
                } >> "'"$STATUSFILE"'"
                echo "[INFO] [$CURRENT_TIME] Running snapshot (interval: ${PERIOD} seconds)..."
                '"$PYTHON"' "'"$SCRIPT"'" >> "'"$LOGFILE"'" 2>&1
                EXIT_CODE=$?
                if [ $EXIT_CODE -ne 0 ]; then
                    ERR_TIME=$(date "+%Y-%m-%d %H:%M:%S")
                    echo "[ERROR] Snapshot failed with exit code $EXIT_CODE at $ERR_TIME." >> "'"$LOGFILE"'"
                    echo "SNAPSHOT_ERROR: Exit code $EXIT_CODE at $ERR_TIME" >> "'"$STATUSFILE"'"
                else
                    SUCCESS_TIME=$(date "+%Y-%m-%d %H:%M:%S")
                    echo "[SUCCESS] Snapshot executed successfully at $SUCCESS_TIME." >> "'"$LOGFILE"'"
                    echo "SNAPSHOT_SUCCESS: $SUCCESS_TIME" >> "'"$STATUSFILE"'"
                fi
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

# Function to display detailed service status by reading the status file.
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
