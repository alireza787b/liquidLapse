#!/bin/bash

BASE_DIR="$(cd "$(dirname "$0")"; pwd)"
VENV="$BASE_DIR/venv"
PYTHON="$VENV/bin/python"
SCRIPT="$BASE_DIR/ai_process/predict_once.py"
LOG="$BASE_DIR/prediction_service.log"
INTERVAL=300  # seconds

PIDFILE="$BASE_DIR/prediction_service.pid"

start() {
    echo "Starting prediction service..."
    if [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null; then
        echo "Service already running."
        exit 1
    fi
    (
        source "$VENV/bin/activate"
        while true; do
            $PYTHON $SCRIPT >> "$LOG" 2>&1
            sleep $INTERVAL
        done
    ) &
    echo $! > "$PIDFILE"
    echo "Service started."
}

stop() {
    echo "Stopping prediction service..."
    if [ -f "$PIDFILE" ]; then
        kill $(cat "$PIDFILE") 2>/dev/null
        rm -f "$PIDFILE"
        echo "Service stopped."
    else
        echo "Service not running."
    fi
}

status() {
    if [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null; then
        echo "Service is running (PID $(cat $PIDFILE))."
    else
        echo "Service is not running."
    fi
    echo "---- Last 10 log lines ----"
    tail -n 10 "$LOG" 2>/dev/null || echo "No log file yet."
}

health() {
    if [ ! -f "$LOG" ]; then
        echo "No log file found."
        exit 1
    fi
    last_run=$(stat -c %Y "$LOG")
    now=$(date +%s)
    diff=$((now - last_run))
    if [ $diff -lt $((INTERVAL * 2)) ]; then
        echo "Service appears healthy (log updated $diff seconds ago)."
    else
        echo "WARNING: Log not updated in $diff seconds. Service may be stalled."
    fi
}

restart() {
    stop
    sleep 2
    start
}

case "$1" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    restart) restart ;;
    health) health ;;
    *)
        echo "Usage: $0 {start|stop|status|restart|health}"
        exit 1
        ;;
esac