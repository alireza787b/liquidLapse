#!/bin/bash

# =============================================================================
# liquidLapse Prediction Service Management Script
# 
# This script manages the continuous prediction service independently from
# the main heatmap capture service. It provides start, stop, restart, and
# status functionality with robust error handling and logging.
#
# Features:
# - Automatic virtual environment detection and activation
# - Cross-platform compatibility (Linux, macOS, WSL)
# - Robust process management with proper cleanup
# - Enhanced logging and status reporting
# - Dependency validation
# - Graceful shutdown handling
#
# Usage:
#   ./prediction_service.sh {start|stop|restart|status|logs|clean|check}
#
# Files:
#   - prediction_service.py: Main service script
#   - prediction_service.log: Service logs
#   - prediction_service.status: Service status JSON
#   - prediction_service.pid: Process ID file
# =============================================================================

set -euo pipefail  # Strict error handling

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
SERVICE_SCRIPT="$BASE_DIR/prediction_service.py"
LOG_FILE="$BASE_DIR/prediction_service.log"
STATUS_FILE="$BASE_DIR/prediction_service.status"
PID_FILE="$BASE_DIR/prediction_service.pid"
CONFIG_FILE="$BASE_DIR/config.yaml"

# Auto-detect virtual environment
VENV_PATHS=(
    "$BASE_DIR/venv"
    "$BASE_DIR/.venv"
    "$BASE_DIR/env"
    "$BASE_DIR/.env"
)

VENV_DIR=""
PYTHON_EXEC=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# Utility Functions
# =============================================================================

print_status() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} [$timestamp] $message"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} [$timestamp] $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} [$timestamp] $message"
            ;;
        "DEBUG")
            echo -e "${BLUE}[DEBUG]${NC} [$timestamp] $message"
            ;;
        "SUCCESS")
            echo -e "${CYAN}[SUCCESS]${NC} [$timestamp] $message"
            ;;
        *)
            echo "[$timestamp] $message"
            ;;
    esac
}

detect_venv() {
    print_status "DEBUG" "Detecting virtual environment..."
    
    # Check for activated virtual environment
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        VENV_DIR="$VIRTUAL_ENV"
        PYTHON_EXEC="$VIRTUAL_ENV/bin/python"
        print_status "SUCCESS" "Using activated virtual environment: $VENV_DIR"
        return 0
    fi
    
    # Search for virtual environment directories
    for venv_path in "${VENV_PATHS[@]}"; do
        if [[ -d "$venv_path" ]]; then
            local python_path="$venv_path/bin/python"
            if [[ -f "$python_path" ]]; then
                VENV_DIR="$venv_path"
                PYTHON_EXEC="$python_path"
                print_status "SUCCESS" "Found virtual environment: $VENV_DIR"
                return 0
            fi
        fi
    done
    
    # Fallback to system Python if no venv found
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_EXEC="python3"
        print_status "WARN" "No virtual environment found, using system Python: $(which python3)"
        return 0
    elif command -v python >/dev/null 2>&1; then
        PYTHON_EXEC="python"
        print_status "WARN" "No virtual environment found, using system Python: $(which python)"
        return 0
    fi
    
    print_status "ERROR" "No Python executable found!"
    return 1
}

validate_python_version() {
    local python_version
    python_version=$($PYTHON_EXEC --version 2>&1 | cut -d' ' -f2)
    local major_version=$(echo "$python_version" | cut -d'.' -f1)
    local minor_version=$(echo "$python_version" | cut -d'.' -f2)
    
    if [[ "$major_version" -lt 3 ]] || [[ "$major_version" -eq 3 && "$minor_version" -lt 7 ]]; then
        print_status "ERROR" "Python 3.7+ required, found: $python_version"
        return 1
    fi
    
    print_status "INFO" "Python version: $python_version ✓"
    return 0
}

check_dependencies() {
    print_status "INFO" "Checking Python dependencies..."
    
    local required_packages=("torch" "torchvision" "PIL" "yaml" "requests")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! $PYTHON_EXEC -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        print_status "ERROR" "Missing required packages: ${missing_packages[*]}"
        print_status "INFO" "Install with: $PYTHON_EXEC -m pip install torch torchvision pillow pyyaml requests"
        return 1
    fi
    
    print_status "SUCCESS" "All required packages are installed ✓"
    return 0
}

check_requirements() {
    local errors=0
    
    print_status "INFO" "Performing system requirements check..."
    
    # Detect and validate Python environment
    if ! detect_venv; then
        ((errors++))
    fi
    
    if [[ -n "$PYTHON_EXEC" ]]; then
        if ! validate_python_version; then
            ((errors++))
        fi
        
        if ! check_dependencies; then
            ((errors++))
        fi
    fi
    
    # Check if Python service script exists
    if [[ ! -f "$SERVICE_SCRIPT" ]]; then
        print_status "ERROR" "Service script not found: $SERVICE_SCRIPT"
        ((errors++))
    else
        print_status "SUCCESS" "Service script found ✓"
    fi
    
    # Check config file
    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_status "ERROR" "Configuration file not found: $CONFIG_FILE"
        ((errors++))
    else
        print_status "SUCCESS" "Configuration file found ✓"
        
        # Validate config structure
        if [[ -n "$PYTHON_EXEC" ]]; then
            if ! $PYTHON_EXEC -c "
import yaml
import sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    if 'prediction' not in config:
        print('ERROR: Missing prediction section in config')
        sys.exit(1)
    if not config['prediction'].get('enabled', False):
        print('WARNING: Prediction service is disabled in config')
        sys.exit(2)
    print('SUCCESS: Configuration is valid')
except Exception as e:
    print(f'ERROR: Invalid config file: {e}')
    sys.exit(1)
" 2>/dev/null; then
                local exit_code=$?
                if [[ $exit_code -eq 1 ]]; then
                    ((errors++))
                elif [[ $exit_code -eq 2 ]]; then
                    print_status "WARN" "Prediction service is disabled in configuration"
                fi
            else
                print_status "SUCCESS" "Configuration is valid ✓"
            fi
        fi
    fi
    
    # Check write permissions
    if ! touch "$LOG_FILE" 2>/dev/null; then
        print_status "ERROR" "Cannot write to log file: $LOG_FILE"
        ((errors++))
    else
        print_status "SUCCESS" "Log file writable ✓"
    fi
    
    if ! touch "$STATUS_FILE" 2>/dev/null; then
        print_status "ERROR" "Cannot write to status file: $STATUS_FILE"
        ((errors++))
    else
        print_status "SUCCESS" "Status file writable ✓"
    fi
    
    if ! touch "$PID_FILE" 2>/dev/null; then
        print_status "ERROR" "Cannot write to PID file: $PID_FILE"
        ((errors++))
    else
        print_status "SUCCESS" "PID file writable ✓"
    fi
    
    return $errors
}

get_service_pid() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE" 2>/dev/null)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        else
            # Clean up stale PID file
            rm -f "$PID_FILE" 2>/dev/null
        fi
    fi
    return 1
}

is_service_running() {
    local pid
    pid=$(get_service_pid)
    return $?
}

get_service_status() {
    if [[ -f "$STATUS_FILE" ]]; then
        if command -v jq >/dev/null 2>&1; then
            jq -r '.status // "unknown"' "$STATUS_FILE" 2>/dev/null || echo "unknown"
        else
            $PYTHON_EXEC -c "
import json
try:
    with open('$STATUS_FILE', 'r') as f:
        data = json.load(f)
    print(data.get('status', 'unknown'))
except:
    print('unknown')
" 2>/dev/null || echo "unknown"
        fi
    else
        echo "unknown"
    fi
}

wait_for_service_start() {
    local timeout=${1:-30}
    local count=0
    
    print_status "INFO" "Waiting for service to start (timeout: ${timeout}s)..."
    
    while [[ $count -lt $timeout ]]; do
        if is_service_running; then
            local status=$(get_service_status)
            if [[ "$status" == "running" ]]; then
                print_status "SUCCESS" "Service started successfully"
                return 0
            fi
        fi
        sleep 1
        ((count++))
        if [[ $((count % 5)) -eq 0 ]]; then
            print_status "DEBUG" "Still waiting... (${count}/${timeout}s)"
        fi
    done
    
    print_status "ERROR" "Service failed to start within ${timeout} seconds"
    return 1
}

wait_for_service_stop() {
    local timeout=${1:-15}
    local count=0
    
    print_status "INFO" "Waiting for service to stop (timeout: ${timeout}s)..."
    
    while [[ $count -lt $timeout ]]; do
        if ! is_service_running; then
            print_status "SUCCESS" "Service stopped successfully"
            return 0
        fi
        sleep 1
        ((count++))
        if [[ $((count % 5)) -eq 0 ]]; then
            print_status "DEBUG" "Still waiting... (${count}/${timeout}s)"
        fi
    done
    
    print_status "WARN" "Service did not stop gracefully within ${timeout} seconds"
    return 1
}

# =============================================================================
# Service Management Functions
# =============================================================================

start_service() {
    print_status "INFO" "Starting liquidLapse Prediction Service..."
    
    # Check if already running
    if is_service_running; then
        local pid=$(get_service_pid)
        print_status "WARN" "Service is already running (PID: $pid)"
        return 1
    fi
    
    # Check requirements
    if ! check_requirements; then
        print_status "ERROR" "Requirements check failed"
        return 1
    fi
    
    # Clean up any stale files
    rm -f "$PID_FILE" "$STATUS_FILE" 2>/dev/null
    
    # Start the service
    print_status "INFO" "Launching service with Python: $PYTHON_EXEC"
    
    # Use nohup for proper daemonization
    nohup $PYTHON_EXEC "$SERVICE_SCRIPT" \
        --base_dir "$BASE_DIR" \
        --config "$CONFIG_FILE" \
        --daemon \
        >> "$LOG_FILE" 2>&1 &
    
    local service_pid=$!
    
    # Give the service a moment to initialize
    sleep 2
    
    # Verify it's still running
    if ! kill -0 $service_pid 2>/dev/null; then
        print_status "ERROR" "Service failed to start"
        print_status "INFO" "Check logs: tail -f $LOG_FILE"
        return 1
    fi
    
    # Wait for proper startup
    if wait_for_service_start 30; then
        local pid=$(get_service_pid)
        print_status "SUCCESS" "Service started successfully (PID: $pid)"
        print_status "INFO" "Monitor logs: tail -f $LOG_FILE"
        return 0
    else
        print_status "ERROR" "Service startup failed or timed out"
        print_status "INFO" "Check logs: tail -f $LOG_FILE"
        return 1
    fi
}

stop_service() {
    print_status "INFO" "Stopping liquidLapse Prediction Service..."
    
    local pid
    if ! pid=$(get_service_pid); then
        print_status "WARN" "Service is not running"
        return 0
    fi
    
    print_status "INFO" "Sending SIGTERM to process $pid..."
    
    # Send SIGTERM for graceful shutdown
    if kill -TERM "$pid" 2>/dev/null; then
        if wait_for_service_stop 15; then
            rm -f "$PID_FILE" "$STATUS_FILE" 2>/dev/null
            return 0
        else
            print_status "WARN" "Graceful shutdown timed out, forcing termination..."
            if kill -KILL "$pid" 2>/dev/null; then
                sleep 2
                rm -f "$PID_FILE" "$STATUS_FILE" 2>/dev/null
                print_status "SUCCESS" "Service forcefully terminated"
                return 0
            else
                print_status "ERROR" "Failed to terminate service"
                return 1
            fi
        fi
    else
        print_status "ERROR" "Failed to send signal to process $pid"
        return 1
    fi
}

restart_service() {
    print_status "INFO" "Restarting liquidLapse Prediction Service..."
    
    if is_service_running; then
        stop_service
        sleep 2
    fi
    
    start_service
}

show_status() {
    print_status "INFO" "liquidLapse Prediction Service Status"
    echo "=" * 50
    
    local pid
    if pid=$(get_service_pid); then
        local status=$(get_service_status)
        print_status "SUCCESS" "Service is running"
        echo "  PID: $pid"
        echo "  Status: $status"
        
        # Show process info if available
        if command -v ps >/dev/null 2>&1; then
            echo "  Process Info:"
            ps -p "$pid" -o pid,ppid,cmd,start,time 2>/dev/null | tail -n +2 | sed 's/^/    /'
        fi
        
        # Show memory usage if available
        if [[ -f "/proc/$pid/status" ]]; then
            local memory=$(grep -E "VmRSS|VmSize" "/proc/$pid/status" 2>/dev/null)
            if [[ -n "$memory" ]]; then
                echo "  Memory Usage:"
                echo "$memory" | sed 's/^/    /'
            fi
        fi
    else
        print_status "WARN" "Service is not running"
    fi
    
    # Show file status
    echo ""
    echo "File Status:"
    for file in "$SERVICE_SCRIPT" "$CONFIG_FILE" "$LOG_FILE" "$STATUS_FILE" "$PID_FILE"; do
        if [[ -f "$file" ]]; then
            local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
            local modified=$(stat -f%Sm "$file" 2>/dev/null || stat -c%y "$file" 2>/dev/null || echo "unknown")
            echo "  $(basename "$file"): exists (${size} bytes, modified: $modified)"
        else
            echo "  $(basename "$file"): missing"
        fi
    done
    
    # Show recent log entries
    if [[ -f "$LOG_FILE" ]]; then
        echo ""
        echo "Recent Log Entries (last 10 lines):"
        tail -n 10 "$LOG_FILE" | sed 's/^/  /'
    fi
}

show_logs() {
    if [[ -f "$LOG_FILE" ]]; then
        print_status "INFO" "Showing prediction service logs..."
        echo "=" * 50
        if [[ "${1:-}" == "follow" ]] || [[ "${1:-}" == "-f" ]]; then
            tail -f "$LOG_FILE"
        else
            tail -n 50 "$LOG_FILE"
            echo ""
            print_status "INFO" "Use '$0 logs follow' to follow logs in real-time"
        fi
    else
        print_status "WARN" "Log file not found: $LOG_FILE"
    fi
}

clean_service() {
    print_status "INFO" "Cleaning up service files..."
    
    if is_service_running; then
        print_status "ERROR" "Cannot clean while service is running. Stop the service first."
        return 1
    fi
    
    local files_to_clean=("$PID_FILE" "$STATUS_FILE")
    local cleaned=0
    
    for file in "${files_to_clean[@]}"; do
        if [[ -f "$file" ]]; then
            rm -f "$file"
            print_status "SUCCESS" "Removed: $(basename "$file")"
            ((cleaned++))
        fi
    done
    
    if [[ $cleaned -eq 0 ]]; then
        print_status "INFO" "No files to clean"
    else
        print_status "SUCCESS" "Cleaned $cleaned files"
    fi
    
    # Optionally clean old log files
    read -p "Do you want to clean log files? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [[ -f "$LOG_FILE" ]]; then
            rm -f "$LOG_FILE"
            print_status "SUCCESS" "Removed: $(basename "$LOG_FILE")"
        fi
    fi
}

run_check() {
    print_status "INFO" "Running comprehensive system check..."
    echo "=" * 50
    
    if check_requirements; then
        print_status "SUCCESS" "All checks passed! ✓"
        echo ""
        print_status "INFO" "System is ready to run the prediction service"
        return 0
    else
        print_status "ERROR" "Some checks failed ✗"
        echo ""
        print_status "INFO" "Please fix the issues above before starting the service"
        return 1
    fi
}

# =============================================================================
# Main Script Logic
# =============================================================================

show_usage() {
    echo "Usage: $0 {start|stop|restart|status|logs|clean|check}"
    echo ""
    echo "Commands:"
    echo "  start    - Start the prediction service"
    echo "  stop     - Stop the prediction service"
    echo "  restart  - Restart the prediction service"
    echo "  status   - Show service status and information"
    echo "  logs     - Show recent logs (use 'logs follow' for real-time)"
    echo "  clean    - Clean up service files (PID, status)"
    echo "  check    - Run comprehensive system requirements check"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 logs follow"
    echo "  $0 check"
}

main() {
    local command="${1:-}"
    
    case "$command" in
        "start")
            start_service
            ;;
        "stop")
            stop_service
            ;;
        "restart")
            restart_service
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "${2:-}"
            ;;
        "clean")
            clean_service
            ;;
        "check")
            run_check
            ;;
        "")
            show_usage
            exit 1
            ;;
        *)
            print_status "ERROR" "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Trap for cleanup on script exit
trap 'print_status "DEBUG" "Script exiting..."' EXIT

# Run main function
main "$@"