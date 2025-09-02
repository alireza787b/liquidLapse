#!/bin/bash
# Fixed service.sh - Production-Ready Service Manager
# CORRECTED: Function scoping issues resolved

set -euo pipefail

# Configuration
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$BASE_DIR/venv"
PYTHON="$VENV_DIR/bin/python"
SCRIPT="$BASE_DIR/liquidLapse.py"

# Files
PIDFILE="$BASE_DIR/liquidLapseService.pid"
STATUSFILE="$BASE_DIR/liquidLapseService.status"
LOGFILE="$BASE_DIR/liquidLapseService.log"
HEALTH_LOG="$BASE_DIR/service_health.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Simple logging functions (inline for subshell compatibility)
log_to_health() {
    echo "[$(date -Iseconds)] $1" >> "$HEALTH_LOG"
}

validate_environment() {
    local errors=0
    echo -e "${BLUE}[INFO]${NC} Validating environment..."
    
    if [ ! -d "$VENV_DIR" ] || [ ! -x "$VENV_DIR/bin/python" ]; then
        echo -e "${RED}[ERROR]${NC} Virtual environment invalid: $VENV_DIR"
        ((errors++))
    fi
    
    if [ ! -f "$SCRIPT" ] || [ ! -r "$SCRIPT" ]; then
        echo -e "${RED}[ERROR]${NC} Main script not found: $SCRIPT"
        ((errors++))
    fi
    
    if [ ! -f "$BASE_DIR/config.yaml" ]; then
        echo -e "${RED}[ERROR]${NC} Config file not found: $BASE_DIR/config.yaml"
        ((errors++))
    fi
    
    if ! "$PYTHON" -c "import selenium, yaml, requests" 2>/dev/null; then
        echo -e "${RED}[ERROR]${NC} Required Python dependencies not available"
        echo -e "${RED}[ERROR]${NC} Testing individual imports:"
        "$PYTHON" -c "import selenium" 2>&1 | head -1 || echo "  - selenium: FAILED"
        "$PYTHON" -c "import yaml" 2>&1 | head -1 || echo "  - yaml: FAILED"  
        "$PYTHON" -c "import requests" 2>&1 | head -1 || echo "  - requests: FAILED"
        ((errors++))
    fi
    
    if [ $errors -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS]${NC} Environment validation passed"
        return 0
    else
        echo -e "${RED}[ERROR]${NC} Environment validation failed with $errors errors"
        return 1
    fi
}

cleanup_resources() {
    echo -e "${BLUE}[INFO]${NC} Cleaning up system resources..."
    
    # Kill Chrome processes
    local chrome_pids=$(pgrep -f "chrome" 2>/dev/null || true)
    if [ -n "$chrome_pids" ]; then
        echo "$chrome_pids" | while read -r pid; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                kill -TERM "$pid" 2>/dev/null || true
                sleep 1
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
    fi
    
    # Clean temp directories
    find /tmp -maxdepth 1 -name ".org.chromium.Chromium.*" -type d -mmin +60 -exec rm -rf {} + 2>/dev/null || true
    find /tmp -maxdepth 1 -name "scoped_dir*" -type d -mmin +60 -exec rm -rf {} + 2>/dev/null || true
    
    # Memory cleanup if low
    local available_mem=$(free -m | awk 'NR==2{print $7}')
    if [ "$available_mem" -lt 300 ]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    fi
}
# REPLACE the start_service function in service.sh with this robust version

start_service() {
    echo -e "${BLUE}[INFO]${NC} Starting liquidLapse service..."
    
    if ! validate_environment; then
        echo -e "${RED}[ERROR]${NC} Environment validation failed"
        exit 1
    fi
    
    if [ -f "$PIDFILE" ]; then
        local existing_pid=$(cat "$PIDFILE" 2>/dev/null || echo "")
        if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
            echo -e "${GREEN}[SUCCESS]${NC} Service already running (PID: $existing_pid)"
            exit 0
        else
            echo -e "${BLUE}[INFO]${NC} Removing stale PID file"
            rm -f "$PIDFILE"
        fi
    fi
    
    cleanup_resources
    
    # Initialize logs
    echo "=== Service Starting at $(date -Iseconds) ===" >> "$LOGFILE"
    echo "[$(date -Iseconds)] Service initialization" >> "$HEALTH_LOG"
    
    cd "$BASE_DIR"
    
    # ROBUST SERVICE LOOP - NO EXIT ON MINOR ERRORS
    nohup bash -c '
        # REMOVE strict error handling for service loop
        set +e  # Allow errors to continue
        
        # Set environment
        export PATH="'"$VENV_DIR"'/bin:$PATH"
        export PYTHONPATH="'"$BASE_DIR"':${PYTHONPATH:-}"
        cd "'"$BASE_DIR"'"
        
        # Service variables
        success_count=0
        failure_count=0
        consecutive_failures=0
        max_consecutive_failures=3
        service_start_time=$(date +%s)
        
        echo "[$(date -Iseconds)] ROBUST service starting with PID $$" >> "'"$HEALTH_LOG"'"
        echo "[$(date -Iseconds)] Working directory: $(pwd)" >> "'"$HEALTH_LOG"'"
        echo "[$(date -Iseconds)] Python executable: $(which python || echo '"'"'"$VENV_DIR/bin/python"'"'"')" >> "'"$HEALTH_LOG"'"
        
        # INFINITE LOOP WITH ERROR RESILIENCE
        while true; do
            # Wrap each cycle in error protection
            (
                cycle_start_time=$(date +%s)
                current_time=$(date "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date)
                
                echo "[$(date -Iseconds)] === Starting cycle (success: $success_count, failures: $failure_count) ===" >> "'"$HEALTH_LOG"'"
                
                # Get interval from config (with fallback)
                period=$(python -c "
import yaml
import sys
try:
    with open(\"config.yaml\", \"r\") as f:
        config = yaml.safe_load(f)
    print(int(config.get(\"check_interval\", 300)))
except Exception as e:
    print(\"Config error:\", str(e), file=sys.stderr)
    print(300)
" 2>>"'$HEALTH_LOG'" || echo "300")
                
                # Validate period
                if ! [[ "$period" =~ ^[0-9]+$ ]] || [ "$period" -lt 60 ]; then
                    period=300
                    echo "[$(date -Iseconds)] Invalid period, using default: $period" >> "'"$HEALTH_LOG"'"
                fi
                
                uptime_seconds=$((cycle_start_time - service_start_time))
                
                # Update status (with error protection)
                {
                    echo "CAPTURE_ATTEMPT: $current_time"
                    echo "INTERVAL_SECONDS: $period"
                    echo "SUCCESS_COUNT: $success_count"
                    echo "FAILURE_COUNT: $failure_count"
                    echo "CONSECUTIVE_FAILURES: $consecutive_failures"
                    echo "UPTIME_SECONDS: $uptime_seconds"
                } > "'"$STATUSFILE"'" 2>/dev/null || true
                
                # Circuit breaker
                if [ "$consecutive_failures" -ge "$max_consecutive_failures" ]; then
                    echo "[$(date -Iseconds)] Circuit breaker active - waiting 300s" >> "'"$HEALTH_LOG"'"
                    echo "CIRCUIT_BREAKER_ACTIVE: Extended delay" > "'"$STATUSFILE"'" 2>/dev/null || true
                    sleep 300 || sleep 60  # Fallback shorter sleep
                    consecutive_failures=0
                    return 0  # Continue main loop
                fi
                
                # Network check (with timeout and fallback)
                network_ok=false
                if timeout 15 curl -s --connect-timeout 10 https://www.google.com > /dev/null 2>&1; then
                    network_ok=true
                elif timeout 15 curl -s --connect-timeout 10 https://1.1.1.1 > /dev/null 2>&1; then
                    network_ok=true
                fi
                
                if [ "$network_ok" = false ]; then
                    echo "[$(date -Iseconds)] Network connectivity failed" >> "'"$HEALTH_LOG"'"
                    echo "NETWORK_ERROR: No connectivity at $current_time" > "'"$STATUSFILE"'" 2>/dev/null || true
                    sleep 120 || sleep 60
                    return 0  # Continue main loop
                fi
                
                # Execute capture with multiple fallbacks
                echo "[$(date -Iseconds)] Starting capture attempt" >> "'"$HEALTH_LOG"'"
                
                execution_start=$(date +%s)
                exit_code=1
                
                # Try with timeout
                if timeout 240 python liquidLapse.py >> "'"$LOGFILE"'" 2>&1; then
                    exit_code=0
                elif timeout 240 "'"$VENV_DIR"'/bin/python" liquidLapse.py >> "'"$LOGFILE"'" 2>&1; then
                    exit_code=0
                else
                    exit_code=1
                fi
                
                execution_duration=$(($(date +%s) - execution_start))
                echo "[$(date -Iseconds)] Capture completed with exit code: $exit_code (duration: ${execution_duration}s)" >> "'"$HEALTH_LOG"'"
                
                # Process results
                if [ $exit_code -eq 0 ]; then
                    # Success
                    success_time=$(date "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date)
                    success_count=$((success_count + 1))
                    consecutive_failures=0
                    
                    echo "[$(date -Iseconds)] SUCCESS #$success_count (${execution_duration}s)" >> "'"$HEALTH_LOG"'"
                    echo "CAPTURE_SUCCESS: $success_time (duration: ${execution_duration}s)" > "'"$STATUSFILE"'" 2>/dev/null || true
                    
                else
                    # Failure
                    failure_time=$(date "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date)
                    failure_count=$((failure_count + 1))
                    consecutive_failures=$((consecutive_failures + 1))
                    
                    echo "[$(date -Iseconds)] FAILURE #$failure_count (consecutive: $consecutive_failures)" >> "'"$HEALTH_LOG"'"
                    echo "CAPTURE_FAILURE: $failure_time (exit: $exit_code)" > "'"$STATUSFILE"'" 2>/dev/null || true
                    
                    # Show recent errors for debugging
                    echo "[$(date -Iseconds)] Recent log output:" >> "'"$HEALTH_LOG"'"
                    tail -3 "'"$LOGFILE"'" 2>/dev/null >> "'"$HEALTH_LOG"'" || true
                fi
                
                # GUARANTEED INTERVAL TIMING
                cycle_duration=$(($(date +%s) - cycle_start_time))
                sleep_duration=$((period - cycle_duration))
                
                if [ $sleep_duration -gt 0 ]; then
                    echo "[$(date -Iseconds)] Cycle completed in ${cycle_duration}s, sleeping ${sleep_duration}s until next cycle" >> "'"$HEALTH_LOG"'"
                    
                    # Sleep in chunks to allow interruption
                    while [ $sleep_duration -gt 0 ]; do
                        chunk_sleep=$((sleep_duration > 60 ? 60 : sleep_duration))
                        sleep $chunk_sleep || sleep 30  # Fallback sleep
                        sleep_duration=$((sleep_duration - chunk_sleep))
                    done
                else
                    echo "[$(date -Iseconds)] Cycle overrun: ${cycle_duration}s > ${period}s - starting next cycle immediately" >> "'"$HEALTH_LOG"'"
                fi
                
                echo "[$(date -Iseconds)] === Cycle completed, starting next ===" >> "'"$HEALTH_LOG"'"
                
            ) || {
                # Cycle error handler
                echo "[$(date -Iseconds)] ERROR: Cycle failed, continuing service..." >> "'"$HEALTH_LOG"'"
                sleep 60 || true  # Brief pause before next cycle
            }
            
        done  # End of while true loop
        
    ' >> "$LOGFILE" 2>&1 &
    
    local service_pid=$!
    echo "$service_pid" > "$PIDFILE"
    
    # Verify service started and is stable
    sleep 5
    if kill -0 "$service_pid" 2>/dev/null; then
        echo -e "${GREEN}[SUCCESS]${NC} Service started and running (PID: $service_pid)"
        
        # Additional verification - check for 10 more seconds
        sleep 10
        if kill -0 "$service_pid" 2>/dev/null; then
            echo -e "${GREEN}[SUCCESS]${NC} Service confirmed stable after 15 seconds"
        else
            echo -e "${RED}[ERROR]${NC} Service died within 15 seconds of startup"
            rm -f "$PIDFILE"
            echo "[$(date -Iseconds)] Service died during startup verification" >> "$HEALTH_LOG"
            exit 1
        fi
    else
        echo -e "${RED}[ERROR]${NC} Service failed to start"
        rm -f "$PIDFILE"
        echo "[$(date -Iseconds)] Service failed to start" >> "$HEALTH_LOG"
        exit 1
    fi
}

stop_service() {
    echo -e "${BLUE}[INFO]${NC} Stopping service..."
    
    if [ ! -f "$PIDFILE" ]; then
        echo -e "${RED}[ERROR]${NC} Service not running (no PID file)"
        exit 1
    fi
    
    local service_pid=$(cat "$PIDFILE" 2>/dev/null || echo "")
    
    if [ -z "$service_pid" ] || ! kill -0 "$service_pid" 2>/dev/null; then
        echo -e "${YELLOW}[WARN]${NC} Service not running (stale PID)"
        rm -f "$PIDFILE"
        exit 0
    fi
    
    # Graceful shutdown
    kill -TERM "$service_pid" 2>/dev/null || true
    
    # Wait for shutdown
    local wait_time=0
    while [ $wait_time -lt 15 ] && kill -0 "$service_pid" 2>/dev/null; do
        sleep 1
        ((wait_time++))
    done
    
    # Force kill if needed
    if kill -0 "$service_pid" 2>/dev/null; then
        echo -e "${YELLOW}[WARN]${NC} Force killing service"
        kill -KILL "$service_pid" 2>/dev/null || true
    fi
    
    cleanup_resources
    rm -f "$PIDFILE"
    
    echo "[$(date -Iseconds)] Service stopped" >> "$HEALTH_LOG"
    echo -e "${GREEN}[SUCCESS]${NC} Service stopped"
}

status_service() {
    echo -e "${BLUE}[INFO]${NC} === Service Status ==="
    
    if [ -f "$PIDFILE" ]; then
        local service_pid=$(cat "$PIDFILE" 2>/dev/null || echo "")
        if [ -n "$service_pid" ] && kill -0 "$service_pid" 2>/dev/null; then
            echo -e "${GREEN}[SUCCESS]${NC} Service RUNNING (PID: $service_pid)"
            ps -p "$service_pid" -o pid,ppid,%cpu,%mem,etime,cmd --no-headers 2>/dev/null || true
        else
            echo -e "${RED}[ERROR]${NC} Service NOT RUNNING (stale PID)"
        fi
    else
        echo -e "${RED}[ERROR]${NC} Service NOT RUNNING (no PID file)"
    fi
    
    echo ""
    if [ -f "$STATUSFILE" ]; then
        echo -e "${BLUE}=== Current Status ===${NC}"
        cat "$STATUSFILE"
        echo ""
    fi
    
    echo -e "${BLUE}=== Recent Activity ===${NC}"
    tail -10 "$HEALTH_LOG" 2>/dev/null || echo "No health log"
    
    echo ""
    echo -e "${BLUE}=== Latest Captures ===${NC}"
    if [ -d "$BASE_DIR/heatmap_snapshots" ]; then
        ls -lt "$BASE_DIR/heatmap_snapshots"/*.png 2>/dev/null | head -5 | while read -r line; do
            echo "$line" | awk '{print $6, $7, $8, $9, "("$5" bytes)"}'
        done
    else
        echo "No snapshots directory"
    fi
}

health_check() {
    echo -e "${BLUE}[INFO]${NC} === Health Check ==="
    
    local issues=0
    
    # Service status
    if [ -f "$PIDFILE" ]; then
        local service_pid=$(cat "$PIDFILE" 2>/dev/null || echo "")
        if [ -n "$service_pid" ] && kill -0 "$service_pid" 2>/dev/null; then
            echo -e "${GREEN}[SUCCESS]${NC} Process: RUNNING"
        else
            echo -e "${RED}[ERROR]${NC} Process: NOT RUNNING"
            ((issues++))
        fi
    else
        echo -e "${RED}[ERROR]${NC} Process: NO PID FILE"
        ((issues++))
    fi
    
    # Log freshness
    if [ -f "$LOGFILE" ]; then
        local log_age=$(( $(date +%s) - $(stat -c %Y "$LOGFILE" 2>/dev/null || echo "0") ))
        if [ $log_age -lt 600 ]; then
            echo -e "${GREEN}[SUCCESS]${NC} Logs: FRESH (${log_age}s ago)"
        else
            echo -e "${YELLOW}[WARN]${NC} Logs: STALE (${log_age}s ago)"
            ((issues++))
        fi
    fi
    
    # Network connectivity
    if curl -s --connect-timeout 10 https://www.google.com > /dev/null 2>&1; then
        echo -e "${GREEN}[SUCCESS]${NC} Network: CONNECTED"
    else
        echo -e "${RED}[ERROR]${NC} Network: DISCONNECTED"
        ((issues++))
    fi
    
    # Memory
    local mem=$(free -m | awk 'NR==2{print $7}')
    if [ "$mem" -gt 300 ]; then
        echo -e "${GREEN}[SUCCESS]${NC} Memory: ${mem}MB available"
    else
        echo -e "${YELLOW}[WARN]${NC} Memory: LOW (${mem}MB)"
        ((issues++))
    fi
    
    echo ""
    if [ $issues -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS]${NC} Overall: HEALTHY"
    else
        echo -e "${YELLOW}[WARN]${NC} Overall: $issues issues detected"
    fi
    
    return $issues
}

restart_service() {
    echo -e "${BLUE}[INFO]${NC} Restarting service..."
    stop_service || true
    sleep 3
    start_service
}

case "${1:-}" in
    start) start_service ;;
    stop) stop_service ;;
    restart) restart_service ;;
    status) status_service ;;
    health) health_check ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|health}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the service"
        echo "  stop    - Stop the service" 
        echo "  restart - Restart the service"
        echo "  status  - Show service status"
        echo "  health  - Health check"
        exit 1
        ;;
esac