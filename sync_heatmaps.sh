#!/usr/bin/env bash
# sync_heatmaps.sh
#
# Interactive script to pull new heatmap PNGs from a remote VPS via rsync.
# Prompts for remote server, paths, and options, then runs rsync with progress.
#
# Usage: ./sync_heatmaps.sh
#   or:  ./sync_heatmaps.sh --dry-run

set -euo pipefail
IFS=$'\n\t'

# ──────── Configuration Defaults ────────
DEFAULT_REMOTE="root@nb1.joomtalk.ir"                     # remote user@host
DEFAULT_REMOTE_DIR="~/liquidLapse/heatmap_snapshots"       # remote heatmaps path
LOCAL_DIR="${HOME}/liquidLapse/heatmap_snapshots"          # local destination
SSH_KEY="${HOME}/.ssh/id_rsa"                              # SSH private key
# ──────────────────────────────────────────

# Colors for output
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ──────── Utility Functions ────────

# Print an error and exit
err() {
  echo -e "${RED}[ERROR]${NC} $*" >&2
  exit 1
}

# Prompt user with default
prompt() {
  local var="$1"; shift
  local def="$1"; shift
  echo -ne "${YELLOW}$*${NC} [${def}]: "
  read -r reply
  echo "${reply:-$def}"
}

# Validate SSH connectivity
check_ssh() {
  ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$1" "echo 2>&1" >/dev/null \
    || err "Cannot SSH to '$1'. Check hostname/key."
}

# ──────── Main Script ────────

echo -e "${GREEN}--- Heatmap Sync Script ---${NC}"
echo "This will pull only NEW PNGs from your VPS to local."
echo

# 1) Gather parameters interactively
REMOTE=$(prompt REMOTE "$DEFAULT_REMOTE" \
  "Enter remote server (user@host)")                                      # :contentReference[oaicite:4]{index=4}

REMOTE_DIR=$(prompt REM_DIR "$DEFAULT_REMOTE_DIR" \
  "Enter remote heatmap directory")                                        # :contentReference[oaicite:5]{index=5}

LOCAL_DIR=$(prompt LOCAL_DIR "$LOCAL_DIR" \
  "Enter local destination directory")                                     # :contentReference[oaicite:6]{index=6}

# 2) Dry-run?
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo -e "${YELLOW}[DRY-RUN]${NC} No files will actually be copied."
fi

# 3) Check SSH connectivity
echo -n "Checking SSH connectivity to ${REMOTE}... "
check_ssh "$REMOTE" && echo -e "${GREEN}OK${NC}"

# 4) Ensure local directory exists
echo -n "Ensuring local directory exists at ${LOCAL_DIR}... "
mkdir -p "$LOCAL_DIR" && echo -e "${GREEN}Done${NC}"                 # :contentReference[oaicite:7]{index=7}

# 5) Perform rsync
RSYNC_OPTS=(-avz --ignore-existing --progress -e "ssh -i $SSH_KEY")
RSYNC_SRC="${REMOTE}:${REMOTE_DIR%/}/"   # strip trailing slash, then add one
RSYNC_DST="$LOCAL_DIR/"

echo
echo -e "${GREEN}Running rsync...${NC}"
if $DRY_RUN; then
  rsync "${RSYNC_OPTS[@]}" --dry-run "$RSYNC_SRC" "$RSYNC_DST"          # :contentReference[oaicite:8]{index=8}
else
  rsync "${RSYNC_OPTS[@]}" "$RSYNC_SRC" "$RSYNC_DST"
fi

# 6) Report results
echo
echo -e "${GREEN}Sync complete.${NC}"
echo "New files (if any) have been transferred to:"
echo -e "  ${YELLOW}${LOCAL_DIR}${NC}"
