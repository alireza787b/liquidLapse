#!/usr/bin/env bash
# sync_heatmaps.sh
#
# Interactive script to pull only NEW heatmap PNGs from your VPS to local.
# Guides you through server, paths, and options, then runs rsync with progress.
#
# Usage: ./sync_heatmaps.sh           # real run
#        ./sync_heatmaps.sh --dry-run # preview only

set -euo pipefail
IFS=$'\n\t'

# ────────── Defaults ──────────
DEFAULT_REMOTE="root@nb1.joomtalk.ir"
# Use absolute path on remote to avoid '~' expansion issues :contentReference[oaicite:0]{index=0}
DEFAULT_REMOTE_DIR="/root/liquidLapse/heatmap_snapshots"
LOCAL_DIR="${HOME}/liquidLapse/heatmap_snapshots"
SSH_KEY="${HOME}/.ssh/id_rsa"
# ─────────────────────────────

# Colors for clarity
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ─────── Utility ───────
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }

prompt() {
  local var="$1"; local def="$2"; shift 2
  read -rp "$* [$def]: " val
  printf -v "$var" '%s' "${val:-$def}"
}
# ─────────────────────────

echo -e "${GREEN}=== Heatmap Sync Script ===${NC}"
echo "This will pull only NEW PNGs from your VPS to your local folder."
echo

# 1) Prompt for remote server & path
prompt REMOTE      "$DEFAULT_REMOTE"     "Enter remote server (user@host)" 
prompt REMOTE_DIR  "$DEFAULT_REMOTE_DIR" "Enter remote heatmap directory (absolute)" :contentReference[oaicite:1]{index=1}

# 2) Prompt for local destination
prompt LOCAL_DIR   "$LOCAL_DIR"          "Enter local destination directory"
mkdir -p "$LOCAL_DIR"  # ensure exists :contentReference[oaicite:2]{index=2}

# 3) Dry-run?
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo -e "${YELLOW}[DRY-RUN]${NC} No files will be actually copied."
fi

# 4) Test SSH connectivity (allowing password fallback) 
echo -n "Testing SSH to ${REMOTE}... "
if ssh -i "$SSH_KEY" -o ConnectTimeout=5 "$REMOTE" exit; then
  echo -e "${GREEN}OK${NC}"
else
  warn "SSH key login failed, will prompt for password if needed"
fi

# 5) Build rsync options
RSYNC_OPTS=(-avz --ignore-existing --progress -e "ssh -i $SSH_KEY")
RSYNC_SRC="${REMOTE}:${REMOTE_DIR%/}/"   # ensure single trailing slash
RSYNC_DST="$LOCAL_DIR/"

# 6) Run rsync
echo; info "Running rsync..."
if $DRY_RUN; then
  rsync "${RSYNC_OPTS[@]}" --dry-run "$RSYNC_SRC" "$RSYNC_DST"
else
  rsync "${RSYNC_OPTS[@]}" "$RSYNC_SRC" "$RSYNC_DST"
fi

# 7) Done
echo; info "Sync complete."
echo "New files (if any) are now in: ${YELLOW}${LOCAL_DIR}${NC}"
