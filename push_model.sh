#!/usr/bin/env bash
# push_model.sh
#
# Interactive script to select a local training run under
# ai_process/<session>/train_*/ and push the entire run folder
# (model + logs) up to a remote VPS, preserving directory structure.
#
# Usage: ./push_model.sh

set -euo pipefail
IFS=$'\n\t'

# ─────────── Configuration Defaults ───────────
DEFAULT_SESSION="test1"
LOCAL_BASE_DIR="${HOME}/liquidLapse/ai_process"
SSH_KEY="${HOME}/.ssh/id_rsa"
DEFAULT_REMOTE="remote_user_name@remote_ip"
# Remote AI-process base: /root/liquidLapse/ai_process
# ───────────────────────────────────────────────

# Color codes
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ─────────── Utility Functions ───────────
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }

prompt() {
  # prompt <varname> <default> <message>
  local var="$1"; local def="$2"; shift 2
  read -rp "$* [$def]: " val
  printf -v "$var" '%s' "${val:-$def}"
}

# Test SSH connectivity; allow password fallback if key fails
check_ssh() {
  ssh -i "$SSH_KEY" -o ConnectTimeout=5 "$1" exit \
    &>/dev/null && return 0 || return 1
}
# ────────────────────────────────────────

echo -e "${GREEN}=== Push Trained Model to VPS ===${NC}"

# 1) Choose AI-process session
prompt SESSION "$DEFAULT_SESSION" "Enter AI-process session name"  
LOCAL_SESSION_DIR="${LOCAL_BASE_DIR}/${SESSION}"
[[ -d "$LOCAL_SESSION_DIR" ]] || err "Session not found: $LOCAL_SESSION_DIR"

# 2) Discover and list train_* runs (newest first)
mapfile -t RUN_DIRS < <(
  find "$LOCAL_SESSION_DIR" -maxdepth 1 -type d -name 'train_*' \
    -printf '%T@ %p\n' | sort -nr | cut -d' ' -f2
)
[[ ${#RUN_DIRS[@]} -gt 0 ]] || err "No training runs found in $LOCAL_SESSION_DIR"

echo "Available training runs:"
for i in "${!RUN_DIRS[@]}"; do
  echo "  $((i+1))) $(basename "${RUN_DIRS[$i]}")"
done

default_idx=1
prompt CHOICE "$default_idx" "Select run by number"
[[ "$CHOICE" =~ ^[0-9]+$ ]] || err "Invalid selection"
(( CHOICE>=1 && CHOICE<=${#RUN_DIRS[@]} )) || err "Selection out of range"

SELECTED_RUN="${RUN_DIRS[$((CHOICE-1))]}"
info "Chosen run: $(basename "$SELECTED_RUN")"

# 3) Confirm local model exists
LOCAL_MODEL_DIR="$SELECTED_RUN"
[[ -d "$LOCAL_MODEL_DIR" ]] || err "Run directory missing: $LOCAL_MODEL_DIR"
[[ -f "$LOCAL_MODEL_DIR/best_model.pt" ]] \
  || err "Model file missing: $LOCAL_MODEL_DIR/best_model.pt"

# 4) Prompt for remote server & target dir
prompt REMOTE     "$DEFAULT_REMOTE" \
  "Enter remote server (user@host)"
prompt REMOTE_DIR "/root/liquidLapse/ai_process/${SESSION}" \
  "Enter remote ai_process/<session> directory"

# 5) Test SSH connectivity
echo -n "Testing SSH to $REMOTE ... "
if check_ssh "$REMOTE"; then
  echo -e "${GREEN}OK (key auth)${NC}"
else
  warn "Key auth failed; will prompt for password if needed"
fi

# 6) Ensure remote train_<timestamp> folder exists
REMOTE_RUN_DIR="${REMOTE_DIR%/}/$(basename "$SELECTED_RUN")"
echo -n "Ensuring remote directory $REMOTE_RUN_DIR ... "
ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p '$REMOTE_RUN_DIR'" \
  && echo -e "${GREEN}Done${NC}"                                 # :contentReference[oaicite:0]{index=0}

# 7) Rsync the entire run folder
info "Syncing local '$LOCAL_MODEL_DIR/' → remote '$REMOTE:$REMOTE_RUN_DIR/'"
rsync -avz --progress -e "ssh -i $SSH_KEY" \
  "$LOCAL_MODEL_DIR/" \
  "$REMOTE":"$REMOTE_RUN_DIR"/                                  # :contentReference[oaicite:1]{index=1}

info "Push complete. Remote run directory now mirrors local."
