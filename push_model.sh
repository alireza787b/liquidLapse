#!/usr/bin/env bash
# push_model.sh
#
# Interactive script (run locally) to select a trained model run and
# rsync its best_model.pt up to your VPS.

set -euo pipefail
IFS=$'\n\t'

# ─────────── CONFIGURATION DEFAULTS ───────────
DEFAULT_SESSION="test1"
LOCAL_BASE_DIR="${HOME}/liquidLapse/ai_process"
SSH_KEY="${HOME}/.ssh/id_rsa"
DEFAULT_REMOTE="root@nb1.joomtalk.ir"
# Remote target base: ~/liquidLapse/ai_process/<session>
# ───────────────────────────────────────────────

# Color codes
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ─────────── Utility Functions ───────────
err()   { echo -e "${RED}[ERROR]   $*${NC}" >&2;  exit 1; }
info()  { echo -e "${GREEN}[INFO]    $*${NC}"; }
warn()  { echo -e "${YELLOW}[WARNING] $*${NC}"; }

prompt() {
  # prompt <varname> <default> <message>
  local var="$1"; local def="$2"; shift 2
  read -rp "$* [$def]: " val
  printf -v "$var" '%s' "${val:-$def}"
}

check_ssh() {
    ssh -i "$SSH_KEY" "$REMOTE" exit
    &>/dev/null || return 1
  return 0
}
# ────────────────────────────────────────

echo -e "${GREEN}=== Push Trained Model to VPS ===${NC}"

# 1) Session
prompt SESSION "$DEFAULT_SESSION" "Enter AI-process session name"
LOCAL_SESSION="${LOCAL_BASE_DIR}/${SESSION}"
[[ -d "$LOCAL_SESSION" ]] || err "Session directory not found: $LOCAL_SESSION"

# 2) List available train_* runs (sorted newest first)
mapfile -t RUNS < <(find "$LOCAL_SESSION" -maxdepth 1 -type d -name 'train_*' \
                    -printf '%T@ %p\n' | sort -nr | cut -d' ' -f2)
if [[ ${#RUNS[@]} -eq 0 ]]; then
  err "No training runs (train_*) found under $LOCAL_SESSION"
fi

echo "Available training runs:"
for idx in "${!RUNS[@]}"; do
  echo "  $((idx+1))) $(basename "${RUNS[$idx]}")"
done

default_idx=1
prompt CHOICE "$default_idx" "Select run by number"
[[ "$CHOICE" =~ ^[0-9]+$ ]] || err "Invalid selection"
(( CHOICE>=1 && CHOICE<=${#RUNS[@]} )) || err "Selection out of range"

SELECTED_RUN="${RUNS[$((CHOICE-1))]}"
MODEL_PATH="${SELECTED_RUN}/best_model.pt"
[[ -f "$MODEL_PATH" ]] || err "Model file not found: $MODEL_PATH"

info "Chosen model: $(basename "$SELECTED_RUN")/best_model.pt"

# 3) Remote server & path
prompt REMOTE "$DEFAULT_REMOTE" "Enter remote server (user@host)"
prompt REMOTE_BASE "~/liquidLapse/ai_process/${SESSION}" \
       "Enter remote base directory for session"

# 4) Test SSH connectivity
echo -n "Testing SSH to $REMOTE ... "
if check_ssh "$REMOTE"; then
  echo -e "${GREEN}OK${NC}"
else
  err "SSH connection to $REMOTE failed"
fi

# 5) Ensure remote directory exists
echo -n "Ensuring remote dir $REMOTE_BASE exists ... "
ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p \"$REMOTE_BASE\"" \
  && echo -e "${GREEN}Done${NC}"

# 6) Rsync the model
info "Syncing $MODEL_PATH → $REMOTE:$REMOTE_BASE/"
rsync -avz \
  -e "ssh -i $SSH_KEY" \
  "$MODEL_PATH" \
  "$REMOTE":"$REMOTE_BASE"/

info "Upload complete."

