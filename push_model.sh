#!/usr/bin/env bash
# push_model.sh
#
# Interactive script to select a trained model run locally and rsync it
# up to a remote VPS. Defaults to the latest training run under
# ai_process/<session>/train_*/best_model.pt, but lets you override.
#
# Usage: ./push_model.sh

set -euo pipefail
IFS=$'\n\t'

# ──────── Defaults ────────
DEFAULT_SESSION="test1"
LOCAL_BASE="${HOME}/liquidLapse/ai_process"
SSH_KEY="${HOME}/.ssh/id_rsa"
REMOTE_DEFAULT="root@nb1.joomtalk.ir"
REMOTE_BASE_DIR="~/liquidLapse/ai_process"
# ──────────────────────────

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ──────── Utility Functions ────────
err() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

prompt() {
  local name="$1" def="$2" msg="$3"
  echo -ne "${YELLOW}$msg${NC} [${def}]: "
  read -r reply
  echo "${reply:-$def}"
}

# ──────── Script Start ────────
echo -e "${GREEN}--- Push Trained Model Script ---${NC}"

# 1) Choose session
SESSION=$(prompt SESSION "$DEFAULT_SESSION" \
  "Enter session name (subfolder under ai_process)")
LOCAL_SESSION_DIR="${LOCAL_BASE}/${SESSION}"
[[ -d "$LOCAL_SESSION_DIR" ]] || err "Session dir not found: $LOCAL_SESSION_DIR"

# 2) List train_* runs
MAPFILE -t RUN_DIRS < <(find "$LOCAL_SESSION_DIR" -maxdepth 1 -type d -name 'train_*' | sort)
if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  err "No train_* directories in $LOCAL_SESSION_DIR"
fi

echo "Available training runs:"
for i in "${!RUN_DIRS[@]}"; do
  run=$(basename "${RUN_DIRS[$i]}")
  echo "  [$((i+1))] $run"
done

# Default to last (newest)
DEFAULT_IDX=${#RUN_DIRS[@]}
IDX=$(prompt IDX "$DEFAULT_IDX" \
  "Select run by number (1–${#RUN_DIRS[@]})")
if ! [[ "$IDX" =~ ^[0-9]+$ ]] || (( IDX<1 || IDX> ${#RUN_DIRS[@]} )); then
  err "Invalid selection"
fi
SELECTED_RUN="${RUN_DIRS[$((IDX-1))]}"
MODEL_PATH="${SELECTED_RUN}/best_model.pt"
[[ -f "$MODEL_PATH" ]] || err "Model file not found: $MODEL_PATH"
info "Selected model: $(basename "$SELECTED_RUN")/best_model.pt"

# 3) Get remote info
REMOTE=$(prompt REMOTE "$REMOTE_DEFAULT" \
  "Enter remote user@host")
REMOTE_DIR=$(prompt REMDIR "$REMOTE_BASE_DIR/$SESSION" \
  "Enter remote ai_process/session dir")

# 4) Verify SSH connectivity
echo -n "Checking SSH connectivity to $REMOTE... "
ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$REMOTE" "echo ok" \
  &>/dev/null || err "SSH to $REMOTE failed"
echo -e "${GREEN}OK${NC}"

# 5) Ensure remote directory exists
echo -n "Ensuring remote directory $REMOTE_DIR exists... "
ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p $REMOTE_DIR" \
  && echo -e "${GREEN}Done${NC}"

# 6) Transfer model via rsync
info "Pushing best_model.pt to $REMOTE:$REMOTE_DIR/"
rsync -avz \
  -e "ssh -i $SSH_KEY" \
  "$MODEL_PATH" \
  "$REMOTE":"$REMOTE_DIR"/
info "Model pushed successfully."

echo -e "${GREEN}All done!${NC}"
