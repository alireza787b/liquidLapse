#!/usr/bin/env bash

set -euo pipefail

HEATMAP_OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEATMAP_BASE_DIR="$HEATMAP_OPS_DIR"
HEATMAP_CONFIG_FILE="$HEATMAP_BASE_DIR/config.yaml"
HEATMAP_SERVICE_SCRIPT="$HEATMAP_BASE_DIR/service.sh"
HEATMAP_PID_FILE="$HEATMAP_BASE_DIR/liquidLapseService.pid"
HEATMAP_BACKUP_STAGING_ROOT="$HEATMAP_BASE_DIR/backup_staging"
HEATMAP_RESTORE_STAGING_ROOT="$HEATMAP_BASE_DIR/restore_staging"
HEATMAP_DEFAULT_MEGA_ROOT="/Root/liquidLapse_backups/heatmap_snapshots_batches"
HEATMAP_DEFAULT_RELAY_STAGING="/root/liquidLapse_backups"
HEATMAP_DEFAULT_RELAY_DOWNLOAD_ROOT="/root/liquidLapse_restore_staging"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ops_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

ops_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

ops_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

ops_fail() {
    ops_error "$*"
    exit 1
}

ops_require_command() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || ops_fail "Required command not found: $cmd"
}

ops_prompt_confirm() {
    local prompt="$1"
    local reply

    if [[ "${OPS_ASSUME_YES:-0}" == "1" ]]; then
        return 0
    fi

    read -r -p "$prompt [y/N]: " reply
    [[ "$reply" =~ ^[Yy]([Ee][Ss])?$ ]]
}

ops_prompt_value() {
    local prompt="$1"
    local default_value="${2:-}"
    local reply

    if [[ -n "$default_value" ]]; then
        read -r -p "$prompt [$default_value]: " reply
        printf '%s\n' "${reply:-$default_value}"
        return
    fi

    read -r -p "$prompt: " reply
    printf '%s\n' "$reply"
}

ops_prompt_secret() {
    local prompt="$1"
    local reply

    read -r -s -p "$prompt: " reply
    echo
    printf '%s\n' "$reply"
}

ops_repo_output_folder_rel() {
    local configured
    configured="$(awk -F':[[:space:]]*' '/^output_folder:/{gsub(/"/, "", $2); print $2; exit}' "$HEATMAP_CONFIG_FILE" 2>/dev/null || true)"
    if [[ -z "$configured" ]]; then
        configured="heatmap_snapshots"
    fi
    printf '%s\n' "$configured"
}

ops_repo_output_folder_abs() {
    local rel_path
    rel_path="$(ops_repo_output_folder_rel)"
    if [[ "$rel_path" = /* ]]; then
        printf '%s\n' "$rel_path"
    else
        printf '%s\n' "$HEATMAP_BASE_DIR/$rel_path"
    fi
}

ops_service_running() {
    if [[ ! -f "$HEATMAP_PID_FILE" ]]; then
        return 1
    fi

    local service_pid
    service_pid="$(tr -d '[:space:]' < "$HEATMAP_PID_FILE" 2>/dev/null || true)"
    [[ -n "$service_pid" ]] || return 1
    kill -0 "$service_pid" 2>/dev/null
}

ops_stop_service_if_running() {
    if ops_service_running; then
        ops_info "Stopping liquidLapse service before file rotation"
        bash "$HEATMAP_SERVICE_SCRIPT" stop
        OPS_SERVICE_WAS_RUNNING=1
    else
        OPS_SERVICE_WAS_RUNNING=0
    fi
}

ops_restart_service_if_needed() {
    if [[ "${OPS_SERVICE_WAS_RUNNING:-0}" == "1" ]]; then
        ops_info "Restarting liquidLapse service"
        bash "$HEATMAP_SERVICE_SCRIPT" start
    fi
}

ops_default_batch_name() {
    printf 'heatmap_batch_%s_%s\n' "$(date -u +%Y%m%dT%H%M%SZ)" "$(hostname -s)"
}

ops_relay_exec() {
    local quoted=()
    local arg

    [[ -n "${MEGA_RELAY_HOST:-}" ]] || ops_fail "MEGA relay host is not configured"

    for arg in "$@"; do
        quoted+=("$(printf '%q' "$arg")")
    done

    ssh -o BatchMode=yes "$MEGA_RELAY_HOST" "${quoted[*]}"
}

ops_relay_shell() {
    local script="$1"

    [[ -n "${MEGA_RELAY_HOST:-}" ]] || ops_fail "MEGA relay host is not configured"
    ssh -o BatchMode=yes "$MEGA_RELAY_HOST" "bash -lc $(printf '%q' "$script")"
}

ops_select_mega_mode() {
    if [[ -n "${MEGA_RELAY_HOST:-}" ]]; then
        ops_require_command ssh
        if ops_relay_shell "command -v mega-whoami >/dev/null 2>&1 && command -v mega-put >/dev/null 2>&1 && command -v mega-get >/dev/null 2>&1 && command -v mega-ls >/dev/null 2>&1"; then
            MEGA_MODE="relay"
            return
        fi
        ops_fail "Relay host '$MEGA_RELAY_HOST' does not have the required MEGA CLI commands"
    fi

    if command -v mega-whoami >/dev/null 2>&1 && command -v mega-put >/dev/null 2>&1 && command -v mega-get >/dev/null 2>&1 && command -v mega-ls >/dev/null 2>&1; then
        MEGA_MODE="local"
        return
    fi

    ops_fail "MEGA CLI not found locally. Install it or pass --relay-host user@host"
}

ops_mega_whoami() {
    if [[ "${MEGA_MODE:-}" == "local" ]]; then
        mega-whoami
        return
    fi

    ops_relay_exec mega-whoami
}

ops_prompt_mega_credentials_if_missing() {
    if [[ -z "${MEGA_EMAIL:-}" ]]; then
        MEGA_EMAIL="$(ops_prompt_value "Enter MEGA account email")"
    fi

    if [[ -z "${MEGA_PASSWORD:-}" ]]; then
        MEGA_PASSWORD="$(ops_prompt_secret "Enter MEGA account password")"
    fi

    if [[ -z "${MEGA_AUTH_CODE:-}" ]]; then
        MEGA_AUTH_CODE="${MEGA_AUTH_CODE:-}"
    fi
}

ops_mega_login() {
    local login_args=()

    [[ -n "${MEGA_EMAIL:-}" ]] || ops_fail "MEGA email is required for login"
    [[ -n "${MEGA_PASSWORD:-}" ]] || ops_fail "MEGA password is required for login"

    login_args+=("$MEGA_EMAIL" "$MEGA_PASSWORD")
    if [[ -n "${MEGA_AUTH_CODE:-}" ]]; then
        login_args=("--auth-code=$MEGA_AUTH_CODE" "${login_args[@]}")
    fi

    if [[ "${MEGA_MODE:-}" == "local" ]]; then
        mega-login "${login_args[@]}"
        return
    fi

    ops_relay_exec mega-login "${login_args[@]}"
}

ops_ensure_mega_session() {
    if ops_mega_whoami >/dev/null 2>&1; then
        ops_info "Using existing MEGA session"
        return
    fi

    ops_warn "No active MEGA session found for ${MEGA_MODE} mode"
    ops_prompt_mega_credentials_if_missing
    ops_info "Logging into MEGA"
    ops_mega_login >/dev/null
    ops_mega_whoami >/dev/null
}

ops_mega_mkdir() {
    if [[ "${MEGA_MODE:-}" == "local" ]]; then
        mega-mkdir -p "$1"
        return
    fi

    ops_relay_exec mega-mkdir -p "$1"
}

ops_mega_put_dir() {
    local source_path="$1"
    local remote_root="$2"

    if [[ "${MEGA_MODE:-}" == "local" ]]; then
        mega-put -c "$source_path" "$remote_root/"
        return
    fi

    ops_relay_exec mega-put -c "$source_path" "$remote_root/"
}

ops_mega_get_dir() {
    local remote_path="$1"
    local local_parent="$2"

    if [[ "${MEGA_MODE:-}" == "local" ]]; then
        mega-get "$remote_path" "$local_parent"
        return
    fi

    ops_fail "ops_mega_get_dir only supports local MEGA mode"
}

ops_mega_file_count() {
    local remote_path="$1"

    if [[ "${MEGA_MODE:-}" == "local" ]]; then
        mega-find "$remote_path" --type=f | wc -l | tr -d '[:space:]'
        return
    fi

    ops_relay_shell "mega-find $(printf '%q' "$remote_path") --type=f | wc -l | tr -d '[:space:]'"
}

ops_mega_list_batch_dirs() {
    local remote_root="$1"

    if [[ "${MEGA_MODE:-}" == "local" ]]; then
        mega-ls "$remote_root"
        return
    fi

    ops_relay_exec mega-ls "$remote_root"
}

ops_mega_path_exists() {
    local remote_path="$1"

    if [[ "${MEGA_MODE:-}" == "local" ]]; then
        mega-find "$remote_path" --type=d >/dev/null 2>&1
        return
    fi

    ops_relay_exec mega-find "$remote_path" --type=d >/dev/null 2>&1
}

ops_generate_manifests() {
    local batch_dir="$1"
    local summary_note="${2:-}"
    local manifest_dir="$batch_dir/manifests"

    mkdir -p "$manifest_dir"

    python3 - "$batch_dir" "$summary_note" <<'PY'
import collections
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone

batch_dir = os.path.abspath(sys.argv[1])
summary_note = sys.argv[2]
manifest_dir = os.path.join(batch_dir, "manifests")
sha_path = os.path.join(manifest_dir, "sha256sums.txt")
dup_path = os.path.join(manifest_dir, "duplicate_hashes.txt")
png_path = os.path.join(manifest_dir, "png_validation.txt")
summary_path = os.path.join(manifest_dir, "summary.json")

roots = []
for candidate in ("heatmap_snapshots", "metadata"):
    abs_candidate = os.path.join(batch_dir, candidate)
    if os.path.isdir(abs_candidate):
        roots.append(candidate)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def validate_png(path):
    try:
        size = os.path.getsize(path)
        if size < 20:
            return False, "too_small"
        with open(path, "rb") as handle:
            signature = handle.read(8)
            if signature != b"\x89PNG\r\n\x1a\n":
                return False, "bad_signature"
            handle.seek(-12, os.SEEK_END)
            tail = handle.read(12)
            if tail[4:8] != b"IEND":
                return False, "missing_iend"
        return True, "ok"
    except OSError as exc:
        return False, f"os_error:{exc}"

entries = []
png_entries = []
for rel_root in roots:
    abs_root = os.path.join(batch_dir, rel_root)
    for root, dirs, files in os.walk(abs_root):
        dirs.sort()
        files.sort()
        for name in files:
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, batch_dir)
            size = os.path.getsize(abs_path)
            digest = sha256_file(abs_path)
            entries.append((rel_path, digest, size))
            if rel_path.startswith("heatmap_snapshots" + os.sep) and rel_path.endswith(".png"):
                ok, reason = validate_png(abs_path)
                png_entries.append((rel_path, digest, size, ok, reason))

entries.sort(key=lambda item: item[0])
png_entries.sort(key=lambda item: item[0])

with open(sha_path, "w", encoding="utf-8") as handle:
    for rel_path, digest, _size in entries:
        handle.write(f"{digest}  {rel_path}\n")

dups = collections.defaultdict(list)
for rel_path, digest, _size, _ok, _reason in png_entries:
    dups[digest].append(rel_path)

duplicate_groups = {digest: paths for digest, paths in dups.items() if len(paths) > 1}
with open(dup_path, "w", encoding="utf-8") as handle:
    if not duplicate_groups:
        handle.write("NONE\n")
    else:
        for digest in sorted(duplicate_groups):
            paths = duplicate_groups[digest]
            handle.write(f"{digest}\t{len(paths)}\t{' | '.join(paths)}\n")

with open(png_path, "w", encoding="utf-8") as handle:
    if not png_entries:
        handle.write("NO_PNG_FILES\n")
    else:
        for rel_path, _digest, size, ok, reason in png_entries:
            state = "OK" if ok else "INVALID"
            handle.write(f"{state}\t{reason}\t{size}\t{rel_path}\n")

heatmap_bytes = sum(size for rel_path, _digest, size in entries if rel_path.startswith("heatmap_snapshots" + os.sep))
metadata_bytes = sum(size for rel_path, _digest, size in entries if rel_path.startswith("metadata" + os.sep))
invalid_pngs = [rel_path for rel_path, _digest, _size, ok, _reason in png_entries if not ok]

summary = {
    "batch_name": os.path.basename(batch_dir),
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "hostname": socket.gethostname(),
    "note": summary_note,
    "total_files": len(entries),
    "total_bytes": sum(size for _rel_path, _digest, size in entries),
    "heatmap_png_count": len(png_entries),
    "heatmap_png_bytes": heatmap_bytes,
    "metadata_file_count": len([1 for rel_path, _digest, _size in entries if rel_path.startswith("metadata" + os.sep)]),
    "metadata_bytes": metadata_bytes,
    "duplicate_png_content_groups": len(duplicate_groups),
    "invalid_png_count": len(invalid_pngs),
    "invalid_pngs": invalid_pngs,
}

with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY
}

ops_verify_local_manifest() {
    local batch_dir="$1"
    local manifest_file="$batch_dir/manifests/sha256sums.txt"

    [[ -f "$manifest_file" ]] || ops_fail "Manifest file not found: $manifest_file"

    ops_info "Verifying local manifest"
    (
        cd "$batch_dir"
        sha256sum -c "manifests/sha256sums.txt" >/dev/null
    )
}

ops_stage_file_count() {
    local batch_dir="$1"
    find "$batch_dir" -type f | wc -l | tr -d '[:space:]'
}

ops_stage_heatmap_count() {
    local heatmap_dir="$1"
    find "$heatmap_dir" -maxdepth 1 -type f -name 'heatmap_*.png' | wc -l | tr -d '[:space:]'
}

ops_git_head() {
    git -C "$HEATMAP_BASE_DIR" rev-parse HEAD 2>/dev/null || printf 'unknown\n'
}

ops_copy_ai_metadata() {
    local destination_root="$1"

    [[ -d "$HEATMAP_BASE_DIR/ai_process" ]] || return 0

    mkdir -p "$destination_root"

    (
        cd "$HEATMAP_BASE_DIR"
        find ai_process -type f -name '*.json' -print0 | rsync -a --from0 --files-from=- . "$destination_root/"
    )
}

ops_write_backup_context() {
    local metadata_dir="$1"
    local batch_name="$2"
    local note="$3"
    local snapshot_rel="$4"
    local snapshot_count="$5"

    mkdir -p "$metadata_dir"
    cat > "$metadata_dir/backup_context.txt" <<EOF
batch_name=$batch_name
generated_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
hostname=$(hostname -s)
git_head=$(ops_git_head)
snapshot_dir=$snapshot_rel
snapshot_count=$snapshot_count
note=$note
EOF
}
