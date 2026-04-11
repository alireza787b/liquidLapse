#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/heatmap_backup_common.sh"

MEGA_REMOTE_ROOT="$HEATMAP_DEFAULT_MEGA_ROOT"
MEGA_RELAY_DOWNLOAD_ROOT="$HEATMAP_DEFAULT_RELAY_DOWNLOAD_ROOT"
MEGA_RELAY_HOST="${MEGA_RELAY_HOST:-}"
MEGA_EMAIL="${MEGA_EMAIL:-}"
MEGA_PASSWORD="${MEGA_PASSWORD:-}"
MEGA_AUTH_CODE="${MEGA_AUTH_CODE:-}"
OPS_ASSUME_YES=0
OPS_DRY_RUN=0
OPS_LIST_ONLY=0
OPS_ACTIVATE_LIVE=0
OPS_CLEANUP_LOCAL_DOWNLOAD=0
OPS_CLEANUP_RELAY_DOWNLOAD=0
BATCH_NAME="latest"

usage() {
    cat <<'EOF'
Usage: ./restore_heatmaps_from_mega.sh [options]

Downloads a MEGA heatmap batch, verifies it when manifests exist, and optionally
merges the restored heatmaps back into the live flat heatmap directory.

Options:
  --batch NAME                  Batch to restore. Default: latest
  --list                        List remote batches and exit.
  --activate-live               Merge restored PNGs into the live heatmap directory.
  --cleanup-download            Remove the local restored batch after verification.
  --cleanup-relay-download      Remove the relay host downloaded copy after rsync.
  --relay-host user@host        Use a remote host with MEGA CLI installed.
  --relay-download-dir PATH     Download cache directory on the relay host.
  --mega-root PATH              Source MEGA folder root.
  --mega-email EMAIL            MEGA login email if no active session exists.
  --mega-password PASSWORD      MEGA login password if no active session exists.
  --mega-auth-code CODE         MFA code for MEGA login when required.
  --yes                         Skip confirmation prompts.
  --dry-run                     Show the plan without changing files.
  --help                        Show this help text.
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --batch)
                BATCH_NAME="${2:?Missing value for --batch}"
                shift 2
                ;;
            --list)
                OPS_LIST_ONLY=1
                shift
                ;;
            --activate-live)
                OPS_ACTIVATE_LIVE=1
                shift
                ;;
            --cleanup-download)
                OPS_CLEANUP_LOCAL_DOWNLOAD=1
                shift
                ;;
            --cleanup-relay-download)
                OPS_CLEANUP_RELAY_DOWNLOAD=1
                shift
                ;;
            --relay-host)
                MEGA_RELAY_HOST="${2:?Missing value for --relay-host}"
                shift 2
                ;;
            --relay-download-dir)
                MEGA_RELAY_DOWNLOAD_ROOT="${2:?Missing value for --relay-download-dir}"
                shift 2
                ;;
            --mega-root)
                MEGA_REMOTE_ROOT="${2:?Missing value for --mega-root}"
                shift 2
                ;;
            --mega-email)
                MEGA_EMAIL="${2:?Missing value for --mega-email}"
                shift 2
                ;;
            --mega-password)
                MEGA_PASSWORD="${2:?Missing value for --mega-password}"
                shift 2
                ;;
            --mega-auth-code)
                MEGA_AUTH_CODE="${2:?Missing value for --mega-auth-code}"
                shift 2
                ;;
            --yes)
                OPS_ASSUME_YES=1
                shift
                ;;
            --dry-run)
                OPS_DRY_RUN=1
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                ops_fail "Unknown argument: $1"
                ;;
        esac
    done
}

resolve_latest_batch() {
    local latest
    latest="$(ops_mega_list_batch_dirs "$MEGA_REMOTE_ROOT" | awk -F/ 'NF{print $NF}' | sort | tail -n 1)"
    [[ -n "$latest" ]] || ops_fail "No MEGA backup batches found under $MEGA_REMOTE_ROOT"
    printf '%s\n' "$latest"
}

list_batches() {
    ops_mega_list_batch_dirs "$MEGA_REMOTE_ROOT" | awk -F/ 'NF{print $NF}' | sort
}

resolve_download_layout() {
    local download_dir="$1"

    if [[ -d "$download_dir/heatmap_snapshots" ]]; then
        RESTORE_BATCH_ROOT="$download_dir"
        RESTORE_HEATMAP_DIR="$download_dir/heatmap_snapshots"
        RESTORE_IS_STRUCTURED=1
        return
    fi

    if find "$download_dir" -maxdepth 1 -type f -name 'heatmap_*.png' | grep -q .; then
        RESTORE_BATCH_ROOT="$download_dir"
        RESTORE_HEATMAP_DIR="$download_dir"
        RESTORE_IS_STRUCTURED=0
        return
    fi

    ops_fail "Downloaded batch layout is not recognized: $download_dir"
}

verify_restore_batch() {
    local batch_root="$1"

    if [[ -f "$batch_root/manifests/sha256sums.txt" ]]; then
        ops_info "Verifying downloaded batch manifest"
        (
            cd "$batch_root"
            sha256sum -c "manifests/sha256sums.txt" >/dev/null
        )
        return
    fi

    ops_warn "No manifest found in this batch. Running lightweight PNG validation only."
    python3 - "$RESTORE_HEATMAP_DIR" <<'PY'
import os
import sys

root = os.path.abspath(sys.argv[1])
invalid = []

for name in sorted(os.listdir(root)):
    if not name.startswith("heatmap_") or not name.endswith(".png"):
        continue
    path = os.path.join(root, name)
    size = os.path.getsize(path)
    if size < 20:
        invalid.append((name, "too_small"))
        continue
    with open(path, "rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            invalid.append((name, "bad_signature"))
            continue
        handle.seek(-12, os.SEEK_END)
        tail = handle.read(12)
        if tail[4:8] != b"IEND":
            invalid.append((name, "missing_iend"))

if invalid:
    for name, reason in invalid:
        print(f"{name}\t{reason}", file=sys.stderr)
    sys.exit(1)
PY
}

activate_live_merge() {
    local restored_dir="$1"
    local live_dir
    local restart_needed=0

    live_dir="$(ops_repo_output_folder_abs)"
    mkdir -p "$live_dir"

    if ! ops_prompt_confirm "Merge restored PNGs into the live directory '$live_dir'?"; then
        ops_fail "Aborted by user"
    fi

    ops_stop_service_if_running
    restart_needed="${OPS_SERVICE_WAS_RUNNING:-0}"

    if ! rsync -a --ignore-existing "$restored_dir"/ "$live_dir"/; then
        if [[ "$restart_needed" == "1" ]]; then
            ops_restart_service_if_needed || true
        fi
        ops_fail "Restore merge failed"
    fi

    if [[ "$restart_needed" == "1" ]]; then
        ops_restart_service_if_needed
    fi
}

main() {
    parse_args "$@"

    ops_require_command awk
    ops_require_command find
    ops_require_command python3
    ops_require_command rsync
    ops_require_command sha256sum

    ops_select_mega_mode
    ops_ensure_mega_session

    if [[ "$OPS_LIST_ONLY" == "1" ]]; then
        list_batches
        exit 0
    fi

    if [[ "$BATCH_NAME" == "latest" ]]; then
        BATCH_NAME="$(resolve_latest_batch)"
    fi

    local remote_batch="$MEGA_REMOTE_ROOT/$BATCH_NAME"
    local local_restore_dir="$HEATMAP_RESTORE_STAGING_ROOT/$BATCH_NAME"
    local relay_download_path="${MEGA_RELAY_DOWNLOAD_ROOT%/}/$BATCH_NAME"

    if [[ "$OPS_DRY_RUN" == "1" ]]; then
        cat <<EOF
Dry run only. Planned actions:
  Mode: $MEGA_MODE
  Remote batch: $remote_batch
  Local restore dir: $local_restore_dir
  Activate live after restore: $OPS_ACTIVATE_LIVE
  Cleanup local download: $OPS_CLEANUP_LOCAL_DOWNLOAD
  Cleanup relay download: $OPS_CLEANUP_RELAY_DOWNLOAD
EOF
        exit 0
    fi

    [[ ! -e "$local_restore_dir" ]] || ops_fail "Local restore directory already exists: $local_restore_dir"
    mkdir -p "$HEATMAP_RESTORE_STAGING_ROOT"

    if [[ "$MEGA_MODE" == "local" ]]; then
        ops_info "Downloading batch directly from MEGA"
        ops_mega_get_dir "$remote_batch" "$HEATMAP_RESTORE_STAGING_ROOT"
    else
        ops_info "Downloading batch on relay host staging"
        ops_relay_exec rm -rf "$relay_download_path"
        ops_relay_exec mkdir -p "$MEGA_RELAY_DOWNLOAD_ROOT"
        ops_relay_exec mega-get "$remote_batch" "$MEGA_RELAY_DOWNLOAD_ROOT"
        ops_info "Copying restored batch from relay host to local staging"
        rsync -az "${MEGA_RELAY_HOST}:${relay_download_path}" "$HEATMAP_RESTORE_STAGING_ROOT/"
    fi

    resolve_download_layout "$local_restore_dir"
    verify_restore_batch "$RESTORE_BATCH_ROOT"

    ops_info "Restore download verified"
    ops_info "Local restore batch: $local_restore_dir"

    if [[ "$OPS_ACTIVATE_LIVE" == "1" ]]; then
        activate_live_merge "$RESTORE_HEATMAP_DIR"
        ops_info "Live heatmap directory updated by merge"
    fi

    if [[ "$OPS_CLEANUP_LOCAL_DOWNLOAD" == "1" ]]; then
        if ops_prompt_confirm "Remove the local restore batch '$local_restore_dir'?"; then
            rm -rf "$local_restore_dir"
            ops_info "Removed local restore batch"
        fi
    fi

    if [[ "$OPS_CLEANUP_RELAY_DOWNLOAD" == "1" && "$MEGA_MODE" == "relay" ]]; then
        if ops_prompt_confirm "Remove the relay restore batch '$relay_download_path'?"; then
            ops_relay_exec rm -rf "$relay_download_path"
            ops_info "Removed relay restore batch"
        fi
    fi
}

main "$@"
