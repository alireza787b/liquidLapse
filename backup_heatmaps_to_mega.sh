#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/heatmap_backup_common.sh"

MEGA_REMOTE_ROOT="$HEATMAP_DEFAULT_MEGA_ROOT"
MEGA_RELAY_STAGING_DIR="$HEATMAP_DEFAULT_RELAY_STAGING"
MEGA_RELAY_HOST="${MEGA_RELAY_HOST:-}"
MEGA_EMAIL="${MEGA_EMAIL:-}"
MEGA_PASSWORD="${MEGA_PASSWORD:-}"
MEGA_AUTH_CODE="${MEGA_AUTH_CODE:-}"
OPS_ASSUME_YES=0
OPS_DRY_RUN=0
OPS_INCLUDE_AI_METADATA=1
OPS_DELETE_LOCAL_AFTER_UPLOAD=0
OPS_DELETE_RELAY_AFTER_UPLOAD=0
BATCH_NAME=""

usage() {
    cat <<'EOF'
Usage: ./backup_heatmaps_to_mega.sh [options]

Creates a verified heatmap backup batch, uploads it to MEGA, and optionally
cleans up the local/relay staging copy after verification.

Options:
  --relay-host user@host         Use a remote host with MEGA CLI installed.
  --relay-staging-dir PATH       Staging directory on the relay host.
  --mega-root PATH               Destination MEGA folder root.
  --batch-name NAME              Override the generated batch name.
  --no-ai-metadata               Skip ai_process JSON metadata snapshot.
  --delete-local-after-upload    Delete the local staged batch after verification.
  --delete-relay-after-upload    Delete the relay staged batch after verification.
  --mega-email EMAIL             MEGA login email if no active session exists.
  --mega-password PASSWORD       MEGA login password if no active session exists.
  --mega-auth-code CODE          MFA code for MEGA login when required.
  --yes                          Skip confirmation prompts.
  --dry-run                      Show the plan without changing files.
  --help                         Show this help text.
EOF
}

backup_cleanup() {
    local exit_code=$?

    if [[ "$exit_code" -ne 0 && "${OPS_SERVICE_RESTART_PENDING:-0}" == "1" ]]; then
        ops_warn "Backup exited early. Attempting to restart liquidLapse service."
        if ! ops_restart_service_if_needed; then
            ops_warn "Automatic service restart failed. Please run ./service.sh start manually."
        fi
    fi

    exit "$exit_code"
}

trap backup_cleanup EXIT

copy_backup_metadata() {
    local batch_dir="$1"
    local note="$2"
    local snapshot_rel="$3"
    local snapshot_count="$4"
    local metadata_dir="$batch_dir/metadata"

    mkdir -p "$metadata_dir"
    cp "$HEATMAP_CONFIG_FILE" "$metadata_dir/config.yaml"

    if [[ -f "$HEATMAP_BASE_DIR/liquidLapseService.status" ]]; then
        cp "$HEATMAP_BASE_DIR/liquidLapseService.status" "$metadata_dir/liquidLapseService.status"
    fi

    if [[ -f "$HEATMAP_BASE_DIR/service_health.log" ]]; then
        tail -n 200 "$HEATMAP_BASE_DIR/service_health.log" > "$metadata_dir/service_health.tail.log"
    fi

    if [[ "$OPS_INCLUDE_AI_METADATA" == "1" ]]; then
        ops_copy_ai_metadata "$metadata_dir"
    fi

    git -C "$HEATMAP_BASE_DIR" status --short --branch > "$metadata_dir/git_status.txt" 2>/dev/null || true
    ops_write_backup_context "$metadata_dir" "$BATCH_NAME" "$note" "$snapshot_rel" "$snapshot_count"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --relay-host)
                MEGA_RELAY_HOST="${2:?Missing value for --relay-host}"
                shift 2
                ;;
            --relay-staging-dir)
                MEGA_RELAY_STAGING_DIR="${2:?Missing value for --relay-staging-dir}"
                shift 2
                ;;
            --mega-root)
                MEGA_REMOTE_ROOT="${2:?Missing value for --mega-root}"
                shift 2
                ;;
            --batch-name)
                BATCH_NAME="${2:?Missing value for --batch-name}"
                shift 2
                ;;
            --no-ai-metadata)
                OPS_INCLUDE_AI_METADATA=0
                shift
                ;;
            --delete-local-after-upload)
                OPS_DELETE_LOCAL_AFTER_UPLOAD=1
                shift
                ;;
            --delete-relay-after-upload)
                OPS_DELETE_RELAY_AFTER_UPLOAD=1
                shift
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

main() {
    parse_args "$@"

    ops_require_command awk
    ops_require_command find
    ops_require_command git
    ops_require_command python3
    ops_require_command rsync
    ops_require_command sha256sum

    local snapshot_dir
    local snapshot_rel
    local batch_dir
    local heatmap_dir
    local snapshot_count
    local note
    local local_file_count
    local remote_file_count
    local relay_stage_path=""

    snapshot_dir="$(ops_repo_output_folder_abs)"
    snapshot_rel="$(ops_repo_output_folder_rel)"
    heatmap_dir="$snapshot_dir"

    [[ -d "$heatmap_dir" ]] || ops_fail "Snapshot directory does not exist: $heatmap_dir"

    BATCH_NAME="${BATCH_NAME:-$(ops_default_batch_name)}"
    batch_dir="$HEATMAP_BACKUP_STAGING_ROOT/$BATCH_NAME"
    [[ ! -e "$batch_dir" ]] || ops_fail "Batch directory already exists: $batch_dir"

    snapshot_count="$(ops_stage_heatmap_count "$heatmap_dir")"
    note="Frozen from $snapshot_rel before MEGA backup"

    ops_select_mega_mode

    if [[ "$OPS_DRY_RUN" == "1" ]]; then
        cat <<EOF
Dry run only. Planned actions:
  Mode: $MEGA_MODE
  Snapshot source: $heatmap_dir
  Snapshot count: $snapshot_count
  Local batch dir: $batch_dir
  MEGA root: $MEGA_REMOTE_ROOT
  Relay host: ${MEGA_RELAY_HOST:-none}
  Include AI metadata: $OPS_INCLUDE_AI_METADATA
  Delete local staged batch after upload: $OPS_DELETE_LOCAL_AFTER_UPLOAD
  Delete relay staged batch after upload: $OPS_DELETE_RELAY_AFTER_UPLOAD
EOF
        exit 0
    fi

    if ! ops_prompt_confirm "Freeze the current heatmap directory into batch '$BATCH_NAME' and continue?"; then
        ops_fail "Aborted by user"
    fi

    mkdir -p "$HEATMAP_BACKUP_STAGING_ROOT"
    ops_stop_service_if_running
    OPS_SERVICE_RESTART_PENDING=1

    mkdir -p "$batch_dir"
    mv "$heatmap_dir" "$batch_dir/heatmap_snapshots"
    mkdir -p "$heatmap_dir"

    copy_backup_metadata "$batch_dir" "$note" "$snapshot_rel" "$snapshot_count"
    ops_generate_manifests "$batch_dir" "$note"
    ops_verify_local_manifest "$batch_dir"

    ops_restart_service_if_needed
    OPS_SERVICE_RESTART_PENDING=0

    ops_ensure_mega_session
    ops_mega_mkdir "$MEGA_REMOTE_ROOT"

    if [[ "$MEGA_MODE" == "relay" ]]; then
        relay_stage_path="${MEGA_RELAY_STAGING_DIR%/}/$BATCH_NAME"
        ops_info "Copying batch to relay host staging: $relay_stage_path"
        rsync -az "$batch_dir" "${MEGA_RELAY_HOST}:${MEGA_RELAY_STAGING_DIR%/}/"
        ops_info "Uploading batch from relay host to MEGA"
        ops_mega_put_dir "$relay_stage_path" "$MEGA_REMOTE_ROOT"
    else
        ops_info "Uploading batch directly to MEGA"
        ops_mega_put_dir "$batch_dir" "$MEGA_REMOTE_ROOT"
    fi

    local_file_count="$(ops_stage_file_count "$batch_dir")"
    remote_file_count="$(ops_mega_file_count "$MEGA_REMOTE_ROOT/$BATCH_NAME")"
    if [[ "$local_file_count" != "$remote_file_count" ]]; then
        ops_fail "Remote file count mismatch after upload (local=$local_file_count remote=$remote_file_count)"
    fi

    ops_info "Backup uploaded and verified"
    ops_info "Remote batch: $MEGA_REMOTE_ROOT/$BATCH_NAME"
    ops_info "Local staged batch: $batch_dir"

    if [[ "$OPS_DELETE_LOCAL_AFTER_UPLOAD" == "1" ]]; then
        if ops_prompt_confirm "Delete the local staged batch '$batch_dir' now that MEGA verification succeeded?"; then
            rm -rf "$batch_dir"
            ops_info "Removed local staged batch"
        fi
    fi

    if [[ "$OPS_DELETE_RELAY_AFTER_UPLOAD" == "1" && "$MEGA_MODE" == "relay" ]]; then
        if ops_prompt_confirm "Delete the relay staged batch '$relay_stage_path' after the successful upload?"; then
            ops_relay_exec rm -rf "$relay_stage_path"
            ops_info "Removed relay staged batch"
        fi
    fi
}

main "$@"
