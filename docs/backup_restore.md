# Heatmap Backup and Restore

`liquidLapse` keeps the live capture path simple on purpose: `heatmap_snapshots/` stays a flat directory of `heatmap_*.png` files. The backup system does not change that runtime layout. It wraps the live folder in a safer archival workflow with batch folders, manifests, metadata snapshots, and MEGA upload / restore verification.

## User-Facing Scripts

- `./backup_heatmaps_to_mega.sh`
- `./restore_heatmaps_from_mega.sh`

`sync_heatmaps.sh` is still the legacy rsync pull helper for a different workflow. The MEGA backup / restore flow is handled by the two scripts above.

## Backup Design

Every backup produces a structured batch under `backup_staging/<batch_name>/` before upload:

```text
backup_staging/<batch_name>/
├── heatmap_snapshots/         # Frozen raw PNGs
├── metadata/
│   ├── config.yaml
│   ├── backup_context.txt
│   ├── git_status.txt
│   ├── liquidLapseService.status      # when present
│   ├── service_health.tail.log        # when present
│   └── ai_process/...*.json           # optional JSON metadata snapshot
└── manifests/
    ├── sha256sums.txt
    ├── summary.json
    ├── png_validation.txt
    └── duplicate_hashes.txt
```

The active live folder is immediately recreated after freeze so capture can continue. The archive structure is for backup only; restore merges PNGs back into the flat live directory.

## Service Continuity

`./service.sh start` starts the capture loop in the background. `./service.sh run` runs the same loop in the foreground and is the preferred entry point for `systemd`, containers, or `tmux`.

For production Linux hosts, install `deploy/liquidlapse-capture.service` and let systemd own the foreground loop:

```bash
sudo install -m 0644 deploy/liquidlapse-capture.service /etc/systemd/system/liquidlapse-capture.service
sudo systemctl daemon-reload
sudo systemctl enable --now liquidlapse-capture.service
```

`tmux` is acceptable for manual maintenance, but it only survives SSH disconnects while the tmux server remains alive. It does not survive a killed tmux session or host reboot.

The backup script stops the service only around the freeze window when it finds a running PID. It restarts the service after the live folder has been recreated, so new captures continue in a fresh flat `heatmap_snapshots/` directory while the frozen batch uploads.

## MEGA Transport Modes

The scripts support two MEGA modes:

1. Local MEGA CLI
   Use this when `mega-whoami`, `mega-put`, `mega-get`, and `mega-ls` are installed on the capture host.
2. SSH relay host
   Use `--relay-host user@host` when the capture host is space-constrained or MEGA CLI is installed elsewhere.

The scripts prefer an existing MEGA session. If no session exists, they can log in with:

- `--mega-email`
- `--mega-password`
- `--mega-auth-code` for MFA when needed

Session reuse is the preferred mode for routine runs.

The MEGA root folder may already exist. Backup creation is idempotent at the root level and verifies that the target folder exists before uploading a batch.

## Backup Usage

Dry-run first:

```bash
./backup_heatmaps_to_mega.sh \
  --relay-host root@204.168.181.45 \
  --mega-root /Root/liquidLapse_backups/heatmap_snapshots_batches \
  --dry-run
```

Real backup:

```bash
./backup_heatmaps_to_mega.sh \
  --relay-host root@204.168.181.45 \
  --mega-root /Root/liquidLapse_backups/heatmap_snapshots_batches
```

Useful options:

- `--batch-name NAME` for a controlled batch label
- `--no-ai-metadata` to skip the `ai_process` JSON snapshot
- `--delete-local-after-upload` to remove the local staged batch only after MEGA count verification succeeds
- `--delete-relay-after-upload` to clear relay staging after a successful upload
- `--yes` for non-interactive automation

For regular space-saving maintenance on the current relay setup:

```bash
./backup_heatmaps_to_mega.sh \
  --relay-host root@204.168.181.45 \
  --mega-root /Root/liquidLapse_backups/heatmap_snapshots_batches \
  --delete-local-after-upload \
  --delete-relay-after-upload \
  --yes
```

This freezes the current live folder, uploads and verifies the structured batch, then removes local and relay staging after the remote file count matches.

## Restore Usage

List batches:

```bash
./restore_heatmaps_from_mega.sh \
  --relay-host root@204.168.181.45 \
  --mega-root /Root/liquidLapse_backups/heatmap_snapshots_batches \
  --list
```

Dry-run restore of the latest batch:

```bash
./restore_heatmaps_from_mega.sh \
  --relay-host root@204.168.181.45 \
  --mega-root /Root/liquidLapse_backups/heatmap_snapshots_batches \
  --batch latest \
  --dry-run
```

Download and verify a specific batch without touching live data:

```bash
./restore_heatmaps_from_mega.sh \
  --relay-host root@204.168.181.45 \
  --mega-root /Root/liquidLapse_backups/heatmap_snapshots_batches \
  --batch heatmap_batch_20260411T132841Z_linode
```

Restore and merge into the live flat folder:

```bash
./restore_heatmaps_from_mega.sh \
  --relay-host root@204.168.181.45 \
  --mega-root /Root/liquidLapse_backups/heatmap_snapshots_batches \
  --batch latest \
  --activate-live
```

Useful options:

- `--cleanup-download` to remove the local restored batch after verification
- `--cleanup-relay-download` to remove the relay host downloaded copy after it is copied back locally
- `--yes` for non-interactive automation

## Safety Guarantees

- The live `heatmap_snapshots/` layout is not migrated or nested.
- Backup freezes the current folder before upload instead of copying in-flight files.
- The service is stopped only for the freeze / merge window when it is already running.
- The scripts generate and verify SHA-256 manifests for structured batches.
- PNG validation is recorded in `png_validation.txt`.
- Duplicate content is reported in `duplicate_hashes.txt` for operator review, but duplicates do not fail the backup by themselves.
- Deletion is always optional and happens only after successful upload / download verification.

## Notes

- Older manually uploaded batches that only contain flat PNG files can still be restored. They will not have the full manifest set, so restore falls back to lightweight PNG validation.
- The new archive structure is intentionally separate from runtime consumption. Downstream consumers still read the same flat live snapshot directory they already expect.
