#!/usr/bin/env python3
"""
test_model.py

Inference on raw heatmap snapshots using a trained CNN→LSTM model checkpoint.

This script will:
  1. Auto-find the latest checkpoint under ai_process/<session>/train_*/best_model.pt
  2. Load the checkpoint (weights + config)
  3. Determine sequence length (from CLI or checkpoint)
  4. Load the last N frames from heatmap_snapshots/ (or anchor via --end_file)
  5. Apply transforms (224×224, ToTensor, ImageNet norm)
  6. Reconstruct the exact CNN→LSTM model from checkpoint['config']
  7. Run one-shot inference and print the predicted target

Usage:
    python test_model.py [options]

Options:
  --base_dir   Base directory (default: ~/liquidLapse)
  --session    Session name under ai_process (default: test1)
  --model      Path to .pt checkpoint (auto-find if omitted)
  --end_file   Snapshot filename substring or timestamp to anchor (default: latest)
  --frames     Number of past frames to use (default: seq_len from checkpoint)
  --device     Torch device, e.g. cpu or cuda:0 (default: cpu)
"""
import os
import sys
import re
import argparse
from glob import glob
from datetime import datetime

import torch
from torchvision import models, transforms
from PIL import Image

# =============================================================================
# Defaults (can override via CLI)
# =============================================================================
DEFAULT_BASE_DIR     = os.path.expanduser("~/liquidLapse")
DEFAULT_SESSION      = "test1"
DEFAULT_SNAPSHOT_DIR = os.path.expanduser("~/liquidLapse/heatmap_snapshots")
DEFAULT_MODEL_PATH   = None   # auto-find if omitted
DEFAULT_END_FILE     = None   # latest if omitted
DEFAULT_FRAMES       = None   # will use checkpoint['config']['seq_len']
DEFAULT_DEVICE       = "cpu"

# =============================================================================
# Utilities
# =============================================================================

def parse_snapshot_ts(fn):
    """Extract datetime from filename: heatmap_YYYY-MM-DD_HH-MM-SS_... .png"""
    m = re.match(r"heatmap_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_", fn)
    if not m:
        raise ValueError(f"Cannot parse timestamp from '{fn}'")
    dt_str = m.group(1) + " " + m.group(2).replace('-', ':')
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")


def list_snapshots(dirpath):
    """Return sorted list of full paths to valid heatmap images."""
    files = [f for f in os.listdir(dirpath) if f.startswith("heatmap_") and f.endswith(".png")]
    entries = []
    for fn in files:
        try:
            ts = parse_snapshot_ts(fn)
            entries.append((ts, os.path.join(dirpath, fn)))
        except Exception:
            continue
    entries.sort(key=lambda x: x[0])
    return [p for _, p in entries]


def find_latest_checkpoint(session, base_dir):
    """Find newest best_model.pt under ai_process/<session>/train_*/."""
    pattern = os.path.join(base_dir, 'ai_process', session, 'train_*', 'best_model.pt')
    candidates = glob(pattern)
    if not candidates:
        print(f"[ERROR] No checkpoint found under {os.path.dirname(pattern)}", file=sys.stderr)
        sys.exit(1)
    latest = max(candidates, key=os.path.getmtime)
    print(f"[INFO] Auto-detected checkpoint: {latest}")
    return latest


def build_transforms():
    """Data transforms matching training preprocessing."""
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

# =============================================================================
# Model Definition
# =============================================================================
class CNN_LSTM(torch.nn.Module):
    def __init__(self, backbone, lstm_hidden, lstm_layers, reg_dropout, freeze):
        super().__init__()
        cnn = getattr(models, backbone)(weights="DEFAULT")
        # feature extractor (all but final FC)
        self.features = torch.nn.Sequential(*list(cnn.children())[:-1])
        feat_dim = cnn.fc.in_features
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
        self.lstm = torch.nn.LSTM(feat_dim, lstm_hidden, lstm_layers, batch_first=True)
        head = [torch.nn.Linear(lstm_hidden, lstm_hidden//2), torch.nn.ReLU()]
        if reg_dropout > 0:
            head.append(torch.nn.Dropout(reg_dropout))
        head.append(torch.nn.Linear(lstm_hidden//2, 1))
        self.regressor = torch.nn.Sequential(*head)

    def forward(self, x):  # x: [B,T,3,224,224]
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.features(x).view(B, T, -1)
        out, _ = self.lstm(feats)
        return self.regressor(out[:, -1, :]).squeeze(1)

# =============================================================================
# Main Inference
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test CNN-LSTM model on raw heatmap snapshots.")
    parser.add_argument('--base_dir', type=str,   default=DEFAULT_BASE_DIR,
                        help="Base project directory")
    parser.add_argument('--session',  type=str,   default=DEFAULT_SESSION,
                        help="Session name under ai_process")
    parser.add_argument('--model',    type=str,   default=DEFAULT_MODEL_PATH,
                        help="Path to checkpoint .pt file")
    parser.add_argument('--end_file', type=str,   default=DEFAULT_END_FILE,
                        help="Filename substring or timestamp to anchor")
    parser.add_argument('--frames',   type=int,   default=DEFAULT_FRAMES,
                        help="Number of frames (overrides checkpoint seq_len)")
    parser.add_argument('--device',   type=str,   default=DEFAULT_DEVICE,
                        help="Torch device: cpu or cuda:0")
    args = parser.parse_args()

    # Select device
    device = torch.device(args.device if (args.device=='cpu' or torch.cuda.is_available()) else 'cpu')
    print(f"[INFO] Device: {device}")

    # Load checkpoint
    ckpt_path = args.model or find_latest_checkpoint(args.session, args.base_dir)
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get('config', {})
    state_dict = ckpt.get('state_dict', {})
    if not config or not state_dict:
        print("[ERROR] Checkpoint missing 'config' or 'state_dict'.", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Loaded config: {config}")

    # Determine number of frames
    num_frames = args.frames or config.get('seq_len')
    print(f"[INFO] Using {num_frames} frames for inference")

    # List snapshots
    snaps = list_snapshots(DEFAULT_SNAPSHOT_DIR)
    if not snaps:
        print(f"[ERROR] No snapshots in {DEFAULT_SNAPSHOT_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Found {len(snaps)} snapshots")

    # Choose end index
    if args.end_file:
        matches = [i for i,f in enumerate(snaps) if args.end_file in os.path.basename(f)]
        if not matches:
            print(f"[ERROR] end_file '{args.end_file}' not found", file=sys.stderr)
            sys.exit(1)
        end_idx = matches[-1]
        print(f"[INFO] Anchored on snapshot: {os.path.basename(snaps[end_idx])}")
    else:
        end_idx = len(snaps) - 1
        print(f"[INFO] Using latest snapshot: {os.path.basename(snaps[end_idx])}")

    # Gather sequence files
    start_idx = max(0, end_idx - num_frames + 1)
    seq_files = snaps[start_idx:end_idx+1]
    if len(seq_files) < num_frames:
        print(f"[WARN] Only {len(seq_files)} frames available, required {num_frames}")
    print(f"[INFO] Sequence frames [{start_idx}→{end_idx}] ({len(seq_files)} frames):")
    for fn in seq_files:
        print("   ", os.path.basename(fn))

    # Build input tensor
    tfm = build_transforms()
    images = []
    for fn in seq_files:
        try:
            img = Image.open(fn).convert('RGB')
        except:
            img = Image.new('RGB', (224,224))
        images.append(tfm(img))
    x = torch.stack(images).unsqueeze(0).to(device)

    # Reconstruct and load model
    model = CNN_LSTM(
        config['backbone'],
        config['lstm_hidden'],
        config['lstm_layers'],
        config['reg_dropout'],
        config['freeze']
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print("[INFO] Model reconstructed and loaded")

    # Inference
    with torch.no_grad():
        pred = model(x).item()
    print(f"\n[RESULT] Predicted next-step target = {pred:.4f}\n")
