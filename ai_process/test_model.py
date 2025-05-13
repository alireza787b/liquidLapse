#!/usr/bin/env python3
"""
test_model.py

Inference on raw heatmap snapshots using a trained CNN→LSTM model checkpoint
that embeds both weights and model configuration.

Usage:
    python test_model.py [options]

The checkpoint must include:
  - 'state_dict': model weights
  - 'config': dict with keys ['backbone','freeze','lstm_hidden','lstm_layers','reg_dropout','seq_len']

By default this script will:
  1. Auto-find the latest checkpoint under ai_process/<session>/train_*/best_model.pt
  2. Load the last `--frames` snapshots from heatmap_snapshots/ (default from checkpoint seq_len)
  3. Apply transforms (224×224, ToTensor, ImageNet norm)
  4. Reconstruct the exact model from checkpoint['config']
  5. Run one-shot inference and print the predicted target

Options:
  --base_dir   Base project directory (default: ~/liquidLapse)
  --session    Session name (default: test1)
  --model      Path to .pt checkpoint (auto-find if omitted)
  --end_file   Snapshot filename substring or timestamp to anchor sequence
  --frames     Number of frames (default: seq_len from checkpoint)
  --device     Torch device (cpu or cuda:0)
"""
import os
import sys
import re
import argparse
import torch
from datetime import datetime
from torchvision import transforms, models
from PIL import Image
from glob import glob

# =============================================================================
# Defaults (overridden via CLI)
# =============================================================================
DEFAULT_BASE_DIR   = os.path.expanduser("~/liquidLapse")
DEFAULT_SESSION    = "test1"
DEFAULT_SNAPSHOT_DIR = os.path.expanduser("~/liquidLapse/heatmap_snapshots")
DEFAULT_MODEL      = None    # auto-find
DEFAULT_END_FILE   = None
DEFAULT_FRAMES     = None    # from checkpoint
DEFAULT_DEVICE     = "cpu"

# =============================================================================
# Utility Functions
# =============================================================================
def parse_snapshot_ts(filename):
    """Extract datetime from a heatmap filename."""
    m = re.match(r"heatmap_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_", filename)
    if not m:
        raise ValueError(f"Cannot parse timestamp from '{filename}'")
    dt_str = m.group(1) + " " + m.group(2).replace('-',':')
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")


def list_snapshots(dirpath):
    """Return sorted list of full paths to snapshot images."""
    files = [f for f in os.listdir(dirpath) if f.startswith("heatmap_") and f.endswith(".png")]
    entries = []
    for fn in files:
        try:
            ts = parse_snapshot_ts(fn)
            entries.append((ts, os.path.join(dirpath, fn)))
        except:
            continue
    entries.sort(key=lambda x: x[0])
    return [p for _,p in entries]


def find_latest_checkpoint(session, base_dir):
    """Auto-detect most recent best_model.pt under ai_process/<session>/train_*."""
    search = os.path.join(base_dir, 'ai_process', session, 'train_*', 'best_model.pt')
    candidates = glob(search)
    if not candidates:
        print(f"[ERROR] No checkpoint found under {os.path.dirname(search)}", file=sys.stderr)
        sys.exit(1)
    latest = max(candidates, key=os.path.getmtime)
    print(f"[INFO] Auto-detected checkpoint: {latest}")
    return latest


def build_transforms():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

# =============================================================================
# Model Definition
# =============================================================================
class CNN_LSTM(torch.nn.Module):
    def __init__(self, backbone, hidden, layers, dropout, freeze):
        super().__init__()
        cnn = getattr(models, backbone)(weights="DEFAULT")
        self.features = torch.nn.Sequential(*list(cnn.children())[:-1])
        feat_dim = cnn.fc.in_features
        if freeze:
            for p in self.features.parameters(): p.requires_grad=False
        self.lstm = torch.nn.LSTM(feat_dim, hidden, layers, batch_first=True)
        head = [torch.nn.Linear(hidden, hidden//2), torch.nn.ReLU()]
        if dropout>0: head.append(torch.nn.Dropout(dropout))
        head.append(torch.nn.Linear(hidden//2,1))
        self.regressor = torch.nn.Sequential(*head)

    def forward(self, x):
        # x: [B,T,3,H,W]
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.features(x).view(B,T,-1)
        out,_ = self.lstm(feats)
        return self.regressor(out[:, -1, :]).squeeze(1)

# =============================================================================
# Main Inference
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Test CNN-LSTM model on heatmap snapshots.")
    p.add_argument('--base_dir', type=str, default=DEFAULT_BASE_DIR)
    p.add_argument('--session',  type=str, default=DEFAULT_SESSION)
    p.add_argument('--model',    type=str, help="Path to .pt checkpoint (auto-find if omitted)")
    p.add_argument('--end_file', type=str, default=DEFAULT_END_FILE)
    p.add_argument('--frames',   type=int, default=DEFAULT_FRAMES)
    p.add_argument('--device',   type=str, default=DEFAULT_DEVICE)
    args = p.parse_args()

    # Device selection
    device = torch.device(args.device if (args.device=='cpu' or torch.cuda.is_available()) else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Load checkpoint
    ckpt_path = args.model or find_latest_checkpoint(args.session, args.base_dir)
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']
    print(f"[INFO] Loaded config: {config}")

    # Determine number of frames
    num_frames = args.frames or config['seq_len']
    print(f"[INFO] Using {num_frames} frames for inference")

    # List and select snapshots
    snaps = list_snapshots(DEFAULT_SNAPSHOT_DIR)
    if not snaps:
        print(f"[ERROR] No snapshots in {DEFAULT_SNAPSHOT_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Found {len(snaps)} snapshots")

    # Determine end index
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

    start_idx = max(0, end_idx - num_frames + 1)
    seq_files = snaps[start_idx:end_idx+1]
    if len(seq_files) < num_frames:
        print(f"[WARN] Only {len(seq_files)} frames available, required {num_frames}")
    print(f"[INFO] Sequence indices {start_idx} to {end_idx} ({len(seq_files)} frames):")
    for fn in seq_files:
        print("   ", os.path.basename(fn))

    # Build input tensor
    tf = build_transforms()
    imgs = []
    for fn in seq_files:
        try:
            img = Image.open(fn).convert('RGB')
        except:
            img = Image.new('RGB', (224,224))
        imgs.append(tf(img))
    x = torch.stack(imgs).unsqueeze(0).to(device)  # [1,T,3,224,224]

    # Reconstruct model and load weights
    model = CNN_LSTM(
        config['backbone'],
        config['lstm_hidden'],
        config['lstm_layers'],
        config['reg_dropout'],
        config['freeze']
    )
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    print(f"[INFO] Model reconstructed and loaded")

    # Inference
    with torch.no_grad():
        pred = model(x).item()
    print(f"\n[RESULT] Predicted next-step target = {pred:.4f}\n")

if __name__ == '__main__':
    main()
