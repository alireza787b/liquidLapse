#!/usr/bin/env python3
"""
test_model.py

Inference on raw heatmap snapshots using a trained CNN→LSTM model checkpoint.

This script will:
  1. Auto-find the latest checkpoint under ai_process/<session>/train_*/best_model.pt
  2. Load the checkpoint (weights + config)
  3. Determine sequence length (CLI override or checkpoint['seq_len'])
  4. Load the last N frames from heatmap_snapshots/ (or anchor via --end_file)
  5. Apply transforms (224×224, ToTensor, ImageNet norm)
  6. Reconstruct the exact model from checkpoint['config']
  7. Run inference and print the predicted target

Usage:
    python test_model.py [options]

Options:
  --base_dir   Base directory (default: ~/liquidLapse)
  --session    Session name under ai_process (default: test1)
  --model      Path to .pt checkpoint (auto-find if omitted)
  --end_file   Snapshot filename substring or timestamp to anchor (default: latest)
  --frames     Number of frames to use (overrides checkpoint seq_len)
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
# Defaults (override via CLI)
# =============================================================================
DEFAULT_BASE_DIR     = os.path.expanduser("~/liquidLapse")
DEFAULT_SESSION      = "test1"
DEFAULT_SNAPSHOT_DIR = os.path.expanduser("~/liquidLapse/heatmap_snapshots")
DEFAULT_MODEL_PATH   = None
DEFAULT_END_FILE     = None
DEFAULT_FRAMES       = None
DEFAULT_DEVICE       = "cpu"

# =============================================================================
# Utility Functions
# =============================================================================
def parse_snapshot_ts(fn):
    m = re.match(r"heatmap_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_", fn)
    if not m:
        raise ValueError(f"Cannot parse timestamp from '{fn}'")
    dt = m.group(1) + " " + m.group(2).replace("-", ":")
    return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

def list_snapshots(dirpath):
    files = [f for f in os.listdir(dirpath) if f.startswith("heatmap_") and f.endswith(".png")]
    entries = []
    for fn in files:
        try:
            ts = parse_snapshot_ts(fn)
            entries.append((ts, os.path.join(dirpath, fn)))
        except:
            continue
    entries.sort(key=lambda x: x[0])
    return [p for _, p in entries]

def find_latest_checkpoint(session, base_dir):
    pattern = os.path.join(base_dir, 'ai_process', session, 'train_*', 'best_model.pt')
    candidates = glob(pattern)
    if not candidates:
        print(f"[ERROR] No checkpoint found under {os.path.dirname(pattern)}", file=sys.stderr)
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
    def __init__(self, backbone, lstm_hidden, lstm_layers, reg_dropout, freeze):
        super().__init__()
        cnn = getattr(models, backbone)(weights="DEFAULT")
        self.features = torch.nn.Sequential(*list(cnn.children())[:-1])
        feat_dim = cnn.fc.in_features
        if freeze:
            for p in self.features.parameters(): p.requires_grad=False
        self.lstm = torch.nn.LSTM(feat_dim, lstm_hidden, lstm_layers, batch_first=True)
        head = [torch.nn.Linear(lstm_hidden, lstm_hidden//2), torch.nn.ReLU()]
        if reg_dropout>0: head.append(torch.nn.Dropout(reg_dropout))
        head.append(torch.nn.Linear(lstm_hidden//2, 1))
        self.regressor = torch.nn.Sequential(*head)

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        feats, _ = self.lstm(self.features(x).view(B,T,-1))
        return self.regressor(feats[:,-1,:]).squeeze(1)

# =============================================================================
# Main Inference
# =============================================================================
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Test CNN-LSTM model on heatmap snapshots.")
    parser.add_argument('--base_dir', type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument('--session',  type=str, default=DEFAULT_SESSION)
    parser.add_argument('--model',    type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--end_file', type=str, default=DEFAULT_END_FILE)
    parser.add_argument('--frames',   type=int, default=DEFAULT_FRAMES)
    parser.add_argument('--device',   type=str, default=DEFAULT_DEVICE)
    args = parser.parse_args()

    # Device
    device = torch.device(args.device if (args.device=='cpu' or torch.cuda.is_available()) else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Checkpoint
    ckpt_path = args.model or find_latest_checkpoint(args.session, args.base_dir)
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get('config', {})
    state_dict = ckpt.get('state_dict', {})
    if not config or not state_dict:
        print("[ERROR] Invalid checkpoint (missing config/state_dict)", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Loaded checkpoint config: {config}")

    # Sequence length
    seq_len = args.frames or config.get('seq_len')
    print(f"[INFO] Sequence length: {seq_len}")

    # Snapshots
    snaps = list_snapshots(DEFAULT_SNAPSHOT_DIR)
    if not snaps:
        print(f"[ERROR] No snapshots in {DEFAULT_SNAPSHOT_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Found {len(snaps)} snapshots")

    # Anchor end index
    if args.end_file:
        matches = [i for i,f in enumerate(snaps) if args.end_file in os.path.basename(f)]
        if not matches:
            print(f"[ERROR] end_file '{args.end_file}' not found", file=sys.stderr)
            sys.exit(1)
        end_idx = matches[-1]
        print(f"[INFO] Anchored on: {os.path.basename(snaps[end_idx])}")
    else:
        end_idx = len(snaps)-1
        print(f"[INFO] Using latest snapshot: {os.path.basename(snaps[end_idx])}")

    start_idx = max(0, end_idx-seq_len+1)
    seq_files = snaps[start_idx:end_idx+1]
    if len(seq_files) < seq_len:
        print(f"[WARN] Only {len(seq_files)} frames available, required {seq_len}")
    print(f"[INFO] Frames {start_idx}→{end_idx} ({len(seq_files)}):")
    for fn in seq_files: print("  ", os.path.basename(fn))

    # Build tensor
    tfm = build_transforms()
    imgs = []
    for fn in seq_files:
        try:
            img = Image.open(fn).convert('RGB')
        except:
            img = Image.new('RGB', (224,224))
        imgs.append(tfm(img))
    x = torch.stack(imgs).unsqueeze(0).to(device)

    # Reconstruct model
    model = CNN_LSTM(
        config['backbone'],
        config['lstm_hidden'],
        config['lstm_layers'],
        config['reg_dropout'],
        config['freeze']
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print("[INFO] Model ready for inference")

    # Predict
    with torch.no_grad():
        pred = model(x).item()
    print(f"\n[RESULT] Predicted next-step target = {pred:.4f}\n")
