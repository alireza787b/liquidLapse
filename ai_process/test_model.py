#!/usr/bin/env python3
"""
test_model_raw.py

Inference on raw heatmap snapshots using a trained CNN→LSTM model.

Defaults can be configured at the top; CLI args override them.
"""
import os, re, glob, argparse, sys, torch
from datetime import datetime
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# ╭─────────────── DEFAULT CONFIG (edit here) ───────────────╮
SESSION_NAME      = "test1"   # ai_process/<SESSION_NAME>
HEATMAP_DIR       = os.path.expanduser("~/liquidLapse/heatmap_snapshots")
DEFAULT_MODEL_GLOB= f"ai_process/{SESSION_NAME}/train_*/best_model.pt"
DEFAULT_END_FILE  = None      # e.g. "heatmap_2025-04-16_06-04-00" or timestamp substring
DEFAULT_FRAMES    = 10        # how many past snapshots to use
DEFAULT_DEVICE    = "cpu"     # or "cuda:0"
# ──────────────────────────────────────────────────────────╯

class CNN_LSTM(nn.Module):
    def __init__(self, backbone, hidden, layers, dropout, freeze=True):
        super().__init__()
        cnn = getattr(models, backbone)(weights="DEFAULT")
        self.features = nn.Sequential(*list(cnn.children())[:-1])
        feat_dim = cnn.fc.in_features
        if freeze:
            for p in self.features.parameters(): p.requires_grad=False
        self.lstm = nn.LSTM(feat_dim, hidden, layers, batch_first=True)
        head = [nn.Linear(hidden,64), nn.ReLU()]
        if dropout>0: head.append(nn.Dropout(dropout))
        head.append(nn.Linear(64,1))
        self.regressor = nn.Sequential(*head)

    def forward(self, x):
        B,T,C,H,W = x.shape
        f = self.features(x.view(B*T,C,H,W)).view(B,T,-1)
        out,_ = self.lstm(f)
        return self.regressor(out[:,-1]).squeeze(1)

def parse_timestamp(fname):
    m = re.search(r"heatmap_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_", fname)
    if not m:
        raise ValueError(f"Cannot parse timestamp in '{fname}'")
    dt = f"{m.group(1)} {m.group(2).replace('-',':')}"
    return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

def find_latest_model(glob_pat):
    candidates = glob.glob(glob_pat)
    if not candidates:
        print(f"[ERROR] No model found matching '{glob_pat}'", file=sys.stderr)
        sys.exit(1)
    latest = sorted(candidates)[-1]
    print(f"[INFO] Auto-detected latest model: {latest}")
    return latest

def list_snapshots(directory):
    files = glob.glob(os.path.join(directory, "heatmap_*.png"))
    return sorted(files, key=lambda f: parse_timestamp(os.path.basename(f)))

def build_transforms():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def main():
    p = argparse.ArgumentParser(description="Infer with CNN→LSTM on raw heatmaps.")
    p.add_argument("--session",  default=SESSION_NAME)
    p.add_argument("--model",    help="Path or glob to best_model.pt")
    p.add_argument("--end-file", default=DEFAULT_END_FILE,
                   help="Filename or timestamp substring for final snapshot")
    p.add_argument("--frames",   type=int, default=DEFAULT_FRAMES,
                   help="Number of past snapshots to use")
    p.add_argument("--device",   default=DEFAULT_DEVICE)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--hidden",   type=int, default=128)
    p.add_argument("--layers",   type=int, default=1)
    p.add_argument("--dropout",  type=float, default=0.25)
    p.add_argument("--freeze",   action="store_true", help="Freeze CNN weights")
    args = p.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu"
                          else "cpu")
    print(f"[INFO] Device: {device}")

    # Model
    model_path = args.model or find_latest_model(
        f"ai_process/{args.session}/train_*/best_model.pt"
    )
    # Snapshots
    snaps = list_snapshots(HEATMAP_DIR)
    if not snaps:
        print(f"[ERROR] No heatmap files found in {HEATMAP_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Found {len(snaps)} snapshots")

    # Determine end index
    if args.end_file:
        idxs = [i for i,f in enumerate(snaps)
                if args.end_file in os.path.basename(f)
                or args.end_file in parse_timestamp(os.path.basename(f)).isoformat()]
        if not idxs:
            print(f"[ERROR] end-file '{args.end_file}' not found", file=sys.stderr)
            sys.exit(1)
        end_idx = idxs[-1]
        print(f"[INFO] Anchored end-file at index {end_idx}: {os.path.basename(snaps[end_idx])}")
    else:
        end_idx = len(snaps)-1
        print(f"[INFO] Using latest snapshot as end-file: {os.path.basename(snaps[end_idx])}")

    # Select sequence
    start_idx = max(0, end_idx - args.frames + 1)
    seq_files = snaps[start_idx:end_idx+1]
    if len(seq_files) < args.frames:
        msg = f"Only {len(seq_files)} frames available (requested {args.frames})"
        if args.end_file:
            print(f"[ERROR] {msg}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"[WARN] {msg}, proceeding with {len(seq_files)} frames")

    print(f"[INFO] Sequence indices [{start_idx}→{end_idx}] ({len(seq_files)} frames):")
    for f in seq_files: print("  ", os.path.basename(f))

    # Load images
    tf = build_transforms()
    imgs = torch.stack([ tf(Image.open(fp).convert("RGB")) for fp in seq_files ])
    x = imgs.unsqueeze(0).to(device)  # [1,T,3,224,224]

    # Build & load model
    model = CNN_LSTM(args.backbone, args.hidden, args.layers, args.dropout, freeze=args.freeze)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[INFO] Model loaded (backbone={args.backbone}, hidden={args.hidden}, layers={args.layers})")

    # Inference
    with torch.no_grad():
        pred = model(x).item()
    print(f"\n[RESULT] Predicted next {TARGET_FIELD} = {pred:.3f}%\n")

if __name__=="__main__":
    main()
