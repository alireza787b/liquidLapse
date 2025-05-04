#!/usr/bin/env python3
"""
test_model.py

Inference on raw heatmap snapshots using your trained CNN→LSTM model.

By default, this script will:
  • Auto–discover the most recent `best_model.pt` anywhere under:
        ai_process/<SESSION_NAME>/train_*/...
  • Auto–select the most recent heatmap in:
        heatmap_snapshots/
  • Grab the last N_FRAMES snapshots (including that “end” frame)
  • Apply the same transforms used in training (224×224, ToTensor, ImageNet norm)
  • Run a single-batch inference and print the predicted next-step percentage change

You can override defaults either by editing the DEFAULT_* constants below
or by passing command-line arguments (which take priority).

Usage examples:
  # 1) Use all defaults (latest model & end snapshot):
  python ai_process/test_model.py

  # 2) Specify a custom model path:
  python ai_process/test_model.py --model /path/to/best_model.pt

  # 3) Anchor on a specific snapshot name:
  python ai_process/test_model.py --end-file "2025-04-16_06-04-00" --frames 8

  # 4) Use GPU:
  python ai_process/test_model.py --device cuda:0

Arguments:
  --session     Name of AI processing session (default: test1)
  --model       Path to a `.pt` model file (default: auto–find)
  --end-file    Filename substring or timestamp to pick final snapshot
  --frames      Number of past snapshots to use (default: 10)
  --device      torch device (e.g. "cpu" or "cuda:0")
  --backbone    CNN backbone name from torchvision.models (default: resnet18)
  --hidden      LSTM hidden‐state size (default: 128)
  --layers      Number of LSTM layers (default: 1)
  --dropout     Regressor head dropout (default: 0.25)
  --freeze      Freeze CNN backbone weights (default: off)

Ensure you have run `train_model.py` successfully so that
`ai_process/<session>/train_*/*/best_model.pt` exists.
"""

import os, re, sys, argparse, torch
from datetime import datetime
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# ╭─────────────── DEFAULT CONFIG (can override here) ───────────────╮
SESSION_NAME    = "test1"
HEATMAP_DIR     = os.path.expanduser("~/liquidLapse/heatmap_snapshots")
DEFAULT_FRAMES  = 10
DEFAULT_DEVICE  = "cpu"
DEFAULT_BACKBONE= "resnet18"
DEFAULT_HIDDEN  = 128
DEFAULT_LAYERS  = 1
DEFAULT_DROPOUT = 0.25
# ──────────────────────────────────────────────────────────────────╯

class CNN_LSTM(nn.Module):
    def __init__(self, backbone, hidden, layers, dropout, freeze=True):
        super().__init__()
        cnn = getattr(models, backbone)(weights="DEFAULT")
        self.features = nn.Sequential(*list(cnn.children())[:-1])
        feat_dim = cnn.fc.in_features
        if freeze:
            for p in self.features.parameters(): p.requires_grad = False
        self.lstm = nn.LSTM(feat_dim, hidden, layers, batch_first=True)
        head = [nn.Linear(hidden, 64), nn.ReLU()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(64, 1))
        self.regressor = nn.Sequential(*head)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        f = self.features(x).view(B, T, -1)
        out, _ = self.lstm(f)
        return self.regressor(out[:, -1]).squeeze(1)

def parse_timestamp(fn):
    m = re.search(r"heatmap_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_", fn)
    if not m:
        raise ValueError(f"Can't parse timestamp from '{fn}'")
    dt = f"{m.group(1)} {m.group(2).replace('-',':')}"
    return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

def find_latest_model(root):
    """Recursively find the newest best_model.pt under ai_process/<session>."""
    candidates = []
    base = os.path.join(os.path.expanduser("~/liquidLapse"), "ai_process", SESSION_NAME)
    for dirpath, _, files in os.walk(base):
        if "best_model.pt" in files:
            candidates.append(os.path.join(dirpath, "best_model.pt"))
    if not candidates:
        print(f"[ERROR] No best_model.pt found under {base}", file=sys.stderr)
        sys.exit(1)
    # pick by modification time
    latest = max(candidates, key=os.path.getmtime)
    print(f"[INFO] Auto-detected model: {latest}")
    return latest

def list_snapshots(dirpath):
    imgs = [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.startswith("heatmap_") and f.endswith(".png")]
    imgs.sort(key=lambda p: parse_timestamp(os.path.basename(p)))
    return imgs

def build_transforms():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session",  default=SESSION_NAME)
    p.add_argument("--model",    help="Path to a best_model.pt (auto if omitted)")
    p.add_argument("--end-file", help="Snapshot filename or timestamp substring")
    p.add_argument("--frames",   type=int, default=DEFAULT_FRAMES)
    p.add_argument("--device",   default=DEFAULT_DEVICE)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--hidden",   type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--layers",   type=int, default=DEFAULT_LAYERS)
    p.add_argument("--dropout",  type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--freeze",   action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    print(f"[INFO] Device: {device}")

    # model
    model_path = args.model or find_latest_model(args.session)
    print(f"[INFO] Using model: {model_path}")

    # snapshots
    snaps = list_snapshots(HEATMAP_DIR)
    if not snaps:
        print(f"[ERROR] No heatmaps in {HEATMAP_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] {len(snaps)} snapshots found")

    # choose end index
    if args.end_file:
        matches = [i for i,f in enumerate(snaps)
                   if args.end_file in os.path.basename(f)
                   or args.end_file in parse_timestamp(os.path.basename(f)).isoformat()]
        if not matches:
            print(f"[ERROR] end-file '{args.end_file}' not found", file=sys.stderr)
            sys.exit(1)
        end_idx = matches[-1]
        print(f"[INFO] Anchored on {os.path.basename(snaps[end_idx])}")
    else:
        end_idx = len(snaps)-1
        print(f"[INFO] Using latest snapshot: {os.path.basename(snaps[end_idx])}")

    start_idx = max(0, end_idx - args.frames + 1)
    seq = snaps[start_idx:end_idx+1]
    if len(seq) < args.frames and args.end_file:
        print(f"[ERROR] Only {len(seq)} frames available, needed {args.frames}", file=sys.stderr)
        sys.exit(1)
    if len(seq) < args.frames:
        print(f"[WARN] Only {len(seq)} frames available, proceeding anyway")

    print(f"[INFO] Sequence frames [{start_idx}→{end_idx}]:")
    for f in seq: print("   ", os.path.basename(f))

    # load and stack
    tf = build_transforms()
    imgs = torch.stack([tf(Image.open(f).convert("RGB")) for f in seq])
    x = imgs.unsqueeze(0).to(device)

    # build model
    model = CNN_LSTM(args.backbone, args.hidden, args.layers, args.dropout, freeze=args.freeze)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()
    print(f"[INFO] Model ready (backbone={args.backbone})")

    # inference
    with torch.no_grad():
        pred = model(x).item()
    print(f"\n[RESULT] Predicted next-step {TARGET_FIELD} = {pred:.3f}%\n")


if __name__ == "__main__":
    main()
