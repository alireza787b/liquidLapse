#!/usr/bin/env python3
"""
test_model.py

Inference on raw heatmap snapshots using the latest (or specified) trained model.
By default, it:
  • Uses the most recent best_model.pt under ai_process/<session>/train_*/
  • Uses the latest heatmap in heatmap_snapshots as the final frame
  • Grabs the preceding N_FRAMES-1 snapshots
  • Predicts next-step % change

You can override via command-line arguments.
"""
import os, re, glob, argparse, sys, torch
from datetime import datetime
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# ╭──────────────────── GLOBAL DEFAULTS ─────────────────────╮
SESSION_NAME = "test1"
HEATMAP_DIR  = os.path.expanduser("~/liquidLapse/heatmap_snapshots")
MODEL_GLOB   = f"ai_process/{SESSION_NAME}/train_*/*/best_model.pt"
N_FRAMES     = 10
DEVICE       = "cpu"   # or "cuda:0"
# ╰───────────────────────────────────────────────────────────╯

# ──────────────────────── MODEL CLASS ─────────────────────────
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

# ──────────────────────── HELPERS ─────────────────────────────
def parse_timestamp(fname):
    m = re.search(r"heatmap_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_", fname)
    if not m:
        raise ValueError(f"Cannot parse timestamp in '{fname}'")
    dt = f"{m.group(1)} {m.group(2).replace('-',':')}"
    return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

def find_latest_model(glob_pat):
    files = glob.glob(glob_pat)
    if not files:
        print(f"[ERROR] No model files match '{glob_pat}'", file=sys.stderr)
        sys.exit(1)
    return sorted(files)[-1]

def list_snapshots(dir_path):
    pats = glob.glob(os.path.join(dir_path, "heatmap_*.png"))
    return sorted(pats, key=lambda f: parse_timestamp(os.path.basename(f)))

def build_transforms():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

# ─────────────────────────── MAIN ─────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session",   default=SESSION_NAME)
    p.add_argument("--model",     help="Path or glob to best_model.pt")
    p.add_argument("--end-file",  help="Filename or timestamp to end on")
    p.add_argument("--frames",    type=int, default=N_FRAMES)
    p.add_argument("--device",    default=DEVICE)
    p.add_argument("--freeze",    action="store_true", help="Freeze CNN weights")
    p.add_argument("--backbone",  default="resnet18")
    p.add_argument("--hidden",    type=int, default=128)
    p.add_argument("--layers",    type=int, default=1)
    p.add_argument("--dropout",   type=float, default=0.25)
    args = p.parse_args()

    # Resolve device
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" else "cpu")
    print(f"[INFO] Using device: {device}")

    # Find model
    model_path = args.model or find_latest_model(
        f"ai_process/{args.session}/train_*/*/best_model.pt"
    )
    print(f"[INFO] Loading model from: {model_path}")

    # Gather snapshots
    snaps = list_snapshots(HEATMAP_DIR)
    print(f"[INFO] Found {len(snaps)} snapshots in {HEATMAP_DIR}")

    # Determine end index
    if args.end_file:
        # allow timestamp or filename substring
        def match(f):
            return args.end_file in os.path.basename(f) or args.end_file in parse_timestamp(os.path.basename(f)).isoformat()
        idxs = [i for i,f in enumerate(snaps) if match(f)]
        if not idxs:
            print(f"[ERROR] end-file '{args.end_file}' not found", file=sys.stderr); sys.exit(1)
        end_idx = idxs[-1]
    else:
        end_idx = len(snaps)-1

    start_idx = max(0, end_idx - args.frames + 1)
    seq_files = snaps[start_idx:end_idx+1]
    if len(seq_files) < args.frames:
        print(f"[WARN] Only {len(seq_files)} frames available, needed {args.frames}")

    print(f"[INFO] Using frames [{start_idx}→{end_idx}]:")
    for f in seq_files:
        print("   ", os.path.basename(f))

    # Load images
    tf = build_transforms()
    imgs = torch.stack([
        tf(Image.open(fp).convert("RGB"))
        for fp in seq_files
    ]).unsqueeze(0).to(device)  # [1,T,3,224,224]

    # Build & load model
    model = CNN_LSTM(
        args.backbone, args.hidden, args.layers, args.dropout, 
        freeze=args.freeze
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[INFO] Model ready. Backbone={args.backbone}, feat→LSTM({args.hidden}×{args.layers})")

    # Inference
    with torch.no_grad():
        pred = model(imgs).item()
    print(f"\n[RESULT] Predicted next {TARGET_FIELD} = {pred:.3f}%\n")


if __name__=="__main__":
    main()
