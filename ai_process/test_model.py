#!/usr/bin/env python3
"""
test_model.py

Inference script that builds a sequence from the last N raw snapshots
and predicts the next-step percentage change using a trained CNN→LSTM model.
"""
import os, re, glob, argparse, torch, json
from datetime import datetime
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# ╭──────────────────── GLOBAL PARAMETERS ─────────────────────╮
SESSION_NAME  = "test1"  # ai_process/<SESSION_NAME>
HEATMAP_DIR   = os.path.expanduser("~/liquidLapse/heatmap_snapshots")
MODEL_GLOB    = f"ai_process/{SESSION_NAME}/train_*/*/best_model.pt"
N_FRAMES      = 10       # number of past snapshots to use
DEVICE        = "cpu"    # e.g. "cuda:0"
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

# ─────────────────────── HELPERS ─────────────────────────────
def parse_timestamp(fname):
    """
    Extract datetime from filenames like:
      heatmap_YYYY-MM-DD_HH-MM-SS_CURRENCY-price.png
    """
    pattern = r"heatmap_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_"
    m = re.search(pattern, fname)
    if not m:
        raise ValueError(f"Filename {fname} doesn't match timestamp pattern")
    dt_str = f"{m.group(1)} {m.group(2).replace('-',':')}"
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")  # :contentReference[oaicite:1]{index=1}

def find_last_model(glob_pattern):
    """
    Pick the latest matching model .pt file
    """
    cands = glob.glob(glob_pattern)
    if not cands:
        raise FileNotFoundError(f"No model found for {glob_pattern}")
    return sorted(cands)[-1]

def build_transforms():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

# ──────────────────────── MAIN ───────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames",    type=int,   default=N_FRAMES,
                        help="Number of past snapshots to use")
    parser.add_argument("--end-file",  type=str,   default=None,
                        help="Specific snapshot filename to end on")
    parser.add_argument("--model-path",type=str,   default=None,
                        help="Path or glob to best_model.pt")
    parser.add_argument("--device",    type=str,   default=DEVICE)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[INFO] Device: {device}")

    # 1) Gather all heatmap filenames and sort by embedded timestamp
    files = glob.glob(os.path.join(HEATMAP_DIR, "heatmap_*.png"))
    files.sort(key=lambda f: parse_timestamp(os.path.basename(f)))
    print(f"[INFO] Found {len(files)} snapshots in {HEATMAP_DIR}")

    # 2) Select last N frames (or anchor on --end-file)
    if args.end_file:
        # find index of the specified file
        base = os.path.basename(args.end_file)
        indices = [i for i,f in enumerate(files) if base in os.path.basename(f)]
        if not indices:
            raise ValueError(f"{args.end_file} not found in snapshots")
        end_idx = indices[-1]
    else:
        end_idx = len(files)-1

    start_idx = max(0, end_idx - args.frames + 1)
    seq_files = files[start_idx:end_idx+1]
    if len(seq_files) < args.frames:
        print(f"[WARN] Only {len(seq_files)} frames available; model expects {args.frames}")

    print(f"[INFO] Using frames {start_idx}→{end_idx} ({len(seq_files)} total)")

    # 3) Load images and apply transforms
    tf = build_transforms()
    imgs = []
    for f in seq_files:
        imgs.append(tf(Image.open(f).convert("RGB")))
    x = torch.stack(imgs).unsqueeze(0).to(device)  # [1,T,3,224,224]

    # 4) Load model architecture & weights
    model_path = args.model_path or find_last_model(MODEL_GLOB)
    from train_model import BACKBONE_NAME, LSTM_HIDDEN, LSTM_LAYERS, REG_DROPOUT, FREEZE_BACKBONE
    model = CNN_LSTM(BACKBONE_NAME, LSTM_HIDDEN, LSTM_LAYERS, REG_DROPOUT, FREEZE_BACKBONE)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()
    print(f"[INFO] Loaded model from {model_path}")

    # 5) Predict
    with torch.no_grad():
        pred = model(x).item()
    print(f"[RESULT] Predicted next {TARGET_FIELD} = {pred:.3f}%")

if __name__=="__main__":
    main()
