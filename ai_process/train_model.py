#!/usr/bin/env python3
"""
Universal CNN→LSTM trainer for BTC liquidity-heatmap sequences
──────────────────────────────────────────────────────────────
* Backbone-agnostic: works with any torchvision classification model
* Auto-selects correct input resize / normalisation from pre-trained weights
* Auto-detects feature-vector dimension for LSTM
* Detailed, colourised terminal reporting + optional TensorBoard
Author: Alireza Ghaderi  ·  Updated 2025-05-09
"""

# ╭─────────────────── GLOBAL CONFIG (edit as needed) ─────────────────╮
SESSION_NAME        = "test1"               # which ai_process/ session
BACKBONE_NAME       = "inceptionv3"            # any model in torchvision.models
FREEZE_BACKBONE     = True                  # freeze CNN during LSTM training?
LSTM_HIDDEN         = 128                   # LSTM hidden size
LSTM_LAYERS         = 1                     # number of stacked LSTM layers
REG_DROPOUT         = 0.25                  # dropout before final FC
TARGET_FIELD        = "change_percent_hour" # label key in JSON
TRAIN_SPLIT         = 0.8
VAL_SPLIT           = 0.2
BATCH_SIZE          = 4
EPOCHS              = 30
LEARNING_RATE       = 1e-4
PATIENCE            = 5
GPU_DEVICE          = 0                     # -1 → CPU
VISUALIZE_MODEL     = True                  # torchinfo + TensorBoard
# ╰────────────────────────────────────────────────────────────────────╯

import os, json, csv, math, time, datetime, sys, warnings
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor   # :contentReference[oaicite:3]{index=3}
from torchvision.models import get_model_weights                              # :contentReference[oaicite:4]{index=4}
from PIL import Image
from termcolor import cprint

# ──────────── Paths ────────────
BASE_DIR   = os.path.expanduser("~/liquidLapse")
SEQ_JSON   = f"ai_process/{SESSION_NAME}/sequences/sequences_info.json"
RUN_TS     = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
RUN_DIR    = os.path.join(BASE_DIR, f"ai_process/{SESSION_NAME}/train_{RUN_TS}")
os.makedirs(RUN_DIR, exist_ok=True)

# ──────────── Dataset ────────────
class HeatmapSeqDataset(Dataset):
    """Load sequences + numeric targets with backbone-specific transforms."""
    def __init__(self, json_rel_path, transform):
        with open(os.path.join(BASE_DIR, json_rel_path)) as f:
            meta_all = json.load(f)
        self.meta = [m for m in meta_all if "future_prediction" in m]
        self.tf   = transform

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        entry = self.meta[idx]
        imgs = torch.stack([
            self.tf(Image.open(it["original_path"]).convert("RGB"))
            for it in entry["items"]
        ])  # [T,3,H,W]
        target = float(entry["future_prediction"][TARGET_FIELD])
        return imgs, torch.tensor(target, dtype=torch.float32)

# ──────────── Backbone helper ────────────
def build_backbone(backbone_name:str, freeze:bool=True):
    """Return (feature_extractor, feat_dim, transform, input_size)."""
    # 1. get default weights enum
    weight_enum = get_model_weights(backbone_name)  # :contentReference[oaicite:5]{index=5}
    weights     = weight_enum.DEFAULT
    model       = getattr(models, backbone_name)(weights=weights)
    # 2. pick node just before classifier automatically
    #    heuristic: last child before attr that looks like classifier
    cls_candidates = ("fc", "classifier", "head", "heads", "_fc")
    leaf_node = None
    for name, _module in model.named_modules():
        if any(name.endswith(a) for a in cls_candidates):
            leaf_node = ".".join(name.split(".")[:-1]) or "flatten"
            break
    if leaf_node is None:  # fallback to last layer before pooling
        leaf_node = list(dict(model.named_modules()).keys())[-2]

    feat_extractor = create_feature_extractor(
        model, return_nodes={leaf_node: "features"}
    )                                                     # :contentReference[oaicite:6]{index=6}
    if freeze:
        for p in feat_extractor.parameters():
            p.requires_grad = False

    # 3. infer feature-vector dim
    dummy = torch.zeros(1,3,*weights.meta["min_size"])
    with torch.no_grad():
        feat_dim = feat_extractor(dummy)["features"].view(1,-1).shape[1]

    transform = weights.transforms()
    input_size = weights.meta["min_size"]
    return feat_extractor, feat_dim, transform, input_size

# ──────────── Model ────────────
class CNN_LSTM(nn.Module):
    def __init__(self, backbone_name, hidden, layers, dropout, freeze=True):
        super().__init__()
        self.features, feat_dim, _, _ = build_backbone(backbone_name, freeze)
        self.lstm = nn.LSTM(feat_dim, hidden, layers, batch_first=True)
        head = [nn.Linear(hidden,64), nn.ReLU()]
        if dropout: head.append(nn.Dropout(dropout))
        head.append(nn.Linear(64,1))
        self.regressor = nn.Sequential(*head)

        # expose sizes for reporting
        self._feat_dim   = feat_dim
        self._lstm_hid   = hidden
        self._lstm_layers= layers

    def forward(self, x):            # x: [B,T,3,H,W]
        B,T,_,H,W = x.shape
        feats = self.features(x.view(B*T,3,H,W))["features"].view(B,T,-1)
        out,_  = self.lstm(feats)
        return self.regressor(out[:,-1]).squeeze(1)

# ──────────── Train / Validate step ────────────
def step(model, loader, loss_fn, optim=None, device="cpu"):
    training = optim is not None
    model.train() if training else model.eval()
    total=0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if training: optim.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        if training:
            loss.backward(); optim.step()
        total += loss.item()*X.size(0)
    return total / len(loader.dataset)

# ──────────── Main ────────────
def main():
    # Safety checks
    assert 0 < TRAIN_SPLIT < 1 and TRAIN_SPLIT+VAL_SPLIT <= 1

    # Device
    device = torch.device(f"cuda:{GPU_DEVICE}" if (GPU_DEVICE>=0 and torch.cuda.is_available()) else "cpu")
    cprint(f"Using device: {device}", "cyan")

    # Backbone + transform
    backbone, feat_dim, tf, in_size = build_backbone(BACKBONE_NAME, FREEZE_BACKBONE)
    cprint(f"Backbone: {BACKBONE_NAME}  |  input {in_size}  |  feature-dim {feat_dim}", "green")

    # Dataset / splits
    ds = HeatmapSeqDataset(SEQ_JSON, tf)
    n = len(ds)
    n_train, n_val = int(TRAIN_SPLIT*n), int(VAL_SPLIT*n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(ds, [n_train,n_val,n_test],
                                             generator=torch.Generator().manual_seed(42))
    cprint(f"Dataset: {n} sequences  (train {n_train}  val {n_val}  test {n_test})", "yellow")

    dl_train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    dl_val   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Model
    model = CNN_LSTM(BACKBONE_NAME, LSTM_HIDDEN, LSTM_LAYERS, REG_DROPOUT, FREEZE_BACKBONE).to(device)
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # Optional torchinfo summary
    if VISUALIZE_MODEL:
        try:
            from torchinfo import summary                                   # :contentReference[oaicite:7]{index=7}
            summary(model, input_size=(BATCH_SIZE, len(ds[0][0]), 3, *in_size))
        except ImportError:
            warn("Install torchinfo for prettier model summaries.")

    # Training loop
    best_val, wait = math.inf, 0
    with open(os.path.join(RUN_DIR, "train_log.csv"), "w", newline="") as lf:
        log = csv.writer(lf); log.writerow(["epoch","train","val","sec"])
        cprint(f"\n[TRAIN] Up to {EPOCHS} epochs (early-stop {PATIENCE})\n", "cyan")
        for epoch in range(1, EPOCHS+1):
            t0 = time.time()
            tr = step(model, dl_train, loss_fn, optimiser, device)
            vl = step(model, dl_val,   loss_fn, None,       device)
            dt = time.time()-t0
            log.writerow([epoch, f"{tr:.4f}", f"{vl:.4f}", f"{dt:.1f}"]); lf.flush()
            print(f"Epoch {epoch:02d} | train {tr:.4f} | val {vl:.4f} | {dt:.1f}s")
            if vl < best_val:
                best_val, wait = vl, 0
                ckpt = os.path.join(RUN_DIR, "best_model.pt")
                torch.save(model.state_dict(), ckpt)
                cprint(f"  [SAVE] New best model → {ckpt}", "green")
            else:
                wait += 1
                if wait >= PATIENCE:
                    cprint(f"  [STOP] Early-stopping (no val-improve {PATIENCE}×)", "red")
                    break
    cprint(f"\nBest val MSE: {best_val:.4f}", "yellow")
    cprint(f"Outputs saved in: {RUN_DIR}", "cyan")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)  # clean fx trace msgs
    main()
