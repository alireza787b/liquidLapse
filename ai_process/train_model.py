#!/usr/bin/env python3
"""
CNN→LSTM Trainer for BTC Liquidity‐Heatmap Sequences
Author : Alireza Ghaderi • 2025‐05‐XX

Loads your preprocessed sequences (images + future target) and trains:
    1. A transfer‐learned CNN trunk (any torchvision model)
    2. An LSTM head over the CNN feature sequences
    3. A final regressor predicting the next %‐change

Features:
  • Automatic backbone extraction (handles any torchvision model)
  • Configurable hyperparameters at the top
  • Robust summary/TensorBoard graphing that won’t crash your run
  • Clear terminal logs: data split, model dims, epoch‐by‐epoch losses
  • Early stopping + best‐model checkpointing
"""

import os
import json
import math
import csv
import datetime
import time
import inspect

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image

# ╭────────────────── GLOBAL CONFIG (edit here) ──────────────────╮
SESSION_NAME    = "test1"               # which ai_process/<session> to use
BACKBONE_NAME   = "inception_v3"        # any torchvision.models.<name>
FREEZE_BACKBONE = True                  # True = freeze CNN weights
LSTM_HIDDEN     = 128                   # hidden dim of LSTM
LSTM_LAYERS     = 1                     # number of LSTM layers
REG_DROPOUT     = 0.25                  # dropout before final FC
TARGET_FIELD    = "change_percent_step" # JSON field to predict
TRAIN_SPLIT     = 0.8                   # train fraction
VAL_SPLIT       = 0.2                   # val fraction (test = remainder)
BATCH_SIZE      = 4                     # per‐device batch size
EPOCHS          = 30                    # max epochs
LEARNING_RATE   = 1e-4                  # Adam LR
PATIENCE        = 5                     # early‐stop patience
GPU_DEVICE      = 0                     # -1 = CPU, else CUDA index
VISUALIZE_MODEL = True                  # try torchinfo + TensorBoard
# ╰───────────────────────────────────────────────────────────────╯

# ─────────────────────────── Paths ────────────────────────────
BASE_DIR = os.path.expanduser("~/liquidLapse")
SEQ_JSON = f"ai_process/{SESSION_NAME}/sequences/sequences_info.json"
RUN_TS   = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
RUN_DIR  = os.path.join(BASE_DIR, f"ai_process/{SESSION_NAME}/train_{RUN_TS}")
os.makedirs(RUN_DIR, exist_ok=True)

# ───────────────────── Backbone Loader ──────────────────────
def load_backbone(name: str, freeze: bool = True):
    """
    Return cnn trunk (no classifier), feature dim and SAFE input_size.
    For Inception‑style nets force 299×299 to avoid kernel‑size errors.
    """
    fn  = getattr(models, name)
    sig = inspect.signature(fn)
    kwargs = {}
    if "aux_logits" in sig.parameters:
        kwargs["aux_logits"] = False
    if "transform_input" in sig.parameters:
        kwargs["transform_input"] = False

    cnn = fn(weights="DEFAULT", **kwargs)

    # strip classifier
    trunk = nn.Sequential(*list(cnn.children())[:-1])

    # feature dim
    if hasattr(cnn, "fc"):
        feat_dim = cnn.fc.in_features
    else:  # EfficientNet / MobileNet
        feat_dim = cnn.classifier[-1].in_features

    # default size, then patch if too small
    cfg = getattr(cnn, "default_cfg", {})
    inp = cfg.get("input_size", (3, 224, 224))
    input_size = (inp[1], inp[2])          # (H,W)

    # Inception nets need ≥75 px; use official 299
    if name.startswith("inception") and max(input_size) < 299:
        input_size = (299, 299)

    if freeze:
        for p in trunk.parameters():
            p.requires_grad = False
    return trunk, feat_dim, input_size

# ───────────────────────── Dataset ────────────────────────────
# HeatmapSeqDataset: use the SAFE size we just guaranteed
class HeatmapSeqDataset(Dataset):
    def __init__(self, json_path: str):
        with open(os.path.join(BASE_DIR, json_path)) as f:
            meta = json.load(f)
        self.meta = [m for m in meta if "future_prediction" in m]

        _, _, self._input_size = load_backbone(BACKBONE_NAME, freeze=FREEZE_BACKBONE)
        H, W = self._input_size
        self.tf = transforms.Compose([
            transforms.Resize((H, W), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        seq = self.meta[idx]
        imgs = torch.stack([
            self.tf(Image.open(item["image_path"]).convert("RGB"))
            for item in seq["items"]
        ])  # [T,3,H,W]
        tgt = float(seq["future_prediction"][TARGET_FIELD])
        return imgs, torch.tensor(tgt, dtype=torch.float32)

# ───────────────────────── Model ─────────────────────────────
class CNN_LSTM(nn.Module):
    """CNN trunk → LSTM over time → regression head."""
    def __init__(self, backbone, hidden, layers, dropout, freeze=True):
        super().__init__()
        trunk, feat_dim, _ = load_backbone(backbone, freeze=freeze)
        self.features = trunk
        self.lstm     = nn.LSTM(feat_dim, hidden, layers, batch_first=True)
        head = [nn.Linear(hidden, 64), nn.ReLU()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(64, 1))
        self.regressor = nn.Sequential(*head)

        # store for reporting
        self._feat_dim    = feat_dim
        self._lstm_hid    = hidden
        self._lstm_layers = layers

    def forward(self, x):
        B, T, C, H, W = x.shape
        assert H >= 75 and W >= 75, \
            f"Input to backbone too small: {(H,W)} – check transforms!"
        y = x.view(B*T, C, H, W)
        f = self.features(y).view(B, T, -1)
        out, _ = self.lstm(f)
        return self.regressor(out[:, -1]).squeeze(1)

# ───────────────────── Training Step ─────────────────────────
def step(model, loader, loss_fn, optim=None, device="cpu"):
    training = optim is not None
    model.train() if training else model.eval()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if training:
            optim.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        if training:
            loss.backward()
            optim.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)

# ─────────────────────────── Main ────────────────────────────
def main():
    assert 0 < TRAIN_SPLIT < 1 and 0 <= VAL_SPLIT < 1 and TRAIN_SPLIT + VAL_SPLIT <= 1

    # device
    dev = (
        torch.device(f"cuda:{GPU_DEVICE}")
        if GPU_DEVICE >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"[INFO] Device: {dev}")

    # dataset & splits
    ds = HeatmapSeqDataset(SEQ_JSON)
    N  = len(ds)
    n_train = int(TRAIN_SPLIT * N)
    n_val   = int(VAL_SPLIT   * N)
    n_test  = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"[DATA] total={N}  train={n_train}  val={n_val}  test={n_test}")

    # loaders
    dl_train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    dl_val   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # model, loss, optimizer
    model     = CNN_LSTM(BACKBONE_NAME, LSTM_HIDDEN, LSTM_LAYERS, REG_DROPOUT, FREEZE_BACKBONE).to(dev)
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=LEARNING_RATE)

    # architecture report
    print("\n[MODEL]")
    print(f"  Backbone:       {BACKBONE_NAME}")
    print(f"  Feature dim:    {model._feat_dim}")
    print(f"  LSTM layers:    {model._lstm_layers}")
    print(f"  LSTM hidden:    {model._lstm_hid}")
    print(f"  Regressor head: {model.regressor}\n")

    # optional visualization (safe against any errors)
    if VISUALIZE_MODEL:
        try:
            from torchinfo import summary
            # get T and H,W for dummy
            T = ds[0][0].shape[0]
            _, _, (h, w) = load_backbone(BACKBONE_NAME)
            print(summary(model, input_size=(BATCH_SIZE, T, 3, h, w)))
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(RUN_DIR)
            dummy = torch.randn(BATCH_SIZE, T, 3, h, w).to(dev)
            tb.add_graph(model, dummy)
            tb.close()
            print(f"[TBOARD] tensorboard --logdir {RUN_DIR}")
        except Exception as e:
            print(f"[WARN] Model viz skipped: {e}")

    # training loop with early stopping
    best_val, wait = math.inf, 0
    log_csv = os.path.join(RUN_DIR, "train_log.csv")
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "elapsed_s"])
        print(f"[TRAIN] Starting up to {EPOCHS} epochs...\n")
        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            tr = step(model, dl_train, loss_fn, optimizer, dev)
            vl = step(model, dl_val,   loss_fn, None,      dev)
            dt = time.time() - t0
            writer.writerow([epoch, f"{tr:.4f}", f"{vl:.4f}", f"{dt:.1f}"])
            print(f"Epoch {epoch:02d} | train={tr:.4f} | val={vl:.4f} | {dt:.1f}s")

            if vl < best_val:
                best_val, wait = vl, 0
                ckpt = os.path.join(RUN_DIR, "best_model.pt")
                torch.save(model.state_dict(), ckpt)
                print(f"  [SAVE] New best ({vl:.4f}) → {ckpt}")
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"  [STOP] Early stopping (no improvement).")
                    break

    print(f"\n[DONE] Best val loss={best_val:.4f}")
    print(f"Outputs in {RUN_DIR}\n")

if __name__ == "__main__":
    main()
