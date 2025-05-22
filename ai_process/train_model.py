#!/usr/bin/env python3
"""
Automatic CNN→LSTM Trainer for Liquidity Heatmap Sequences

Enhanced with real-time progress bars (tqdm) and TensorBoard logging for live monitoring.

This script loads clean, fixed-length sequences and trains a CNN-LSTM regression model.
All parameters are declared at the top as global defaults but can be overridden via CLI.
Sequence length is inferred automatically from the JSON.
After training, it saves both the model’s state_dict and its config parameters into a checkpoint.

Usage:
    python train_model.py [options]

To monitor training in TensorBoard:
    tensorboard --logdir=<base_dir>/ai_process/<session>/train_<timestamp>

Upon saving, the checkpoint includes:
  - "state_dict": model weights
  - "config": dict of model constructor args + seq_len

In testing, you can load this single checkpoint and reconstruct the exact model without re-specifying hyperparameters.
"""
import os
import json
import math
import datetime
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from tqdm.auto import tqdm
from PIL import Image

# =============================================================================
# Global Defaults (modifiable here or via CLI)
# =============================================================================
DEFAULT_BASE_DIR        = os.path.expanduser("~/liquidLapse")
DEFAULT_SESSION         = "test1"
DEFAULT_SEQ_JSON_TMPL   = "ai_process/{session}/sequences/sequences_info.json"
DEFAULT_BACKBONE        = "googlenet"
DEFAULT_FREEZE_BACKBONE = True
DEFAULT_LSTM_HIDDEN     = 128
DEFAULT_LSTM_LAYERS     = 1
DEFAULT_REG_DROPOUT     = 0.25
DEFAULT_TARGET_FIELD    = "future_future_4h_change_percent"
DEFAULT_TRAIN_SPLIT     = 0.8
DEFAULT_VAL_SPLIT       = 0.1
DEFAULT_BATCH_SIZE      = 4
DEFAULT_EPOCHS          = 30
DEFAULT_LR              = 1e-4
DEFAULT_PATIENCE        = 5
DEFAULT_GPU_DEVICE      = 0
DEFAULT_VISUALIZE       = False
DEFAULT_SEED            = 42

# =============================================================================
# Dataset
# =============================================================================
class HeatmapSequenceDataset(Dataset):
    """Loads sequences of images & numeric targets from JSON metadata."""
    def __init__(self, json_path, target_field, transform=None):
        with open(json_path, 'r') as f:
            meta = json.load(f)
        self.data = [m for m in meta if m.get(target_field) is not None]
        self.target_field = target_field
        self.transform = transform or transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.seq_len = len(self.data[0]['items']) if self.data else 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        imgs = []
        for itm in seq['items']:
            img_path = itm.get('image_path')
            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                except Exception:
                    img = Image.new('RGB',(224,224))
            else:
                img = Image.new('RGB',(224,224))
            imgs.append(self.transform(img))
        X = torch.stack(imgs)  # [T,3,H,W]
        y = torch.tensor(float(seq[self.target_field]), dtype=torch.float32)
        return X, y

# =============================================================================
# Model
# =============================================================================
class CNN_LSTM(nn.Module):
    def __init__(self, backbone, hidden, layers, dropout, freeze):
        super().__init__()
        cnn = getattr(models, backbone)(weights="DEFAULT")
        self.features = nn.Sequential(*list(cnn.children())[:-1])
        feat_dim = cnn.fc.in_features
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
        self.lstm = nn.LSTM(feat_dim, hidden, layers, batch_first=True)
        head = [nn.Linear(hidden, hidden//2), nn.ReLU()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(hidden//2, 1))
        self.regressor = nn.Sequential(*head)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.features(x).view(B, T, -1)
        out, _ = self.lstm(feats)
        return self.regressor(out[:, -1, :]).squeeze(1)

# =============================================================================
# Training Utilities
# =============================================================================
def train_epoch(model, loader, loss_fn, optimizer, device):
    """Train for one epoch, with tqdm progress bar and batch-level loss."""
    model.train()
    running_loss = 0.0
    total = 0
    bar = tqdm(loader, desc="Train", leave=False)
    for X, y in bar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        batch_size = X.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size
        bar.set_postfix(train_loss=(running_loss / total))
    return running_loss / total


def eval_epoch(model, loader, loss_fn, device):
    """Evaluate for one epoch, with tqdm progress bar."""
    model.eval()
    running_loss = 0.0
    total = 0
    bar = tqdm(loader, desc="Val  ", leave=False)
    with torch.no_grad():
        for X, y in bar:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y)
            batch_size = X.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            bar.set_postfix(val_loss=(running_loss / total))
    return running_loss / total

# =============================================================================
# Main Training
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN-LSTM on heatmap sequences.")
    parser.add_argument('--base_dir',       type=str,   default=DEFAULT_BASE_DIR)
    parser.add_argument('--session',        type=str,   default=DEFAULT_SESSION)
    parser.add_argument('--target_field',   type=str,   default=DEFAULT_TARGET_FIELD)
    parser.add_argument('--backbone',       type=str,   default=DEFAULT_BACKBONE)
    parser.add_argument('--freeze_backbone',action='store_true', default=DEFAULT_FREEZE_BACKBONE)
    parser.add_argument('--lstm_hidden',    type=int,   default=DEFAULT_LSTM_HIDDEN)
    parser.add_argument('--lstm_layers',    type=int,   default=DEFAULT_LSTM_LAYERS)
    parser.add_argument('--reg_dropout',    type=float, default=DEFAULT_REG_DROPOUT)
    parser.add_argument('--train_split',    type=float, default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument('--val_split',      type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument('--batch_size',     type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs',         type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument('--lr',             type=float, default=DEFAULT_LR)
    parser.add_argument('--patience',       type=int,   default=DEFAULT_PATIENCE)
    parser.add_argument('--gpu_device',     type=int,   default=DEFAULT_GPU_DEVICE)
    parser.add_argument('--visualize',      action='store_true', default=DEFAULT_VISUALIZE)
    parser.add_argument('--seed',           type=int,   default=DEFAULT_SEED)
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    if args.gpu_device >= 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # setup directories and TensorBoard
    seq_json = os.path.join(args.base_dir, DEFAULT_SEQ_JSON_TMPL.format(session=args.session))
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = os.path.join(args.base_dir, f"ai_process/{args.session}/train_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    # device selection
    device = (torch.device(f"cuda:{args.gpu_device}")
              if args.gpu_device >= 0 and torch.cuda.is_available() else torch.device('cpu'))
    print(f"[INFO] Device: {device}")

    # data loading and splits
    dataset = HeatmapSequenceDataset(seq_json, args.target_field)
    N = len(dataset)
    seq_len = dataset.seq_len
    n_train = int(args.train_split * N)
    n_val   = int(args.val_split   * N)
    n_test  = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dl_val   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    print(f"[DATA] total={N}, train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, seq_len={seq_len}")

    # model, loss, optimizer
    model = CNN_LSTM(
        args.backbone,
        args.lstm_hidden,
        args.lstm_layers,
        args.reg_dropout,
        args.freeze_backbone
    ).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # optional model summary
    if args.visualize:
        try:
            from torchinfo import summary
            print(summary(model, input_size=(args.batch_size, seq_len,3,224,224)))
        except ImportError:
            print("[WARN] Install torchinfo for model summary.")

    # training loop with early stopping
    best_val, wait = math.inf, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, dl_train, loss_fn, optimizer, device)
        va_loss = eval_epoch(model, dl_val,   loss_fn,      device)
        epoch_time = time.time() - t0
        eta = epoch_time * (args.epochs - epoch)

        # console summary
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train={tr_loss:.4f} | val={va_loss:.4f} | "
            f"{epoch_time:.1f}s (ETA {eta/60:.1f}m)"
        )

        # log to TensorBoard
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val",   va_loss, epoch)
        writer.flush()

        # checkpointing & early stop
        if va_loss < best_val:
            best_val, wait = va_loss, 0
            ckpt_path = os.path.join(run_dir, 'best_model.pt')
            checkpoint = {
                'state_dict': model.state_dict(),
                'config': {
                    'backbone': args.backbone,
                    'freeze': args.freeze_backbone,
                    'lstm_hidden': args.lstm_hidden,
                    'lstm_layers': args.lstm_layers,
                    'reg_dropout': args.reg_dropout,
                    'seq_len': seq_len
                }
            }
            torch.save(checkpoint, ckpt_path)
            print(f"[SAVE] Checkpoint saved to {ckpt_path}")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[STOP] Early stopping at epoch {epoch}")
                break

    # test evaluation
    if n_test > 0:
        dl_test = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        test_loss = eval_epoch(model, dl_test, loss_fn, device)
        print(f"[TEST] Test loss: {test_loss:.4f}")

    writer.close()
    print(f"[DONE] Best val loss: {best_val:.4f} | Outputs: {run_dir}")
