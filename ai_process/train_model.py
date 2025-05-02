#!/usr/bin/env python3
"""
CNN‑to‑LSTM trainer for BTC liquidity‑heat‑map sequences
────────────────────────────────────────────────────────
*   Loads sequences_info.json produced by sequence_generator.py
*   Extracts image‑level CNN features (transfer‑learning)
*   Feeds sequence into an LSTM and predicts the next‑step %‑change
*   Saves best model, CSV log, and optional TensorBoard run

Author : Alireza Ghaderi • 2025‑05‑02
"""

# ╭────────────────── GLOBAL CONFIG (edit here) ──────────────────╮
SESSION_NAME        = "test1"           # which AI‑process session to train on
BACKBONE_NAME       = "resnet18"        # any torchvision model   e.g. "resnet50","efficientnet_b0"
FREEZE_BACKBONE     = True              # True = keep CNN weights fixed (faster/safer on small data)
LSTM_HIDDEN         = 128               # size of LSTM hidden state  (↗ = more capacity)
LSTM_LAYERS         = 1                 # stacked LSTM layers       (2‑3 if you have lots of data)
REG_DROPOUT         = 0.25              # dropout before final FC (0 = no dropout)
TARGET_FIELD        = "change_percent_step"  # label inside future_prediction; swap to "change_percent_hour" for 1‑h horizon
TRAIN_SPLIT         = 0.8               # fraction of dataset for training
VAL_SPLIT           = 0.2               # fraction for validation   (TEST = remainder)
BATCH_SIZE          = 4                 # GPU memory ↑⇒ raise batch_size
EPOCHS              = 30                # hard cap; early‑stop may finish sooner
LEARNING_RATE       = 1e-4              # Adam initial LR (try 3e‑4 or 1e‑5)
PATIENCE            = 5                 # stop if val‑loss hasn’t improved after this many epochs
GPU_DEVICE          = 0                 # set -1 to force CPU, or choose CUDA device index
VISUALIZE_MODEL     = True              # prints torchinfo summary + logs graph to TensorBoard
# ╰───────────────────────────────────────────────────────────────╯


import os, json, math, csv, time, datetime
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image

# ───────────────────────── paths ─────────────────────────
BASE_DIR   = os.path.expanduser("~/liquidLapse")
SEQ_JSON   = f"ai_process/{SESSION_NAME}/sequences/sequences_info.json"
RUN_TS     = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
RUN_DIR    = os.path.join(BASE_DIR, f"ai_process/{SESSION_NAME}/train_{RUN_TS}")
os.makedirs(RUN_DIR, exist_ok=True)

# ───────────────────────── data set ──────────────────────
class HeatmapSeqDataset(Dataset):
    """Returns X:[T,3,H,W]  y:float"""
    def __init__(self, seq_json_path, tf=None):
        with open(os.path.join(BASE_DIR, seq_json_path), 'r') as f:
            meta = json.load(f)
        # keep only sequences that hold the future target
        self.meta = [m for m in meta if "future_prediction" in m]
        self.tf   = tf or transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
    def __len__(self): return len(self.meta)
    def __getitem__(self, idx):
        seq = self.meta[idx]
        imgs = torch.stack([ self.tf(Image.open(it["image_path"]).convert("RGB"))
                             for it in seq["items"] ])           # [T,3,224,224]
        target = torch.tensor(seq["future_prediction"][TARGET_FIELD],
                              dtype=torch.float32)
        return imgs, target

# ───────────────────────── model ─────────────────────────
class CNN_LSTM(nn.Module):
    def __init__(self, backbone_name, hidden, layers, dropout, freeze=True):
        super().__init__()
        cnn = getattr(models, backbone_name)(weights="DEFAULT")
        self.features = nn.Sequential(*list(cnn.children())[:-1])   # GAP output
        feat_dim = cnn.fc.in_features
        if freeze:
            for p in self.features.parameters(): p.requires_grad=False
        self.lstm = nn.LSTM(feat_dim, hidden, layers,
                            batch_first=True)
        head = [nn.Linear(hidden, 64), nn.ReLU()]
        if dropout>0: head += [nn.Dropout(dropout)]
        head += [nn.Linear(64,1)]
        self.regressor = nn.Sequential(*head)

    def forward(self, x):                     # x [B,T,3,H,W]
        B,T,C,H,W = x.shape
        f = self.features(x.view(B*T,C,H,W)).view(B,T,-1)  # [B,T,F]
        h,_ = self.lstm(f)                                 # [B,T,H]
        return self.regressor(h[:,-1]).squeeze(1)          # [B]

# ───────────────────────── training helpers ─────────────
def step(model, loader, loss_fn, opt=None, device="cpu"):
    train = opt is not None
    model.train() if train else model.eval()
    total = 0.0
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        if train: opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        if train:
            loss.backward(); opt.step()
        total += loss.item()*X.size(0)
    return total/len(loader.dataset)

# ───────────────────────── main ─────────────────────────
def main():
    # sanity‑check ratios
    assert 0 < TRAIN_SPLIT < 1 and 0 <= VAL_SPLIT < 1 and TRAIN_SPLIT+VAL_SPLIT<=1
    device = (torch.device(f"cuda:{GPU_DEVICE}") 
              if GPU_DEVICE>=0 and torch.cuda.is_available() else torch.device("cpu"))
    
    ds = HeatmapSeqDataset(SEQ_JSON)
    n_train  = int(TRAIN_SPLIT * len(ds))
    n_val    = int(VAL_SPLIT   * len(ds))
    n_test   = len(ds)-n_train-n_val
    train_ds,val_ds,test_ds = random_split(ds,[n_train,n_val,n_test],
                                           generator=torch.Generator().manual_seed(42))
    dl_train = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True, drop_last=True)
    dl_val   = DataLoader(val_ds,  batch_size=BATCH_SIZE,shuffle=False,drop_last=False)

    model = CNN_LSTM(BACKBONE_NAME,LSTM_HIDDEN,LSTM_LAYERS,REG_DROPOUT,FREEZE_BACKBONE).to(device)
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()),
                                 lr=LEARNING_RATE)

    # optional visualisation
    if VISUALIZE_MODEL:
        try:
            from torchinfo import summary
            print(summary(model, input_size=(BATCH_SIZE, len(ds.meta[0]["items"]), 3,224,224)))
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(RUN_DIR)
            dummy = torch.randn(BATCH_SIZE,len(ds.meta[0]["items"]),3,224,224).to(device)
            tb.add_graph(model, dummy); tb.close()
            print(f"[TensorBoard]   tensorboard --logdir {RUN_DIR}")
        except ImportError:
            print("Install torchinfo / tensorboard to enable model graph.")

    print(f"Run dir : {RUN_DIR}")
    best_val, stagnate = math.inf,0
    log_path = os.path.join(RUN_DIR,"log.csv")
    with open(log_path,"w",newline='') as lf:
        log = csv.writer(lf); log.writerow(["epoch","train","val"])
        for epoch in range(1,EPOCHS+1):
            tr = step(model, dl_train, loss_fn, optimiser, device)
            vl = step(model, dl_val,   loss_fn, None,      device)
            log.writerow([epoch,tr,vl]); lf.flush()
            print(f"Epoch {epoch:02d} | train {tr:.4f}  val {vl:.4f}")
            if vl < best_val:
                best_val, stagnate = vl,0
                torch.save(model.state_dict(), os.path.join(RUN_DIR,"best_model.pt"))
            else:
                stagnate += 1
                if stagnate>=PATIENCE:
                    print("Early stop.")
                    break
    print(f"Finished.  Best val MSE={best_val:.4f}.  Model → {RUN_DIR}/best_model.pt")

if __name__ == "__main__":
    main()
