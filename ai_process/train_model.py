#!/usr/bin/env python3
"""
CNN→LSTM trainer with detailed terminal reporting for BTC liquidity-heatmap sequences.
Author: Alireza Ghaderi • 2025-05-02
"""

# ╭────────────────── GLOBAL CONFIG (edit here) ──────────────────╮
SESSION_NAME        = "test1"               # which processed session to use
BACKBONE_NAME       = "resnet50"            # torchvision.models.<name>
FREEZE_BACKBONE     = True                  # True = freeze CNN weights
LSTM_HIDDEN         = 128                   # hidden size of LSTM
LSTM_LAYERS         = 1                     # number of stacked LSTM layers
REG_DROPOUT         = 0.25                  # dropout before final FC (0 = none)
TARGET_FIELD        = "change_percent_step" # field in future_prediction JSON
TRAIN_SPLIT         = 0.8                   # fraction for training set
VAL_SPLIT           = 0.2                   # fraction for validation set
BATCH_SIZE          = 4                     # batch size per GPU :contentReference[oaicite:1]{index=1}
EPOCHS              = 30                    # maximum epochs
LEARNING_RATE       = 1e-4                  # Adam LR (try 3e-4 or 1e-5) :contentReference[oaicite:2]{index=2}
PATIENCE            = 5                     # early-stop patience :contentReference[oaicite:3]{index=3}
GPU_DEVICE          = 0                     # -1=CPU or CUDA index
VISUALIZE_MODEL     = True                  # print torchinfo + log TensorBoard graph
# ╰───────────────────────────────────────────────────────────────╯

import os, json, math, csv, datetime, time
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image

# ─────────────────────────── Paths ───────────────────────────
BASE_DIR = os.path.expanduser("~/liquidLapse")
SEQ_JSON = f"ai_process/{SESSION_NAME}/sequences/sequences_info.json"
RUN_TS   = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
RUN_DIR  = os.path.join(BASE_DIR, f"ai_process/{SESSION_NAME}/train_{RUN_TS}")
os.makedirs(RUN_DIR, exist_ok=True)

# ───────────────────────── DataSet ────────────────────────────
class HeatmapSeqDataset(Dataset):
    """Loads sequences of heatmap images and their numeric targets."""
    def __init__(self, json_path, tf=None):
        with open(os.path.join(BASE_DIR, json_path),'r') as f:
            meta = json.load(f)
        # keep only entries with a defined future_prediction
        self.meta = [m for m in meta if "future_prediction" in m]
        self.tf = tf or transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
    def __len__(self): return len(self.meta)
    def __getitem__(self, idx):
        seq = self.meta[idx]
        # load T images, apply transforms
        imgs = torch.stack([ self.tf(Image.open(it["image_path"]).convert("RGB"))
                             for it in seq["items"] ])  # [T,3,224,224]
        # numeric target
        y = float(seq["future_prediction"][TARGET_FIELD])
        return imgs, torch.tensor(y, dtype=torch.float32)

# ─────────────────────────── Model ────────────────────────────
class CNN_LSTM(nn.Module):
    def __init__(self, backbone, hidden, layers, dropout, freeze=True):
        super().__init__()
        cnn = getattr(models, backbone)(weights="DEFAULT")  # transfer learning :contentReference[oaicite:4]{index=4}
        self.features = nn.Sequential(*list(cnn.children())[:-1])  # global-avg pool output
        feat_dim = cnn.fc.in_features
        if freeze:
            for p in self.features.parameters(): p.requires_grad=False
        self.lstm = nn.LSTM(feat_dim, hidden, layers, batch_first=True)
        head = [nn.Linear(hidden,64), nn.ReLU()]
        if dropout>0: head.append(nn.Dropout(dropout))
        head.append(nn.Linear(64,1))
        self.regressor = nn.Sequential(*head)
        # Store dims for reporting
        self._feat_dim   = feat_dim
        self._lstm_hid   = hidden
        self._lstm_layers= layers

    def forward(self, x):  # x: [B,T,3,H,W]
        B,T,C,H,W = x.shape
        x = x.view(B*T,C,H,W)
        f = self.features(x).view(B,T,-1)  # [B,T,feat_dim]
        out,_ = self.lstm(f)               # [B,T,hidden]
        return self.regressor(out[:,-1]).squeeze(1)

# ────────────────────── Training Utilities ──────────────────────
def step(model, loader, loss_fn, optim=None, device="cpu"):
    training = optim is not None
    model.train() if training else model.eval()
    total_loss=0.0
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        if training: optim.zero_grad()
        preds = model(X)
        loss  = loss_fn(preds, y)
        if training:
            loss.backward(); optim.step()
        total_loss += loss.item()*X.size(0)
    return total_loss/len(loader.dataset)

# ─────────────────────────── Main ────────────────────────────
def main():
    # verify splits
    assert 0<TRAIN_SPLIT<1 and 0<=VAL_SPLIT<1 and TRAIN_SPLIT+VAL_SPLIT<=1

    # device selection
    device = (torch.device(f"cuda:{GPU_DEVICE}") 
              if GPU_DEVICE>=0 and torch.cuda.is_available() 
              else torch.device("cpu"))
    print(f"[INFO] Using device: {device}")

    # load dataset
    ds = HeatmapSeqDataset(SEQ_JSON)
    N = len(ds)
    n_train = int(TRAIN_SPLIT*N)
    n_val   = int(VAL_SPLIT*N)
    n_test  = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train,n_val,n_test],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"[DATA] total: {N}  train: {n_train}  val: {n_val}  test: {n_test}")

    # data loaders
    dl_train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    dl_val   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # build model
    model = CNN_LSTM(BACKBONE_NAME, LSTM_HIDDEN, LSTM_LAYERS, REG_DROPOUT, FREEZE_BACKBONE).to(device)
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=LEARNING_RATE)

    # report model details
    print("\n[MODEL] Backbone:", BACKBONE_NAME)
    print(f"        Feature dim: {model._feat_dim}")
    print(f"        LSTM layers: {model._lstm_layers}  hidden: {model._lstm_hid}")
    print("        Regressor head:", model.regressor, "\n")

    # optional graph visualisation
    if VISUALIZE_MODEL:
        try:
            from torchinfo import summary
            print(summary(model, input_size=(BATCH_SIZE,len(ds[0][0]),3,224,224)))
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(RUN_DIR)
            dummy = torch.randn(BATCH_SIZE,len(ds[0][0]),3,224,224).to(device)
            tb.add_graph(model, dummy); tb.close()
            print(f"[TBOARD] tensorboard --logdir {RUN_DIR}")
        except ImportError:
            print("[WARN] Install torchinfo/tensorboard to visualise model.")

    # training loop
    best_val,wait=math.inf,0
    log_csv = os.path.join(RUN_DIR,"train_log.csv")
    with open(log_csv,"w",newline="") as logf:
        writer = csv.writer(logf); writer.writerow(["epoch","train_loss","val_loss","time_s"])
        print(f"\n[TRAIN] Starting training for up to {EPOCHS} epochs...\n")
        for epoch in range(1, EPOCHS+1):
            t0 = time.time()
            tr_loss = step(model, dl_train, loss_fn, optimizer, device)
            val_loss= step(model, dl_val,   loss_fn, None,      device)
            dt = time.time()-t0
            writer.writerow([epoch, f"{tr_loss:.4f}", f"{val_loss:.4f}", f"{dt:.1f}"])
            print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {val_loss:.4f} | {dt:.1f}s")
            if val_loss < best_val:
                best_val, wait = val_loss, 0
                ckpt = os.path.join(RUN_DIR,"best_model.pt")
                torch.save(model.state_dict(), ckpt)
                print(f"  [SAVE] New best model (val={val_loss:.4f}) → {ckpt}")
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"  [STOP] Early stopping after {epoch} epochs (no improvement).")
                    break

    print(f"\n[DONE] Best val loss: {best_val:.4f}")
    print(f"Outputs in: {RUN_DIR}\n")

if __name__ == "__main__":
    main()
