#!/usr/bin/env python3
"""
CNN→LSTM trainer with dynamic backbone compatibility.
Author: Alireza Ghaderi • 2025-05-02
"""

import os, json, math, csv, datetime, time, inspect
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image

# ╭────────────────── GLOBAL CONFIG (edit here) ──────────────────╮
SESSION_NAME    = "test1"               # which processed session to use
BACKBONE_NAME   = "inception_v3"        # any torchvision.models.<name>
FREEZE_BACKBONE = True                  # True = freeze CNN weights
LSTM_HIDDEN     = 128                   # hidden size of LSTM
LSTM_LAYERS     = 1                     # number of stacked LSTM layers
REG_DROPOUT     = 0.25                  # dropout before final FC (0 = none)
TARGET_FIELD    = "change_percent_step" # field in future_prediction JSON
TRAIN_SPLIT     = 0.8                   # fraction for training set
VAL_SPLIT       = 0.2                   # fraction for validation set
BATCH_SIZE      = 4                     # batch size per device
EPOCHS          = 30                    # maximum epochs
LEARNING_RATE   = 1e-4                  # Adam LR
PATIENCE        = 5                     # early-stop patience
GPU_DEVICE      = 0                     # -1=CPU or CUDA index
VISUALIZE_MODEL = True                  # show torchinfo + TensorBoard
# ╰───────────────────────────────────────────────────────────────╯

# ───────────────────────── Paths ───────────────────────────
BASE_DIR = os.path.expanduser("~/liquidLapse")
SEQ_JSON = f"ai_process/{SESSION_NAME}/sequences/sequences_info.json"
RUN_TS   = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
RUN_DIR  = os.path.join(BASE_DIR, f"ai_process/{SESSION_NAME}/train_{RUN_TS}")
os.makedirs(RUN_DIR, exist_ok=True)

# ─────────────────────────── Backbone Utils ───────────────────────────
def load_backbone(name, freeze=True):
    """
    Instantiates a torchvision model by name, auto-disabling any
    auxiliary heads (e.g., inception_v3(aux_logits=False)),
    and returns (cnn, feat_dim, input_size).
    """
    model_fn = getattr(models, name)
    sig = inspect.signature(model_fn)
    kwargs = {}
    # Disable aux_logits if supported
    if "aux_logits" in sig.parameters:
        kwargs["aux_logits"] = False
    # Disable transform_input if supported (we handle transforms ourselves)
    if "transform_input" in sig.parameters:
        kwargs["transform_input"] = False
    # Load pretrained weights
    cnn = model_fn(weights="DEFAULT", **kwargs)
    # Remove final classifier layers, keep convolution + pool
    trunk = nn.Sequential(*list(cnn.children())[:-1])
    # Feature dimension = in_features of classifier
    feat_dim = cnn.fc.in_features if hasattr(cnn, "fc") else (
        cnn.classifier[-1].in_features if hasattr(cnn, "classifier") else None
    )
    # Determine expected input size: NCHW tuple
    cfg = getattr(cnn, "default_cfg", {})
    input_size = cfg.get("input_size", (3, 224, 224))[1:]  # (H,W) :contentReference[oaicite:2]{index=2}
    # Freeze if requested
    if freeze:
        for p in trunk.parameters(): p.requires_grad = False
    return trunk, feat_dim, input_size

# ───────────────────────── Dataset ────────────────────────────
class HeatmapSeqDataset(Dataset):
    """Loads sequences of heatmap images and their numeric targets."""
    def __init__(self, json_path):
        with open(os.path.join(BASE_DIR, json_path),'r') as f:
            meta = json.load(f)
        self.meta = [m for m in meta if "future_prediction" in m]
        # Dynamically get transform from backbone
        _, _, (h, w) = load_backbone(BACKBONE_NAME, freeze=FREEZE_BACKBONE)
        self.tf = transforms.Compose([
            transforms.Resize((h, w)),     # match model’s default_cfg :contentReference[oaicite:3]{index=3}
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        seq = self.meta[idx]
        imgs = torch.stack([
            self.tf(Image.open(it["image_path"]).convert("RGB"))
            for it in seq["items"]
        ])  # [T,3,H,W]
        y = float(seq["future_prediction"][TARGET_FIELD])
        return imgs, torch.tensor(y, dtype=torch.float32)

# ───────────────────────── Model ────────────────────────────
class CNN_LSTM(nn.Module):
    """Takes [B,T,3,H,W] → CNN trunk → LSTM → regression."""
    def __init__(self, backbone, hidden, layers, dropout, freeze=True):
        super().__init__()
        trunk, feat_dim, _ = load_backbone(backbone, freeze=freeze)
        self.features = trunk           # outputs [B*T, feat_dim, 1, 1]
        self.lstm = nn.LSTM(feat_dim, hidden, layers, batch_first=True)
        head = [nn.Linear(hidden, 64), nn.ReLU()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(64, 1))
        self.regressor = nn.Sequential(*head)
        # Store for reporting
        self._feat_dim    = feat_dim
        self._lstm_hid    = hidden
        self._lstm_layers = layers

    def forward(self, x):
        B, T, C, H, W = x.shape
        # collapse batch & time to run CNN trunk
        f = self.features(x.view(B*T, C, H, W))
        f = f.view(B, T, -1)            # [B, T, feat_dim]
        out, _ = self.lstm(f)            # [B, T, hidden]
        return self.regressor(out[:, -1]).squeeze(1)

# ─────────────────────── Training Utilities ───────────────────────
def step(model, loader, loss_fn, optim=None, device="cpu"):
    train = optim is not None
    model.train() if train else model.eval()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if train: optim.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        if train:
            loss.backward(); optim.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)

# ─────────────────────────── Main ────────────────────────────
def main():
    assert 0 < TRAIN_SPLIT < 1 and 0 <= VAL_SPLIT < 1 and TRAIN_SPLIT + VAL_SPLIT <= 1

    device = (torch.device(f"cuda:{GPU_DEVICE}") if GPU_DEVICE >= 0 and torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"[INFO] Device: {device}")

    ds = HeatmapSeqDataset(SEQ_JSON)
    N = len(ds)
    n_train = int(TRAIN_SPLIT * N)
    n_val   = int(VAL_SPLIT   * N)
    n_test  = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )
    print(f"[DATA] total={N}  train={n_train}  val={n_val}  test={n_test}")

    dl_train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dl_val   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = CNN_LSTM(BACKBONE_NAME, LSTM_HIDDEN, LSTM_LAYERS, REG_DROPOUT, FREEZE_BACKBONE).to(device)
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # Report architecture
    print("\n[MODEL]")
    print(f"  Backbone:       {BACKBONE_NAME}")
    print(f"  Feature dim:    {model._feat_dim}")
    print(f"  LSTM layers:    {model._lstm_layers}")
    print(f"  LSTM hidden:    {model._lstm_hid}")
    print(f"  Regressor head: {model.regressor}\n")

    # Visualization (optional)
    if VISUALIZE_MODEL:
        try:
            from torchinfo import summary
            # dummy input shape: (B,T,C,H,W)
            T = len(ds[0][0])
            _, (h, w) = load_backbone(BACKBONE_NAME)
            print(summary(model, input_size=(BATCH_SIZE, T, 3, h, w)))
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(RUN_DIR)
            dummy = torch.randn(BATCH_SIZE, T, 3, h, w).to(device)
            tb.add_graph(model, dummy); tb.close()
            print(f"[TBOARD] tensorboard --logdir {RUN_DIR}")
        except ImportError:
            print("[WARN] Install torchinfo & tensorboard to visualise model.")

    # Training loop
    best_val, wait = math.inf, 0
    log_csv = os.path.join(RUN_DIR, "train_log.csv")
    with open(log_csv, "w", newline="") as logf:
        writer = csv.writer(logf)
        writer.writerow(["epoch", "train_loss", "val_loss", "time_s"])
        print(f"[TRAIN] Starting up to {EPOCHS} epochs...\n")
        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            tl = step(model, dl_train, loss_fn, optimizer, device)
            vl = step(model, dl_val,   loss_fn, None,      device)
            dt = time.time() - t0
            writer.writerow([epoch, f"{tl:.4f}", f"{vl:.4f}", f"{dt:.1f}"])
            print(f"Epoch {epoch:02d} | train={tl:.4f} | val={vl:.4f} | {dt:.1f}s")
            if vl < best_val:
                best_val, wait = vl, 0
                pth = os.path.join(RUN_DIR, "best_model.pt")
                torch.save(model.state_dict(), pth)
                print(f"  [SAVE] New best ({vl:.4f}) → {pth}")
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"  [STOP] Early stopping (no improvement).")
                    break

    print(f"\n[DONE] Best val loss={best_val:.4f}")
    print(f"Outputs in {RUN_DIR}\n")

if __name__ == "__main__":
    main()
