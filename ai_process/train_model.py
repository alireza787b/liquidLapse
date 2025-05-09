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
BACKBONE_NAME       = "googlenet"            # any model in torchvision.models
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
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import get_model_weights
from PIL import Image
from termcolor import cprint
from torchvision.models.feature_extraction import get_graph_node_names

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

def warn(msg): cprint(f"[WARNING] {msg}", "yellow")

def build_backbone(backbone_name: str, freeze: bool = True):
    """
    Build a backbone that returns the tensor *just before* the main classifier,
    no matter what the model's internal module naming scheme is.

    Returns:
        feat_extractor : nn.Module
        feat_dim       : int        – flattened feature size
        transform      : callable   – default preprocessing pipeline
        input_size     : tuple(H,W)
    """
    weights_enum = get_model_weights(backbone_name)
    weights      = weights_enum.DEFAULT
    
    # Special handling for Inception_v3 and other models with aux_logits
    if backbone_name == "inception_v3":
        model = getattr(models, backbone_name)(weights=weights, aux_logits=True)
    else:
        model = getattr(models, backbone_name)(weights=weights)

    # Get graph node names for the model
    eval_nodes, _ = get_graph_node_names(model)

    # Model-specific feature extraction approach
    if backbone_name == "inception_v3":
        # For Inception v3, the feature layer is typically before avgpool
        feature_node = "avgpool" 
    else:
        # For other models, find the node right before classifier/fc/head
        excluded_terms = ["classifier", ".fc", ".head", "logits", "aux", "AuxLogits"]
        
        # Find the last layer before any classifier-like nodes
        cls_index = -1
        for i, node in reversed(list(enumerate(eval_nodes))):
            if any(term in node for term in excluded_terms):
                cls_index = i
                break
        
        if cls_index > 0:
            feature_node = eval_nodes[cls_index - 1]
        else:
            # Fallback to the last node if no classifier found
            feature_node = eval_nodes[-2] if len(eval_nodes) > 1 else eval_nodes[-1]
            warn(f"Couldn't identify classifier node for {backbone_name}, using {feature_node}")

    # Create feature extractor with identified node
    try:
        feat_extractor = create_feature_extractor(
            model, return_nodes={feature_node: "features"}
        )
    except ValueError as e:
        # If the first attempt fails, try a different approach for problematic models
        cprint(f"Error with '{feature_node}', falling back to alternative extraction.", "yellow")
        
        # Alternative approach: use a reliable intermediate layer
        # Common layers across most CNN architectures
        fallback_layers = ["flatten", "avgpool", "adaptive_avgpool", "AdaptiveAvgPool2d"]
        
        # Find a suitable fallback layer
        for node in reversed(eval_nodes):
            if any(layer in node.lower() for layer in fallback_layers):
                feature_node = node
                break
        else:
            # Last resort: use the layer just before the end
            feature_node = eval_nodes[-2] if len(eval_nodes) > 1 else eval_nodes[0]
        
        cprint(f"Using fallback feature node: {feature_node}", "yellow")
        feat_extractor = create_feature_extractor(
            model, return_nodes={feature_node: "features"}
        )

    # Freeze backbone if specified
    if freeze:
        for p in feat_extractor.parameters(): 
            p.requires_grad = False

    # Infer feature dimension with a dummy forward pass
    dummy = torch.zeros(1, 3, *weights.meta["min_size"])
    with torch.no_grad():
        feat_output = feat_extractor(dummy)["features"]
        # Handle both flattened and spatial features
        if len(feat_output.shape) > 2:  # If output is [B, C, H, W]
            feat_dim = feat_output.flatten(1).shape[1]
        else:  # Already flattened [B, D]
            feat_dim = feat_output.shape[1]

    return feat_extractor, feat_dim, weights.transforms(), weights.meta["min_size"]


# ──────────── Model ────────────
class CNN_LSTM(nn.Module):
    def __init__(self, backbone_name, hidden, layers, dropout, freeze=True):
        super().__init__()
        self.features, feat_dim, _, _ = build_backbone(backbone_name, freeze)
        self.lstm = nn.LSTM(feat_dim, hidden, layers, batch_first=True)
        self.flatten = nn.Flatten()  # Explicit flatten layer
        
        head = [nn.Linear(hidden, 64), nn.ReLU()]
        if dropout: head.append(nn.Dropout(dropout))
        head.append(nn.Linear(64, 1))
        self.regressor = nn.Sequential(*head)

        # expose sizes for reporting
        self._feat_dim    = feat_dim
        self._lstm_hid    = hidden
        self._lstm_layers = layers

    def forward(self, x):            # x: [B,T,3,H,W]
        B, T, C, H, W = x.shape
        
        # Process each frame through CNN backbone
        feats_list = []
        for t in range(T):
            feat = self.features(x[:, t])["features"]
            # Handle any remaining spatial dimensions
            if len(feat.shape) > 2:
                feat = self.flatten(feat)
            feats_list.append(feat)
        
        # Stack time dimension
        feats = torch.stack(feats_list, dim=1)  # [B,T,D]
        
        # Process through LSTM
        out, _ = self.lstm(feats)
        
        # Take final time step for prediction
        return self.regressor(out[:, -1]).squeeze(1)

# ──────────── Train / Validate step ────────────
def step(model, loader, loss_fn, optim=None, device="cpu"):
    training = optim is not None
    model.train() if training else model.eval()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if training: optim.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        if training:
            loss.backward(); optim.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)

# ──────────── Main ────────────
def main():
    # Safety checks
    assert 0 < TRAIN_SPLIT < 1 and TRAIN_SPLIT+VAL_SPLIT <= 1

    # Device
    device = torch.device(f"cuda:{GPU_DEVICE}" if (GPU_DEVICE>=0 and torch.cuda.is_available()) else "cpu")
    cprint(f"Using device: {device}", "cyan")

    # Backbone + transform
    try:
        backbone, feat_dim, tf, in_size = build_backbone(BACKBONE_NAME, FREEZE_BACKBONE)
        cprint(f"Backbone: {BACKBONE_NAME}  |  input {in_size}  |  feature-dim {feat_dim}", "green")
    except Exception as e:
        cprint(f"Error building backbone: {str(e)}", "red")
        cprint("Available backbones:", "cyan")
        available_models = [name for name in dir(models) 
                            if callable(getattr(models, name)) and name[0].islower()]
        for name in sorted(available_models):
            try:
                # Check if this is actually a model function
                if hasattr(models, name) and callable(getattr(models, name)):
                    weights_enum = get_model_weights(name)
                    if hasattr(weights_enum, 'DEFAULT'):
                        cprint(f"  - {name}", "green")
            except:
                pass
        return

    # Dataset / splits
    try:
        ds = HeatmapSeqDataset(SEQ_JSON, tf)
        n = len(ds)
        n_train, n_val = int(TRAIN_SPLIT*n), int(VAL_SPLIT*n)
        n_test = n - n_train - n_val
        train_ds, val_ds, test_ds = random_split(ds, [n_train,n_val,n_test],
                                                generator=torch.Generator().manual_seed(42))
        cprint(f"Dataset: {n} sequences  (train {n_train}  val {n_val}  test {n_test})", "yellow")
    except Exception as e:
        cprint(f"Error loading dataset: {str(e)}", "red")
        return

    dl_train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    dl_val   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Model
    model = CNN_LSTM(BACKBONE_NAME, LSTM_HIDDEN, LSTM_LAYERS, REG_DROPOUT, FREEZE_BACKBONE).to(device)
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # Optional torchinfo summary
    if VISUALIZE_MODEL:
        try:
            from torchinfo import summary
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