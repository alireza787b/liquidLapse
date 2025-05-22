#!/usr/bin/env python3
"""
fastapi_app.py

A FastAPI backend that serves BTC‐heatmap predictions via REST API.

Endpoints:
  GET /predict
    - Auto-detects latest checkpoint under ai_process/<session>/train_*/best_model.pt
    - Loads weights + config
    - Uses last N frames (or anchor via end_file) from heatmap_snapshots/
    - Returns JSON with session, checkpoint, config, frames_used, prediction, optional backtest error

Usage:
    uvicorn fastapi_app:app --reload --port 8100

Swagger UI: http://localhost:8100/docs
"""
import os
import sys
import re
import json
import argparse
from glob import glob
from datetime import datetime
from functools import lru_cache
from typing import Optional, Dict, Any

import torch
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =============================================================================
# Default Configuration
# =============================================================================
BASE_DIR        = os.path.expanduser("~/liquidLapse")
HEATMAP_DIR     = os.path.join(BASE_DIR, "heatmap_snapshots")
MODEL_BASE_DIR  = os.path.join(BASE_DIR, "ai_process")
DEFAULT_SESSION = "test1"
DEFAULT_DEVICE  = "cpu"

# =============================================================================
# Pydantic Models
# =============================================================================
class PredictResponse(BaseModel):
    session: str = Field(..., description="AI session name")
    checkpoint: str = Field(..., description="Path to model checkpoint used")
    config: Dict[str, Any] = Field(..., description="Model config loaded from checkpoint")
    frames_used: int = Field(..., description="Number of frames fed into model")
    timestamp: datetime = Field(..., description="Timestamp of last snapshot used")
    prediction: float = Field(..., description="Model's predicted target value")
    ground_truth: Optional[float] = Field(None, description="Actual target from dataset_info.json (if available)")
    error: Optional[float] = Field(None, description="prediction - ground_truth (if available)")

# =============================================================================
# FastAPI Setup
# =============================================================================
app = FastAPI(
    title="BTC Heatmap Prediction API",
    version="1.0.0",
    description="Serve CNN→LSTM predictions on liquidity-heatmap sequences"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# =============================================================================
# Utility Functions
# =============================================================================

def parse_snapshot_ts(fn: str) -> datetime:
    m = re.match(r"heatmap_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_", fn)
    if not m:
        raise ValueError(f"Cannot parse timestamp from '{fn}'")
    dt = m.group(1) + " " + m.group(2).replace('-',':')
    return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")


def list_snapshots(dirpath: str) -> list[str]:
    files = [f for f in os.listdir(dirpath) if f.startswith("heatmap_") and f.endswith(".png")]
    entries = []
    for fn in files:
        try:
            ts = parse_snapshot_ts(fn)
            entries.append((ts, os.path.join(dirpath, fn)))
        except:
            continue
    entries.sort(key=lambda x: x[0])
    return [p for _, p in entries]


def find_latest_checkpoint(session: str) -> str:
    pattern = os.path.join(MODEL_BASE_DIR, session, 'train_*', 'best_model.pt')
    candidates = glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {os.path.dirname(pattern)}")
    return max(candidates, key=os.path.getmtime)


def build_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

@lru_cache(maxsize=4)
def load_dataset_info(session: str) -> list[dict]:
    path = os.path.join(MODEL_BASE_DIR, session, 'dataset_info.json')
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return []

# =============================================================================
# Model Class
# =============================================================================
class CNN_LSTM(torch.nn.Module):
    def __init__(self, backbone, lstm_hidden, lstm_layers, reg_dropout, freeze):
        super().__init__()
        cnn = getattr(models, backbone)(weights="DEFAULT")
        self.features = torch.nn.Sequential(*list(cnn.children())[:-1])
        feat_dim = cnn.fc.in_features
        if freeze:
            for p in self.features.parameters(): p.requires_grad=False
        self.lstm = torch.nn.LSTM(feat_dim, lstm_hidden, lstm_layers, batch_first=True)
        head = [torch.nn.Linear(lstm_hidden, lstm_hidden//2), torch.nn.ReLU()]
        if reg_dropout>0: head.append(torch.nn.Dropout(reg_dropout))
        head.append(torch.nn.Linear(lstm_hidden//2,1))
        self.regressor = torch.nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T,C,H,W = x.shape
        x2 = x.view(B*T, C, H, W)
        feats = self.features(x2).view(B, T, -1)
        out, _ = self.lstm(feats)
        return self.regressor(out[:,-1,:]).squeeze(1)

# =============================================================================
# Caching loaded models
# =============================================================================
MODEL_CACHE: dict[str, tuple[torch.nn.Module, dict]] = {}

# =============================================================================
# Prediction Endpoint
# =============================================================================
@app.get("/predict", response_model=PredictResponse)
async def predict(
    session: str = Query(DEFAULT_SESSION, description="AI session name"),
    model_path: Optional[str] = Query(None, description="Checkpoint .pt path"),
    end_file:   Optional[str] = Query(None, description="Snapshot substring to anchor"),
    frames:     Optional[int] = Query(None, description="Number of frames (overrides seq_len)"),
    device:     str = Query(DEFAULT_DEVICE, description="Torch device: cpu or cuda:0")
):
    # 1) Resolve checkpoint
    try:
        ckpt = model_path or find_latest_checkpoint(session)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 2) Load or reuse model
    if ckpt not in MODEL_CACHE:
        raw = torch.load(ckpt, map_location=device)
        cfg = raw.get('config')
        sd  = raw.get('state_dict')
        if not cfg or not sd:
            raise HTTPException(status_code=500, detail="Invalid checkpoint format")
        model = CNN_LSTM(
            cfg['backbone'], cfg['lstm_hidden'], cfg['lstm_layers'],
            cfg['reg_dropout'], cfg['freeze']
        )
        model.load_state_dict(sd)
        model.eval()
        MODEL_CACHE[ckpt] = (model, cfg)
    else:
        model, cfg = MODEL_CACHE[ckpt]

    seq_len = frames or cfg.get('seq_len')

    # 3) Gather snapshots
    snaps = list_snapshots(HEATMAP_DIR)
    if not snaps:
        raise HTTPException(status_code=404, detail="No snapshots found")

    # 4) Determine window
    if end_file:
        idxs = [i for i,f in enumerate(snaps) if end_file in os.path.basename(f)]
        if not idxs:
            raise HTTPException(status_code=404, detail=f"end_file '{end_file}' not found")
        end_idx = idxs[-1]
    else:
        end_idx = len(snaps)-1
    start_idx = max(0, end_idx - seq_len + 1)
    seq_files = snaps[start_idx:end_idx+1]
    if len(seq_files) < seq_len:
        # partial window allowed, but warn
        print(f"[WARN] only {len(seq_files)} frames, required {seq_len}")

    # 5) Preprocess
    tfm = build_transforms()
    tensors = []
    for fn in seq_files:
        try:
            img = Image.open(fn).convert('RGB')
        except:
            img = Image.new('RGB', (224,224))
        tensors.append(tfm(img))
    x = torch.stack(tensors).unsqueeze(0).to(device)

    # 6) Inference
    with torch.no_grad():
        pred = model(x).item()

    # 7) Attempt backtest
    ds = load_dataset_info(session)
    gt, err = None, None
    if ds:
        ts = parse_snapshot_ts(os.path.basename(seq_files[-1]))
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        entry = next((e for e in ds if e['timestamp']==ts_str), None)
        if entry:
            tf = cfg.get('target_field')
            gt = entry.get(tf)
            if gt is not None:
                err = pred - gt

    return PredictResponse(
        session=session,
        checkpoint=os.path.basename(ckpt),
        config=cfg,
        frames_used=len(seq_files),
        timestamp=parse_snapshot_ts(os.path.basename(seq_files[-1])),
        prediction=pred,
        ground_truth=gt,
        error=err
    )
