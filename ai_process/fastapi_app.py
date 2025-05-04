# fastapi_app.py
# usage: uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000



import os, glob, re, sys, torch
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from datetime import datetime
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from pydantic import BaseModel

# ────────── CONFIGURATION (edit or override via CLI/env) ──────────
SESSION_NAME      = "test1"  # ai_process/<SESSION_NAME>
HEATMAP_DIR       = "~/liquidLapse/heatmap_snapshots"
MODEL_BASE_DIR    = "~/liquidLapse/ai_process"
DEFAULT_FRAMES    = 10       # # of past images to use
DEFAULT_DEVICE    = "cpu"    # e.g. "cuda:0"
DEFAULT_BACKBONE  = "resnet18"
DEFAULT_HIDDEN    = 128
DEFAULT_LAYERS    = 1
DEFAULT_DROPOUT   = 0.25
# ────────────────────────────────────────────────────────────────


class PredictParams(BaseModel):
    session: Optional[str] = SESSION_NAME
    end_file: Optional[str] = None
    frames: Optional[int]    = DEFAULT_FRAMES
    backbone: Optional[str]  = DEFAULT_BACKBONE
    hidden: Optional[int]    = DEFAULT_HIDDEN
    layers: Optional[int]    = DEFAULT_LAYERS
    dropout: Optional[float] = DEFAULT_DROPOUT

class PredictResponse(BaseModel):
    session: str
    used_model: str
    end_snapshot: str
    frames_used: int
    prediction: float

# 3) Helper functions:
def parse_timestamp(fn: str) -> datetime:
    m = re.search(r"heatmap_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_", fn)
    if not m:
        raise ValueError(f"Cannot parse timestamp from '{fn}'")
    dt = f"{m.group(1)} {m.group(2).replace('-',':')}"
    return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")  # :contentReference[oaicite:2]{index=2}

def list_snapshots(directory: str) -> List[str]:
    path = os.path.expanduser(directory)
    files = [os.path.join(path, f) for f in os.listdir(path)
             if f.startswith("heatmap_") and f.endswith(".png")]
    files.sort(key=lambda p: parse_timestamp(os.path.basename(p)))
    return files

def find_latest_model(session: str) -> str:
    base = os.path.expanduser(os.path.join(MODEL_BASE_DIR, session))
    candidates = glob.glob(os.path.join(base, "train_*", "best_model.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model under {base}")
    latest = max(candidates, key=os.path.getmtime)
    return latest  # :contentReference[oaicite:3]{index=3}

def build_transforms():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])  # :contentReference[oaicite:4]{index=4}

# 4) Model class (reuse from training script)...

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

# 5) FastAPI app setup

app = FastAPI(
    title="BTC Heatmap Prediction API",
    version="1.0.0",
    description="Predict next % change from liquidity-heatmap snapshots"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in prod :contentReference[oaicite:5]{index=5}
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# 6) Startup event to preload model 
@app.on_event("startup")
async def load_model():
    global MODEL, TF
    # defaults
    try:
        mpath = find_latest_model(SESSION_NAME)
    except Exception as e:
        raise RuntimeError(f"Model load failed: {e}")
    from fastapi_app import CNN_LSTM  # guard import
    MODEL = CNN_LSTM(
        DEFAULT_BACKBONE, DEFAULT_HIDDEN, DEFAULT_LAYERS, DEFAULT_DROPOUT, freeze=True
    )
    ckpt = torch.load(mpath, map_location=DEFAULT_DEVICE)
    MODEL.load_state_dict(ckpt)
    MODEL.eval()
    TF = build_transforms()
    print(f"[STARTUP] Loaded model {mpath}")

# 7) /predict endpoint

@app.get("/predict", response_model=PredictResponse)
async def predict(params: PredictParams = Depends()):
    # 1) Determine model path
    model_path = params.model or find_latest_model(params.session)
    # 2) Snapshot list
    snaps = list_snapshots(HEATMAP_DIR)
    if not snaps:
        raise HTTPException(404, "No snapshots available")
    # 3) End index
    if params.end_file:
        idxs = [i for i,f in enumerate(snaps)
                if params.end_file in os.path.basename(f)
                or params.end_file in parse_timestamp(os.path.basename(f)).isoformat()]
        if not idxs:
            raise HTTPException(404, f"end-file '{params.end_file}' not found")
        end = idxs[-1]
    else:
        end = len(snaps)-1
    start = max(0, end - params.frames + 1)
    seq_files = snaps[start:end+1]
    if len(seq_files) < params.frames and params.end_file:
        raise HTTPException(400, "Insufficient frames for requested end-file")
    # 4) Load images
    imgs = torch.stack([TF(Image.open(fp).convert("RGB")) for fp in seq_files])
    x = imgs.unsqueeze(0)
    # 5) Load model if backbone/hyperparams differ
    # (for MVP we use preloaded MODEL)
    with torch.no_grad():
        pred = MODEL(x).item()
    return PredictResponse(
        session=params.session,
        used_model=model_path,
        end_snapshot=os.path.basename(snaps[end]),
        frames_used=len(seq_files),
        prediction=pred
    )
