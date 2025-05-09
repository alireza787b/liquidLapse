#!/usr/bin/env python3
"""
fastapi_app.py

A FastAPI backend that serves live BTC‐heatmap predictions.

By default, GET /predict will:
  1. Auto-discover the latest model in:
       ~/liquidLapse/ai_process/<session>/train_*/best_model.pt
  2. Auto-select the latest heatmap in:
       ~/liquidLapse/heatmap_snapshots/
  3. Use the last N_FRAMES snapshots (configurable) to predict the next % change.
  4. Return a JSON payload with session, model used, snapshot info, frames count, and prediction.

Override via query parameters:
  • session=anotherSession  
  • model=/full/path/to/best_model.pt  
  • end_file=2025-04-16_06-04-00 (filename or timestamp substring)  
  • frames=8  
  • backbone, hidden, layers, dropout, freeze  

Run:
    uvicorn fastapi_app:app --reload --port 8100

Swagger: http://localhost:8100/docs
"""

import os, glob, re, sys, torch
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import json

# ╭─────────── DEFAULT CONFIGURATION ────────────╮
SESSION_NAME     = "test1"
HEATMAP_DIR      = os.path.expanduser("~/liquidLapse/heatmap_snapshots")
MODEL_BASE_DIR   = os.path.expanduser("~/liquidLapse/ai_process")
DEFAULT_FRAMES   = 10
DEFAULT_DEVICE   = "cpu"
DEFAULT_BACKBONE = "googlenet"
DEFAULT_HIDDEN   = 128
DEFAULT_LAYERS   = 1
DEFAULT_DROPOUT  = 0.25

# Define the prediction target variable: `change_percent_step` or `change_percent_hour`
TARGET_VARIABLE = "change_percent_hour"  # Change this to "change_percent_hour" for hourly predictions
# ╰──────────────────────────────────────────────╯

# ─────────────────── Pydantic Schemas ───────────────────
class PredictParams(BaseModel):
    session:    Optional[str] = SESSION_NAME
    model:      Optional[str] = None
    end_file:   Optional[str] = None
    frames:     Optional[int] = DEFAULT_FRAMES
    backbone:   Optional[str] = DEFAULT_BACKBONE
    hidden:     Optional[int] = DEFAULT_HIDDEN
    layers:     Optional[int] = DEFAULT_LAYERS
    dropout:    Optional[float] = DEFAULT_DROPOUT
    freeze:     Optional[bool] = True

class PredictResponse(BaseModel):
    session:       str
    used_model:    str
    model_name:    str
    target_variable: str
    predictions:   List[dict]
    frames_used:   int

# ─────────────────────── Helpers ───────────────────────
def parse_timestamp(fname: str) -> datetime:
    pattern = r"heatmap_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_"
    m = re.search(pattern, fname)
    if not m:
        raise ValueError(f"Cannot parse timestamp from '{fname}'")
    dt_str = f"{m.group(1)} {m.group(2).replace('-',':')}"
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

def list_snapshots(directory: str) -> List[str]:
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith("heatmap_") and f.endswith(".png")
    ]
    files.sort(key=lambda p: parse_timestamp(os.path.basename(p)))
    return files

def find_latest_model(session: str) -> str:
    base = os.path.join(MODEL_BASE_DIR, session)
    pattern = os.path.join(base, "train_*", "best_model.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No model found under '{base}'")
    latest = max(candidates, key=os.path.getmtime)
    return latest

def build_transforms():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

# Load the dataset_info.json file into memory
def load_dataset_info(session_name):
    dataset_path = os.path.join(MODEL_BASE_DIR, session_name, 'dataset_info.json')
    try:
        with open(dataset_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[WARNING] Dataset info not found at {dataset_path}")
        return []

# ─────────────────── Model Definition ───────────────────
class CNN_LSTM(nn.Module):
    def __init__(self, backbone, hidden, layers, dropout, freeze=True):
        super().__init__()
        cnn = getattr(models, backbone)(weights="DEFAULT")
        self.features = nn.Sequential(*list(cnn.children())[:-1])
        feat_dim = cnn.fc.in_features
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
        self.lstm = nn.LSTM(feat_dim, hidden, layers, batch_first=True)
        head = [nn.Linear(hidden,64), nn.ReLU()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(64,1))
        self.regressor = nn.Sequential(*head)

    def forward(self, x):
        B,T,C,H,W = x.shape
        f = self.features(x.view(B*T,C,H,W)).view(B,T,-1)
        out,_ = self.lstm(f)
        return self.regressor(out[:,-1]).squeeze(1)

# ─────────────────── FastAPI Setup ───────────────────
app = FastAPI(
    title="BTC Heatmap Prediction API",
    version="1.0.0",
    description="Predict next % change from BTC liquidity heatmaps"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Schema for debug info
class DebugInfo(BaseModel):
    dataset_size: int
    dataset_samples: List[dict]
    heatmap_files: List[str]
    match_results: List[dict]

# Global variables
MODEL = None
TF = None
DATASET_INFO = None

# Preload model & transforms on startup to avoid per-request overhead
@app.on_event("startup")
async def startup_event():
    global MODEL, TF, DATASET_INFO
    try:
        mpath = find_latest_model(SESSION_NAME)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    MODEL = CNN_LSTM(DEFAULT_BACKBONE, DEFAULT_HIDDEN,
                     DEFAULT_LAYERS, DEFAULT_DROPOUT, freeze=True)
    ckpt = torch.load(mpath, map_location=DEFAULT_DEVICE)
    MODEL.load_state_dict(ckpt)
    MODEL.eval()
    TF = build_transforms()
    DATASET_INFO = load_dataset_info(SESSION_NAME)
    print(f"[STARTUP] Loaded model {mpath}")
    
    # Print dataset info summary for debugging
    if DATASET_INFO:
        print(f"[STARTUP] Loaded {len(DATASET_INFO)} dataset entries")
        # Print a few examples of filenames in the dataset
        for i, entry in enumerate(DATASET_INFO[:3]):
            print(f"[DATASET] Sample {i}: {entry['new_filename']}")
    else:
        print("[WARNING] No dataset info loaded")

# ─────────────────── Prediction Endpoint ───────────────────
@app.get("/predict", response_model=PredictResponse)
async def predict(params: PredictParams = Depends()):
    # 1) Resolve model path
    try:
        model_path = params.model or find_latest_model(params.session)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 2) Gather snapshots
    snaps = list_snapshots(HEATMAP_DIR)
    if not snaps:
        raise HTTPException(404, "No snapshots found")

    # 3) Determine end index
    if params.end_file:
        matches = [
            i for i,f in enumerate(snaps)
            if params.end_file in os.path.basename(f)
            or params.end_file in parse_timestamp(os.path.basename(f)).isoformat()
        ]
        if not matches:
            raise HTTPException(404, f"end-file '{params.end_file}' not found")
        end_idx = matches[-1]
    else:
        end_idx = len(snaps) - 1

    start_idx = max(0, end_idx - params.frames + 1)
    seq_files = snaps[start_idx:end_idx+1]
    if params.end_file and len(seq_files) < params.frames:
        raise HTTPException(400, f"Only {len(seq_files)} frames available for requested end-file")

    # 4) Load & preprocess images
    imgs = torch.stack([
        TF(Image.open(fp).convert("RGB")) for fp in seq_files
    ]).unsqueeze(0)  # [1,T,3,224,224]

    # 5) Inference
    with torch.no_grad():
        prediction = MODEL(imgs).item()

    # Load dataset info if not already loaded
    dataset_info = DATASET_INFO
    if not dataset_info and params.session != SESSION_NAME:
        dataset_info = load_dataset_info(params.session)

    # Try to find ground truth
    ground_truth = None
    percent_change = None
    
    last_file = os.path.basename(seq_files[-1])
    if dataset_info:
        # Extract date and time from the filename to match with dataset_info
        timestamp_match = re.search(r"heatmap_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_", last_file)
        if timestamp_match:
            date_str = timestamp_match.group(1).replace("-", "")  # Convert 2025-04-01 to 20250401
            time_str = timestamp_match.group(2)  # Keep 03-50-56 format
            
            # Try matching on timestamp components
            matched_metadata = next(
                (entry for entry in dataset_info 
                 if date_str in entry['new_filename'] and time_str in entry['new_filename']),
                None
            )
            
            if not matched_metadata:
                # Fallback: try to extract timestamp and match by closest timestamp
                dt_str = f"{timestamp_match.group(1)} {timestamp_match.group(2).replace('-',':')}"
                current_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                current_ts = current_dt.timestamp()
                
                # Find closest entry by timestamp
                closest_entry = None
                min_diff = float('inf')
                
                for entry in dataset_info:
                    if 'unix_timestamp' in entry:
                        diff = abs(entry['unix_timestamp'] - current_ts)
                        if diff < min_diff:
                            min_diff = diff
                            closest_entry = entry
                            
                # Only use if within 5 minutes (300 seconds)
                if closest_entry and min_diff <= 300:
                    matched_metadata = closest_entry
        
        if matched_metadata:
            ground_truth = float(matched_metadata[TARGET_VARIABLE])
            if ground_truth != 0:  # Avoid division by zero
                percent_change = (prediction - ground_truth) / abs(ground_truth) * 100

    return PredictResponse(
        session=params.session,
        used_model=os.path.basename(model_path),
        model_name=params.backbone,
        target_variable=TARGET_VARIABLE,
        predictions=[{
            "frame": 1,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "percent_change": percent_change,
            "timestamp": parse_timestamp(seq_files[-1]).isoformat(),
        }],
        frames_used=len(seq_files)
    )

# ─────────────────── Multiple Prediction Endpoint ───────────────────
@app.get("/predict_multiple", response_model=PredictResponse)
async def predict_multiple(params: PredictParams = Depends()):
    # 1) Resolve model path
    try:
        model_path = params.model or find_latest_model(params.session)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 2) Gather snapshots
    snaps = list_snapshots(HEATMAP_DIR)
    if not snaps:
        raise HTTPException(404, "No snapshots found")

    # 3) Determine end index
    if params.end_file:
        matches = [
            i for i,f in enumerate(snaps)
            if params.end_file in os.path.basename(f)
            or params.end_file in parse_timestamp(os.path.basename(f)).isoformat()
        ]
        if not matches:
            raise HTTPException(404, f"end-file '{params.end_file}' not found")
        end_idx = matches[-1]
    else:
        end_idx = len(snaps) - 1

    # Load dataset info if not already loaded
    dataset_info = DATASET_INFO
    if not dataset_info and params.session != SESSION_NAME:
        dataset_info = load_dataset_info(params.session)

    # Make predictions for multiple timeframes
    predictions = []
    
    # For each potential end point (going backwards from the end frame)
    for i in range(min(params.frames, end_idx + 1)):
        current_end_idx = end_idx - i
        current_start_idx = max(0, current_end_idx - params.frames + 1)
        
        # Skip if we don't have enough frames
        if current_end_idx - current_start_idx + 1 < 2:  # Need at least 2 frames for LSTM
            continue
            
        current_seq_files = snaps[current_start_idx:current_end_idx+1]
        
        # Load & preprocess images for this sequence
        try:
            current_imgs = torch.stack([
                TF(Image.open(fp).convert("RGB")) for fp in current_seq_files
            ]).unsqueeze(0)  # [1,T,3,224,224]
            
            # Make prediction
            with torch.no_grad():
                prediction = MODEL(current_imgs).item()
                
            # Find ground truth for this timeframe if available
            ground_truth = None
            percent_change = None
            current_file = os.path.basename(current_seq_files[-1])
            
            if dataset_info:
                # Extract date and time from the filename to match with dataset_info
                timestamp_match = re.search(r"heatmap_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_", current_file)
                if timestamp_match:
                    date_str = timestamp_match.group(1).replace("-", "")  # Convert 2025-04-01 to 20250401
                    time_str = timestamp_match.group(2)  # Keep 03-50-56 format
                    
                    # Try matching on timestamp components
                    matched_metadata = next(
                        (entry for entry in dataset_info 
                         if date_str in entry['new_filename'] and time_str in entry['new_filename']),
                        None
                    )
                    
                    if not matched_metadata:
                        # Fallback: try to extract timestamp and match by closest timestamp
                        dt_str = f"{timestamp_match.group(1)} {timestamp_match.group(2).replace('-',':')}"
                        current_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                        current_ts = current_dt.timestamp()
                        
                        # Find closest entry by timestamp
                        closest_entry = None
                        min_diff = float('inf')
                        
                        for entry in dataset_info:
                            if 'unix_timestamp' in entry:
                                diff = abs(entry['unix_timestamp'] - current_ts)
                                if diff < min_diff:
                                    min_diff = diff
                                    closest_entry = entry
                                    
                        # Only use if within 5 minutes (300 seconds)
                        if closest_entry and min_diff <= 300:
                            matched_metadata = closest_entry
                
                if matched_metadata:
                    ground_truth = float(matched_metadata[TARGET_VARIABLE])
                    if ground_truth != 0:  # Avoid division by zero
                        percent_change = (prediction - ground_truth) / abs(ground_truth) * 100
                    print(f"Matched {current_file} to {matched_metadata['new_filename']} with {ground_truth}")
            
            predictions.append({
                "frame": i+1,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "percent_change": percent_change,
                "timestamp": parse_timestamp(current_seq_files[-1]).isoformat(),
                "images_used": len(current_seq_files)
            })
            
        except Exception as e:
            print(f"Error processing frame {i+1}: {str(e)}")
            continue

    if not predictions:
        raise HTTPException(400, "Could not generate any predictions")

    return PredictResponse(
        session=params.session,
        used_model=os.path.basename(model_path),
        model_name=params.backbone,
        target_variable=TARGET_VARIABLE,
        predictions=predictions,
        frames_used=params.frames
    )
    
# ─────────────────── Debug Endpoint ───────────────────
@app.get("/debug", response_model=DebugInfo)
async def debug_info(session: str = SESSION_NAME):
    """Debug endpoint to check dataset and file matching"""
    # Load dataset info
    dataset_info = DATASET_INFO
    if not dataset_info and session != SESSION_NAME:
        dataset_info = load_dataset_info(session)
    
    # Get snapshot filenames
    snaps = list_snapshots(HEATMAP_DIR)
    heatmap_files = [os.path.basename(f) for f in snaps[:5]]  # Just show first 5
    
    # Test matching logic
    match_results = []
    for idx, filename in enumerate(heatmap_files):
        timestamp_match = re.search(r"heatmap_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_", filename)
        if timestamp_match:
            date_str = timestamp_match.group(1).replace("-", "")
            time_str = timestamp_match.group(2)
            
            # Direct match
            direct_match = next(
                (entry['new_filename'] for entry in dataset_info 
                 if date_str in entry['new_filename'] and time_str in entry['new_filename']),
                None
            )
            
            # Timestamp match
            dt_str = f"{timestamp_match.group(1)} {timestamp_match.group(2).replace('-',':')}"
            current_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            current_ts = current_dt.timestamp()
            
            closest_entry = None
            min_diff = float('inf')
            for entry in dataset_info:
                if 'unix_timestamp' in entry:
                    diff = abs(entry['unix_timestamp'] - current_ts)
                    if diff < min_diff:
                        min_diff = diff
                        closest_entry = entry
            
            match_results.append({
                "filename": filename,
                "extracted_date": date_str,
                "extracted_time": time_str,
                "direct_match": direct_match,
                "closest_match": closest_entry['new_filename'] if closest_entry else None,
                "time_diff_seconds": min_diff if closest_entry else None
            })
    
    return DebugInfo(
        dataset_size=len(dataset_info) if dataset_info else 0,
        dataset_samples=dataset_info[:3] if dataset_info else [],
        heatmap_files=heatmap_files,
        match_results=match_results
    )