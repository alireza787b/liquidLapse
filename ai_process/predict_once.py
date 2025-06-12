#!/usr/bin/env python3
"""
Single prediction runner for liquidLapse.
Loads config, finds model, predicts, and appends result to session JSON.
"""

import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path
from glob import glob
import uuid
import yaml
import torch
from torchvision import models, transforms
from PIL import Image
import requests


# =============================================================================
# Global Configuration Defaults
# =============================================================================
DEFAULT_BASE_DIR = os.path.expanduser("~/liquidLapse")
DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_SNAPSHOT_DIR = "heatmap_snapshots"
DEFAULT_LOG_FILE = "prediction_service.log"
DEFAULT_STATUS_FILE = "prediction_service.status"
DEFAULT_PID_FILE = "prediction_service.pid"

# Service control flags
running = True
model_cache = None
last_model_path = None

# =============================================================================
# Utility Functions
# =============================================================================

def log_message(message, level="INFO"):
    """Enhanced logging with timestamps and levels"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    print(log_entry, flush=True)
    return log_entry

def ensure_directory(path, description="directory"):
    """Robust directory creation with detailed error reporting"""
    try:
        os.makedirs(path, exist_ok=True)
        log_message(f"Ensured {description}: {path}")
        return True
    except PermissionError:
        log_message(f"ERROR: Permission denied creating {description}: {path}", "ERROR")
        return False
    except OSError as e:
        log_message(f"ERROR: Failed to create {description} {path}: {e}", "ERROR")
        return False

def load_config(config_path):
    """Load and validate configuration with robust error handling"""
    try:
        if not os.path.exists(config_path):
            log_message(f"ERROR: Config file not found: {config_path}", "ERROR")
            return None
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate prediction config section
        if 'prediction' not in config:
            log_message("ERROR: 'prediction' section missing from config.yaml", "ERROR")
            return None
            
        pred_config = config['prediction']
        required_fields = ['enabled', 'prediction_interval', 'target_field']
        for field in required_fields:
            if field not in pred_config:
                log_message(f"ERROR: Required field '{field}' missing from prediction config", "ERROR")
                return None
        
        log_message("Configuration loaded successfully")
        return config
        
    except yaml.YAMLError as e:
        log_message(f"ERROR: Invalid YAML in config file: {e}", "ERROR")
        return None
    except Exception as e:
        log_message(f"ERROR: Failed to load config: {e}", "ERROR")
        return None

def setup_directories(base_dir, config):
    """Create all necessary directories for prediction service"""
    pred_config = config['prediction']
    prediction_folder = pred_config.get('prediction_folder', 'predictions')
    
    directories = {
        'base': base_dir,
        'ai_process': os.path.join(base_dir, 'ai_process'),
        'predictions': os.path.join(base_dir, 'ai_process', prediction_folder),
        'sessions': os.path.join(base_dir, 'ai_process', prediction_folder, 'sessions'),
        'snapshots': os.path.join(base_dir, DEFAULT_SNAPSHOT_DIR)
    }
    
    success = True
    for name, path in directories.items():
        if not ensure_directory(path, f"{name} directory"):
            success = False
    
    return success, directories

def signal_handler(signum, frame):
    """Graceful shutdown handler"""
    global running
    log_message(f"Received signal {signum}, initiating graceful shutdown...")
    running = False

def parse_snapshot_timestamp(filename):
    """Extract datetime from heatmap filename: heatmap_YYYY-MM-DD_HH-MM-SS_*.png"""
    import re
    match = re.match(r"heatmap_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_", filename)
    if not match:
        return None
    
    date_str = match.group(1)
    time_str = match.group(2).replace('-', ':')
    datetime_str = f"{date_str} {time_str}"
    
    try:
        return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None

def get_latest_snapshots(snapshot_dir, count):
    """Get the most recent N snapshots, sorted by timestamp"""
    try:
        if not os.path.exists(snapshot_dir):
            log_message(f"WARNING: Snapshot directory not found: {snapshot_dir}", "WARN")
            return []
        
        files = [f for f in os.listdir(snapshot_dir) 
                if f.startswith("heatmap_") and f.endswith(".png")]
        
        # Parse timestamps and sort
        timestamped_files = []
        for filename in files:
            timestamp = parse_snapshot_timestamp(filename)
            if timestamp:
                full_path = os.path.join(snapshot_dir, filename)
                timestamped_files.append((timestamp, full_path, filename))
        
        timestamped_files.sort(key=lambda x: x[0], reverse=True)
        return timestamped_files[:count]
        
    except Exception as e:
        log_message(f"ERROR: Failed to get latest snapshots: {e}", "ERROR")
        return []

def find_latest_model(base_dir, config=None):
    """Find the latest trained model checkpoint in the specified session"""
    try:
        ai_process_dir = os.path.join(base_dir, 'ai_process')

        # Always get session from config, default to 'test1'
        session = "test1"
        if config and "prediction" in config and "session" in config["prediction"]:
            session = config["prediction"]["session"] or "test1"

        if not os.path.exists(ai_process_dir):
            log_message("ERROR: ai_process directory not found", "ERROR")
            return None

        session_dir = os.path.join(ai_process_dir, session)
        if not os.path.exists(session_dir):
            log_message(f"ERROR: Session directory not found: {session_dir}", "ERROR")
            return None

        model_pattern = os.path.join(session_dir, 'train_*', 'best_model.pt')
        model_files = glob(model_pattern)

        if not model_files:
            log_message(f"ERROR: No trained models found in session: {session}", "ERROR")
            return None

        # Get the most recent model
        latest_model = max(model_files, key=os.path.getmtime)
        log_message(f"Found latest model: {latest_model}")

        return {
            'path': latest_model,
            'session': session,
            'train_dir': os.path.dirname(latest_model)
        }

    except Exception as e:
        log_message(f"ERROR: Failed to find latest model: {e}", "ERROR")
        return None

def get_btc_price(config):
    """Fetch current BTC price from configured API"""
    try:
        btc_url = config.get('btc_price_url', 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')
        response = requests.get(btc_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if 'bitcoin' in data and 'usd' in data['bitcoin']:
            price = data['bitcoin']['usd']
            log_message(f"Retrieved BTC price: ${price:,.2f}")
            return price
        else:
            log_message("WARNING: Unexpected BTC price API response format", "WARN")
            return None
            
    except requests.exceptions.RequestException as e:
        log_message(f"WARNING: Failed to fetch BTC price: {e}", "WARN")
        return None
    except Exception as e:
        log_message(f"WARNING: Error parsing BTC price: {e}", "WARN")
        return None

# =============================================================================
# Model Management
# =============================================================================

class CNN_LSTM(torch.nn.Module):
    """CNN-LSTM model definition (must match training)"""
    def __init__(self, backbone, lstm_hidden, lstm_layers, reg_dropout, freeze):
        super().__init__()
        cnn = getattr(models, backbone)(weights="DEFAULT")
        self.features = torch.nn.Sequential(*list(cnn.children())[:-1])
        feat_dim = cnn.fc.in_features
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
        self.lstm = torch.nn.LSTM(feat_dim, lstm_hidden, lstm_layers, batch_first=True)
        head = [torch.nn.Linear(lstm_hidden, lstm_hidden//2), torch.nn.ReLU()]
        if reg_dropout > 0:
            head.append(torch.nn.Dropout(reg_dropout))
        head.append(torch.nn.Linear(lstm_hidden//2, 1))
        self.regressor = torch.nn.Sequential(*head)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.features(x).view(B, T, -1)
        out, _ = self.lstm(feats)
        return self.regressor(out[:, -1, :]).squeeze(1)

def load_model(model_info, device, keep_in_memory=False):
    """Load model from checkpoint with caching option"""
    global model_cache, last_model_path
    
    model_path = model_info['path']
    
    # Check if we can use cached model
    if keep_in_memory and model_cache is not None and last_model_path == model_path:
        log_message("Using cached model")
        return model_cache, model_cache.checkpoint_config
    
    try:
        log_message(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'config' not in checkpoint or 'state_dict' not in checkpoint:
            log_message("ERROR: Invalid checkpoint format", "ERROR")
            return None, None
        
        config = checkpoint['config']
        
        # Reconstruct model
        model = CNN_LSTM(
            backbone=config['backbone'],
            lstm_hidden=config['lstm_hidden'],
            lstm_layers=config['lstm_layers'],
            reg_dropout=config['reg_dropout'],
            freeze=config['freeze']
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device).eval()
        
        # Cache if requested
        if keep_in_memory:
            model_cache = model
            model_cache.checkpoint_config = config
            last_model_path = model_path
        
        log_message("Model loaded successfully")
        return model, config
        
    except Exception as e:
        log_message(f"ERROR: Failed to load model: {e}", "ERROR")
        return None, None

# =============================================================================
# Prediction Pipeline
# =============================================================================

def create_transforms():
    """Create image transforms matching training pipeline"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def prepare_sequence(snapshots, seq_len, transforms):
    """Prepare image sequence for model input"""
    try:
        # Take the most recent seq_len snapshots
        if len(snapshots) < seq_len:
            log_message(f"WARNING: Only {len(snapshots)} snapshots available, need {seq_len}", "WARN")
        
        # Use the most recent images, pad with black if necessary
        used_snapshots = snapshots[:seq_len] if len(snapshots) >= seq_len else snapshots
        
        images = []
        snapshot_info = []
        
        for timestamp, path, filename in used_snapshots:
            try:
                img = Image.open(path).convert('RGB')
                images.append(transforms(img))
                snapshot_info.append({
                    'filename': filename,
                    'timestamp': timestamp.isoformat(),
                    'path': path
                })
            except Exception as e:
                log_message(f"WARNING: Failed to load image {filename}: {e}", "WARN")
                # Use black image as fallback
                img = Image.new('RGB', (224, 224))
                images.append(transforms(img))
                snapshot_info.append({
                    'filename': filename,
                    'timestamp': timestamp.isoformat() if timestamp else None,
                    'path': path,
                    'error': str(e)
                })
        
        # Pad with black images if not enough snapshots
        while len(images) < seq_len:
            img = Image.new('RGB', (224, 224))
            images.append(transforms(img))
            snapshot_info.append({
                'filename': 'padding_black.png',
                'timestamp': None,
                'path': None,
                'padded': True
            })
        
        # Stack into tensor [seq_len, 3, 224, 224] then add batch dim
        sequence_tensor = torch.stack(images).unsqueeze(0)  # [1, seq_len, 3, 224, 224]
        
        return sequence_tensor, snapshot_info
        
    except Exception as e:
        log_message(f"ERROR: Failed to prepare sequence: {e}", "ERROR")
        return None, None

def make_prediction(model, sequence_tensor, device):
    """Run model inference"""
    try:
        start_time = time.time()
        sequence_tensor = sequence_tensor.to(device)
        
        with torch.no_grad():
            prediction = model(sequence_tensor).item()
        
        processing_time = (time.time() - start_time) * 1000  # milliseconds
        
        log_message(f"Prediction: {prediction:.4f} (processed in {processing_time:.1f}ms)")
        return prediction, processing_time
        
    except Exception as e:
        log_message(f"ERROR: Failed to make prediction: {e}", "ERROR")
        return None, None

def create_prediction_record(model_info, model_config, snapshots, prediction, processing_time, btc_price, target_field, device):
    """Create compact prediction metadata record"""
    prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    # Only keep start/end timestamps
    sequence_start = snapshots[-1]['timestamp'] if snapshots else None
    sequence_end = snapshots[0]['timestamp'] if snapshots else None

    return {
        'prediction_id': prediction_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model_session': model_info['session'],
        'model_path': model_info['path'],
        'target_field': target_field,
        'sequence_start': sequence_start,
        'sequence_end': sequence_end,
        'btc_price': btc_price,
        'prediction': prediction,
        'processing_time_ms': processing_time,
        'device': str(device)
    }

def save_prediction(prediction_record, directories, session):
    """Save prediction record to JSON with proper organization"""
    try:
        # Create session-specific directory
        session_dir = os.path.join(directories['sessions'], session)
        ensure_directory(session_dir, f"session prediction directory")
        
        # Create daily prediction file
        today = datetime.now().strftime('%Y%m%d')
        prediction_file = os.path.join(session_dir, f'predictions_{today}.json')
        
        # Load existing predictions or create new list
        predictions = []
        if os.path.exists(prediction_file):
            try:
                with open(prediction_file, 'r', encoding='utf-8') as f:
                    predictions = json.load(f)
            except json.JSONDecodeError:
                log_message(f"WARNING: Corrupted prediction file, creating new: {prediction_file}", "WARN")
                predictions = []
        
        # Add new prediction
        predictions.append(prediction_record)
        
        # Save updated predictions
        with open(prediction_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        # Update session metadata
        metadata_file = os.path.join(session_dir, 'metadata.json')
        metadata = {
            'session': session,
            'last_prediction': prediction_record['timestamp'],
            'total_predictions': len(predictions),
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        log_message(f"Prediction saved: {prediction_record['prediction_id']}")
        return True
        
    except Exception as e:
        log_message(f"ERROR: Failed to save prediction: {e}", "ERROR")
        return False

# =============================================================================
# Main Service Loop
# =============================================================================

def prediction_cycle(config, directories, device):
    """Single prediction cycle"""
    try:
        pred_config = config['prediction']
        
        # Find latest model
        model_info = find_latest_model(directories['base'], pred_config.get('session', 'auto'))
        if not model_info:
            return False
        
        # Load model
        keep_in_memory = pred_config.get('keep_model_in_memory', False)
        model, model_config = load_model(model_info, device, keep_in_memory)
        if model is None:
            return False
        
        # Get required sequence length
        seq_len = model_config.get('seq_len', 12)
        log_message(f"Using sequence length: {seq_len}")
        
        # Get latest snapshots
        snapshots = get_latest_snapshots(directories['snapshots'], seq_len)
        if not snapshots:
            log_message("WARNING: No snapshots available for prediction", "WARN")
            return False
        
        # Prepare sequence
        transforms = create_transforms()
        sequence_tensor, snapshot_info = prepare_sequence(snapshots, seq_len, transforms)
        if sequence_tensor is None:
            return False
        
        # Make prediction
        prediction, processing_time = make_prediction(model, sequence_tensor, device)
        if prediction is None:
            return False
        
        # Get market context
        btc_price = get_btc_price(config)
        
        # Create prediction record
        target_field = pred_config.get('target_field', 'future_future_4h_change_percent')
        prediction_record = create_prediction_record(
            model_info, model_config, snapshot_info, prediction, 
            processing_time, btc_price, target_field, device
        )
        
        # Save prediction
        return save_prediction(prediction_record, directories, model_info['session'])
        
    except Exception as e:
        log_message(f"ERROR: Prediction cycle failed: {e}", "ERROR")
        log_message(f"Traceback: {traceback.format_exc()}", "DEBUG")
        return False

def write_status(status, pid_file, status_file):
    """Write service status and PID"""
    try:
        with open(pid_file, 'w') as f:
            f.write(str(os.getpid()))
        
        status_data = {
            'status': status,
            'pid': os.getpid(),
            'timestamp': datetime.now().isoformat(),
            'start_time': datetime.now().isoformat() if status == 'running' else None
        }
        
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
            
    except Exception as e:
        log_message(f"WARNING: Failed to write status: {e}", "WARN")
def main():
    parser = argparse.ArgumentParser(description="liquidLapse Single Prediction Runner")
    parser.add_argument('--base_dir', type=str, default=os.path.expanduser("~/liquidLapse"),
                        help="Base directory for liquidLapse project")
    parser.add_argument('--config', type=str, default="config.yaml",
                        help="Configuration file path")
    parser.add_argument('--device', type=str, default="auto",
                        help="Device for inference (auto, cpu, cuda:0)")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    config_path = os.path.join(base_dir, args.config) if not os.path.isabs(args.config) else args.config

    log_message("=" * 60)
    log_message("liquidLapse Single Prediction Runner")
    log_message(f"Base directory: {base_dir}")
    log_message(f"Config path: {config_path}")
    log_message("=" * 60)

    config = load_config(config_path)
    if not config:
        log_message("FATAL: Failed to load configuration", "ERROR")
        sys.exit(1)

    if not config['prediction'].get('enabled', False):
        log_message("Prediction service is disabled in configuration")
        sys.exit(0)

    # Setup directories
    success, directories = setup_directories(base_dir, config)
    if not success:
        log_message("FATAL: Failed to setup required directories", "ERROR")
        sys.exit(1)

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log_message(f"Using device: {device}")

    # --- SINGLE PREDICTION CYCLE ---
    try:
        pred_config = config['prediction']
        model_info = find_latest_model(directories['base'], config)
        if not model_info:
            sys.exit(1)

        keep_in_memory = pred_config.get('keep_model_in_memory', False)
        model, model_config = load_model(model_info, device, keep_in_memory)
        if model is None:
            sys.exit(1)

        seq_len = model_config.get('seq_len', 12)
        log_message(f"Using sequence length: {seq_len}")

        snapshots = get_latest_snapshots(directories['snapshots'], seq_len)
        if not snapshots:
            log_message("WARNING: No snapshots available for prediction", "WARN")
            sys.exit(1)

        transforms_ = create_transforms()
        sequence_tensor, snapshot_info = prepare_sequence(snapshots, seq_len, transforms_)
        if sequence_tensor is None:
            sys.exit(1)

        prediction, processing_time = make_prediction(model, sequence_tensor, device)
        if prediction is None:
            sys.exit(1)

        btc_price = get_btc_price(config)
        target_field = pred_config.get('target_field', 'future_future_4h_change_percent')
        prediction_record = create_prediction_record(
            model_info, model_config, snapshot_info, prediction,
            processing_time, btc_price, target_field, device
        )

        # Save prediction
        if save_prediction(prediction_record, directories, model_info['session']):
            log_message("Prediction saved successfully")
            sys.exit(0)
        else:
            log_message("Failed to save prediction", "ERROR")
            sys.exit(1)

    except Exception as e:
        log_message(f"ERROR: Prediction failed: {e}", "ERROR")
        log_message(f"Traceback: {traceback.format_exc()}", "DEBUG")
        sys.exit(1)

if __name__ == "__main__":
    main()