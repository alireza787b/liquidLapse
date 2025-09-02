"""
Enhanced Preprocessor for Liquidity Heatmap Dataset - Production Version

Critical improvements:
- Robust price parsing with multiple formats
- Intelligent future horizon validation  
- Data quality checks and validation
- Gap detection and handling
- Professional error reporting

Author: Enhanced for liquidLapse production
Date: 2025-09-01
"""

import os
import re
import glob
import json
import shutil
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Enhanced logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# Configuration (Auto-adjusted based on data analysis)
# =============================================================================
DEFAULT_SOURCE_DIR = os.path.expanduser("~/liquidLapse/heatmap_snapshots")
DEFAULT_AI_PROCESS_DIR = os.path.expanduser("~/liquidLapse/ai_process")
DEFAULT_SESSION_NAME = "test1"

# Intelligent future horizons (will be validated against actual data)
FUTURE_HORIZONS = [
    {'label': 'one_step',   'step_count': 1},
    {'label': 'future_30m', 'window_seconds': 1800},   # 30 minutes
    {'label': 'future_1h',  'window_seconds': 3600},   # 1 hour
    {'label': 'future_4h',  'window_seconds': 14400},  # 4 hours
]

# Data quality thresholds
MIN_IMAGES_REQUIRED = 100      # Minimum images for meaningful training
MAX_TIME_GAP_MINUTES = 30      # Maximum acceptable gap between captures
MIN_FUTURE_COVERAGE = 0.7      # Minimum 70% of images must have future targets

class DataQualityValidator:
    """Professional data quality validation"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        
    def add_issue(self, message: str):
        self.issues.append(message)
        logging.error(f"DATA ISSUE: {message}")
        
    def add_warning(self, message: str):
        self.warnings.append(message)
        logging.warning(f"DATA WARNING: {message}")
        
    def validate_dataset(self, images_info: List[Dict]) -> bool:
        """Comprehensive dataset validation"""
        if len(images_info) < MIN_IMAGES_REQUIRED:
            self.add_issue(f"Insufficient data: {len(images_info)} images (minimum: {MIN_IMAGES_REQUIRED})")
            
        # Check time gaps
        gaps = []
        for i in range(1, len(images_info)):
            time_diff = (images_info[i]['timestamp'] - images_info[i-1]['timestamp']).total_seconds() / 60
            if time_diff > MAX_TIME_GAP_MINUTES:
                gaps.append((i-1, i, time_diff))
        
        if gaps:
            self.add_warning(f"Found {len(gaps)} time gaps > {MAX_TIME_GAP_MINUTES} minutes")
            for start_idx, end_idx, minutes in gaps[:5]:  # Show first 5
                logging.warning(f"  Gap: {minutes:.1f}min between images {start_idx}-{end_idx}")
                
        # Check price data quality
        null_prices = sum(1 for img in images_info if img['price'] is None)
        if null_prices > len(images_info) * 0.1:  # >10% null prices
            self.add_warning(f"High null price rate: {null_prices}/{len(images_info)} ({null_prices/len(images_info)*100:.1f}%)")
            
        return len(self.issues) == 0

def parse_filename_robust(filename: str) -> Optional[Dict]:
    """
    Robust filename parsing with multiple format support
    """
    pattern = r'heatmap_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_([A-Z]+)-(.+?)\.png$'
    basename = os.path.basename(filename)
    match = re.match(pattern, basename)
    
    if not match:
        logging.warning(f"Filename {basename} does not match expected pattern")
        return None
        
    date_str, time_str, currency, price_str = match.groups()
    
    try:
        timestamp = datetime.strptime(f"{date_str} {time_str.replace('-', ':')}", "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        logging.error(f"Invalid timestamp in {basename}: {e}")
        return None
    
    # Robust price parsing
    price = parse_price_robust(price_str)
    
    return {
        'date_str': date_str,
        'time_str': time_str,
        'currency': currency,
        'price': price,
        'timestamp': timestamp,
        'original_price_str': price_str  # Keep original for debugging
    }

def parse_price_robust(price_str: str) -> Optional[float]:
    """
    Robust price parsing supporting multiple formats:
    - "67000.12"
    - "67,000.12" 
    - "67000.12USD"
    - "NA" (missing)
    """
    if not price_str or price_str.upper() in ['NA', 'NULL', 'NONE']:
        return None
        
    # Clean the price string
    cleaned = re.sub(r'[,$A-Za-z\s]', '', price_str)
    
    try:
        return float(cleaned)
    except ValueError:
        logging.warning(f"Could not parse price: '{price_str}' -> '{cleaned}'")
        return None

def analyze_capture_interval(images_info: List[Dict]) -> float:
    """Analyze actual capture interval from data"""
    if len(images_info) < 2:
        return 300.0  # Default 5 minutes
        
    intervals = []
    for i in range(1, min(len(images_info), 20)):  # Sample first 20 intervals
        diff = (images_info[i]['timestamp'] - images_info[i-1]['timestamp']).total_seconds()
        intervals.append(diff)
    
    avg_interval = sum(intervals) / len(intervals)
    logging.info(f"Detected average capture interval: {avg_interval:.0f} seconds ({avg_interval/60:.1f} minutes)")
    
    return avg_interval

def validate_future_horizons(images_info: List[Dict], future_horizons: List[Dict], capture_interval: float) -> List[Dict]:
    """
    Validate and adjust future horizons based on actual data availability
    """
    total_span = (images_info[-1]['timestamp'] - images_info[0]['timestamp']).total_seconds()
    
    validated_horizons = []
    
    for horizon in future_horizons:
        if 'step_count' in horizon:
            # Step-based horizon
            max_steps_available = len(images_info) - 1
            if horizon['step_count'] <= max_steps_available:
                validated_horizons.append(horizon)
                logging.info(f"✓ Future horizon '{horizon['label']}': {horizon['step_count']} steps")
            else:
                logging.warning(f"✗ Skipping '{horizon['label']}': needs {horizon['step_count']} steps, only {max_steps_available} available")
                
        elif 'window_seconds' in horizon:
            # Time-based horizon
            if horizon['window_seconds'] < total_span:
                # Check how many images would have this target available
                coverage = 0
                for i in range(len(images_info)):
                    future_time = images_info[i]['timestamp'] + timedelta(seconds=horizon['window_seconds'])
                    if any(abs((img['timestamp'] - future_time).total_seconds()) < capture_interval for img in images_info[i+1:]):
                        coverage += 1
                
                coverage_ratio = coverage / len(images_info)
                
                if coverage_ratio >= MIN_FUTURE_COVERAGE:
                    validated_horizons.append(horizon)
                    logging.info(f"✓ Future horizon '{horizon['label']}': {horizon['window_seconds']}s ({coverage_ratio*100:.1f}% coverage)")
                else:
                    logging.warning(f"✗ Skipping '{horizon['label']}': only {coverage_ratio*100:.1f}% coverage (minimum: {MIN_FUTURE_COVERAGE*100:.1f}%)")
            else:
                logging.warning(f"✗ Skipping '{horizon['label']}': needs {horizon['window_seconds']}s, only {total_span:.0f}s available")
    
    return validated_horizons

def process_images_enhanced(source_dir: str, target_session_dir: str, future_horizons: List[Dict]) -> List[Dict]:
    """
    Enhanced image processing with validation and quality checks
    """
    # Get all files
    files = sorted(glob.glob(os.path.join(source_dir, "heatmap_*.png")))
    if not files:
        raise ValueError("No heatmap files found in source directory")
    
    logging.info(f"Found {len(files)} heatmap files")
    
    # Parse filenames
    images_info = []
    failed_parses = 0
    
    for filepath in files:
        info = parse_filename_robust(filepath)
        if info:
            info['original_filepath'] = filepath
            images_info.append(info)
        else:
            failed_parses += 1
    
    if failed_parses > 0:
        logging.warning(f"Failed to parse {failed_parses} filenames")
    
    if not images_info:
        raise ValueError("No valid heatmap files could be parsed")
    
    # Sort by timestamp
    images_info.sort(key=lambda x: x['timestamp'])
    
    # Data quality validation
    validator = DataQualityValidator()
    if not validator.validate_dataset(images_info):
        raise ValueError(f"Data quality validation failed: {validator.issues}")
    
    # Analyze capture interval
    capture_interval = analyze_capture_interval(images_info)
    
    # Validate and adjust future horizons
    validated_horizons = validate_future_horizons(images_info, future_horizons, capture_interval)
    
    if not validated_horizons:
        raise ValueError("No valid future horizons available for this dataset")
    
    # Interpolate missing prices
    interpolated_count = 0
    for i, info in enumerate(images_info):
        if info['price'] is None:
            interpolated = interpolate_price(i, images_info)
            if interpolated is not None:
                info['price'] = interpolated
                info['interpolated'] = True
                interpolated_count += 1
            else:
                info['interpolated'] = False
        else:
            info['interpolated'] = False
    
    logging.info(f"Interpolated {interpolated_count} missing prices")
    
    # Compute changes and future targets
    prev_price = None
    for i, info in enumerate(images_info):
        # Step change
        if prev_price is not None and info['price'] is not None:
            info['change_percent_step'] = round((info['price'] - prev_price) / prev_price * 100, 3)
        else:
            info['change_percent_step'] = 0.0
        
        # Encode change
        change_pct = info['change_percent_step']
        fmt = f"{abs(change_pct):06.3f}"
        info['change'] = ('p' if change_pct >= 0 else 'n') + fmt.replace('.', '')
        
        # Window-based historic change (24 hours)
        info['change_percent_window'] = compute_time_window_change(info, images_info, i, 86400)
        
        # Future predictions for validated horizons
        for horizon in validated_horizons:
            label = horizon['label']
            fut_price, fut_pct, avg_pct = compute_future_info(
                i, images_info, 
                step_count=horizon.get('step_count'),
                window_seconds=horizon.get('window_seconds')
            )
            info[f'future_{label}_price'] = fut_price
            info[f'future_{label}_change_percent'] = fut_pct
            info[f'avg_{label}_change_percent'] = avg_pct
        
        # Update previous price
        if info['price'] is not None:
            prev_price = info['price']
    
    # Add UNIX timestamps
    for info in images_info:
        info['unix_timestamp'] = int(info['timestamp'].timestamp())
    
    # Create output structure
    imgs_dir = os.path.join(target_session_dir, 'images')
    if os.path.exists(imgs_dir):
        shutil.rmtree(imgs_dir)
    os.makedirs(imgs_dir, exist_ok=True)
    
    # Process and copy files
    metadata = []
    for idx, info in enumerate(images_info, start=1):
        price_str = f"{info['price']:.2f}".replace('.', '') if info['price'] is not None else 'NA'
        new_name = (
            f"{idx:06d}_{info['timestamp'].strftime('%Y%m%d_%H%M%S')}_"
            f"{info['currency']}_{price_str}_{info['change']}"
        )
        
        dst_path = os.path.join(imgs_dir, new_name + '.png')
        shutil.copy2(info['original_filepath'], dst_path)
        
        # Create metadata entry
        entry = {
            'id': idx,
            'original_filepath': info['original_filepath'],
            'new_filename': new_name,
            'target_filepath': dst_path,
            'date_str': info['date_str'],
            'time_str': info['time_str'],
            'timestamp': info['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            'unix_timestamp': info['unix_timestamp'],
            'currency': info['currency'],
            'price': info['price'],
            'change': info['change'],
            'change_percent_step': info['change_percent_step'],
            'change_percent_window': info['change_percent_window'],
            'interpolated': info['interpolated']
        }
        
        # Add future fields for validated horizons only
        for horizon in validated_horizons:
            label = horizon['label']
            entry[f'future_{label}_price'] = info.get(f'future_{label}_price')
            entry[f'future_{label}_change_percent'] = info.get(f'future_{label}_change_percent')
            entry[f'avg_{label}_change_percent'] = info.get(f'avg_{label}_change_percent')
        
        metadata.append(entry)
    
    # Summary statistics
    valid_targets = {}
    for horizon in validated_horizons:
        label = horizon['label']
        count = sum(1 for entry in metadata if entry.get(f'future_{label}_change_percent') is not None)
        valid_targets[label] = count
        logging.info(f"Future target '{label}': {count}/{len(metadata)} ({count/len(metadata)*100:.1f}%) valid")
    
    return metadata

# Keep original helper functions (interpolate_price, compute_time_window_change, compute_future_info)
def interpolate_price(index, images_info):
    """Linear interpolate missing price using nearest valid neighbors."""
    prev_i = index - 1
    while prev_i >= 0 and images_info[prev_i]['price'] is None:
        prev_i -= 1
    next_i = index + 1
    while next_i < len(images_info) and images_info[next_i]['price'] is None:
        next_i += 1
    if prev_i < 0 and next_i < len(images_info):
        return images_info[next_i]['price']
    if prev_i < 0 or next_i >= len(images_info):
        return None
    prev_info, next_info = images_info[prev_i], images_info[next_i]
    total_secs = (next_info['timestamp'] - prev_info['timestamp']).total_seconds()
    if total_secs == 0:
        return prev_info['price']
    frac = (images_info[index]['timestamp'] - prev_info['timestamp']).total_seconds() / total_secs
    return prev_info['price'] + (next_info['price'] - prev_info['price']) * frac

def compute_time_window_change(current_info, images_info, idx, window_seconds):
    """Compute percent change from price at least window_seconds before current timestamp."""
    now_ts = current_info['timestamp']
    for j in range(idx - 1, -1, -1):
        past = images_info[j]
        if (now_ts - past['timestamp']).total_seconds() >= window_seconds and past['price'] is not None:
            return round((current_info['price'] - past['price']) / past['price'] * 100, 3)
    return 0.0

def compute_future_info(current_index, images_info, step_count=None, window_seconds=None):
    """Compute future price & change at given step_count or window_seconds ahead."""
    cur = images_info[current_index]
    cur_price = cur['price']
    future_idx = None
    
    if step_count is not None:
        idx = current_index + step_count
        if idx < len(images_info):
            future_idx = idx
    elif window_seconds is not None:
        for j in range(current_index + 1, len(images_info)):
            if (images_info[j]['timestamp'] - cur['timestamp']).total_seconds() >= window_seconds:
                future_idx = j
                break
    
    if future_idx is None:
        return None, None, None
        
    fut_price = images_info[future_idx]['price']
    if cur_price is None or fut_price is None:
        return None, None, None
        
    pct_change = round((fut_price - cur_price) / cur_price * 100, 3)
    
    # Average change over interval
    total_pct, count = 0.0, 0
    for k in range(current_index + 1, future_idx + 1):
        prev_p = images_info[k - 1]['price']
        next_p = images_info[k]['price']
        if prev_p and next_p:
            step_pct = (next_p - prev_p) / prev_p * 100
            total_pct += step_pct
            count += 1
    
    avg_pct = round(total_pct / count, 3) if count > 0 else pct_change
    
    return fut_price, pct_change, avg_pct

def main(args):
    source_dir = os.path.expanduser(args.source_dir or DEFAULT_SOURCE_DIR)
    ai_dir = os.path.expanduser(args.ai_process_dir or DEFAULT_AI_PROCESS_DIR)
    session = args.session or DEFAULT_SESSION_NAME
    
    target_dir = os.path.join(ai_dir, session)
    
    logging.info(f"Starting enhanced preprocessing...")
    logging.info(f"Source: {source_dir}")
    logging.info(f"Target: {target_dir}")
    logging.info(f"Session: {session}")
    
    # Clean target directory
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        # Process with validation
        metadata = process_images_enhanced(source_dir, target_dir, FUTURE_HORIZONS)
        
        # Write metadata
        metadata_path = os.path.join(target_dir, 'dataset_info.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        
        logging.info(f"✓ Processing completed successfully")
        logging.info(f"✓ Processed {len(metadata)} images")
        logging.info(f"✓ Metadata saved: {metadata_path}")
        
    except Exception as e:
        logging.error(f"✗ Processing failed: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enhanced preprocess with validation")
    parser.add_argument('--source_dir', type=str, help='Source directory')
    parser.add_argument('--ai_process_dir', type=str, help='AI process directory') 
    parser.add_argument('--session', type=str, help='Session name')
    args = parser.parse_args()
    main(args)
