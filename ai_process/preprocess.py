#!/usr/bin/env python3
"""
Preprocessor for Liquidity Heatmap Image Dataset with Future Prediction Features

This script processes heatmap images stored in a source directory by:
  - Parsing filenames to extract date, time, currency, and price.
  - Interpolating missing price values (marked as 'NA') via linear interpolation.
  - Computing the step change (relative to previous valid price) and configurable time-window-based historic change.
  - Computing flexible future prediction features: one-step-ahead, N-step, and time-window-based future price and change ratios, plus average change over the interval.
    If future data is unavailable for a given horizon, the corresponding fields are set to None.
  - Copying images into a target session folder under an "images" subfolder, renaming files with metadata (ID, timestamp, currency, price, encoded change).
  - Generating a JSON file (dataset_info.json) within the session folder containing metadata for all processed images, including future prediction fields.

Configuration for future predictions is centralized at the top of the script for easy tuning.

Usage:
    python preprocess.py \
      --session test1 \
      [--source_dir <path>] \
      [--ai_process_dir <path>] \
      [--future_horizons "label1:step=1;label2:window=86400;..."]

Author: Alireza Ghaderi
Date: 2025-05-11
"""

import os
import re
import glob
import json
import shutil
import argparse
import logging
from datetime import datetime

# =============================================================================
# Global parameters (modifiable here; command-line arguments override these)
# =============================================================================
DEFAULT_SOURCE_DIR = os.path.expanduser("~/liquidLapse/heatmap_snapshots")
DEFAULT_AI_PROCESS_DIR = os.path.expanduser("~/liquidLapse/ai_process")
DEFAULT_SESSION_NAME = "test1"

# -----------------------------------------------------------------------------
# Future prediction configuration:
# Define a list of horizons; each entry must have a unique 'label' and either:
#   - 'step_count': integer number of steps ahead
#   - 'window_seconds': integer seconds ahead
# Example:
# FUTURE_HORIZONS = [
#     {'label': 'one_step',   'step_count': 1},
#     {'label': 'future_1d',  'window_seconds': 86400},  # 1-day window
#     {'label': 'five_min',   'window_seconds': 300},
#     {'label': 'step3',      'step_count': 3},
# ]
FUTURE_HORIZONS = [
    {'label': 'one_step',   'step_count': 1},
    {'label': 'future_1h',  'window_seconds': 3600},  # Default 1-hour window
    {'label': 'future_4h',  'window_seconds': 14400},  # Default 4-hour window
    {'label': 'future_1d',  'window_seconds': 86400},  # Default 1-day window
]
# -----------------------------------------------------------------------------

# Logging configuration for detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# Helper Functions
# =============================================================================

def parse_filename(filename):
    """
    Parse the filename to extract date, time, currency, and price.
    Expected format: heatmap_YYYY-MM-DD_HH-MM-SS_CURRENCY-price.png
    """
    pattern = r'heatmap_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_([A-Z]+)-(.+?)\.png$'
    basename = os.path.basename(filename)
    match = re.match(pattern, basename)
    if not match:
        logging.warning(f"Filename {basename} does not match expected pattern.")
        return None
    date_str, time_str, currency, price_str = match.groups()
    try:
        price = float(price_str)
    except ValueError:
        price = None  # Mark missing prices as None
    timestamp = datetime.strptime(f"{date_str} {time_str.replace('-', ':')}", "%Y-%m-%d %H:%M:%S")
    return {
        'date_str': date_str,
        'time_str': time_str,
        'currency': currency,
        'price': price,
        'timestamp': timestamp
    }


def get_all_image_files(source_dir):
    """
    Return sorted list of all heatmap PNG files in source_dir.
    """
    return sorted(glob.glob(os.path.join(source_dir, "heatmap_*.png")))


def interpolate_price(index, images_info):
    """
    Linear interpolate missing price using nearest valid neighbors.
    """
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


def compute_change_info(current_price, previous_price):
    """
    Compute encoded and numeric percent change from previous_price.
    """
    if previous_price is None or current_price is None:
        return {'encoded': 'p0000', 'change_percent': 0.0}
    pct = round((current_price - previous_price) / previous_price * 100, 3)
    fmt = f"{abs(pct):06.3f}"  # zero-padded to fixed width
    code = ('p' if pct >= 0 else 'n') + fmt.replace('.', '')
    return {'encoded': code, 'change_percent': pct}


def compute_time_window_change(current_info, images_info, idx, window_seconds):
    """
    Compute percent change from price at least window_seconds before current timestamp.
    Returns 0.0 if no valid historic data.
    """
    now_ts = current_info['timestamp']
    for j in range(idx - 1, -1, -1):
        past = images_info[j]
        if (now_ts - past['timestamp']).total_seconds() >= window_seconds and past['price'] is not None:
            return round((current_info['price'] - past['price']) / past['price'] * 100, 3)
    return 0.0


def compute_future_info(current_index, images_info, step_count=None, window_seconds=None):
    """
    Compute future price & change at a given step_count ahead or window_seconds ahead,
    plus average change over the interval. If unavailable, returns (None, None, None).
    """
    cur = images_info[current_index]
    cur_price = cur['price']
    future_idx = None
    # Determine future index by step or time window
    if step_count is not None:
        idx = current_index + step_count
        if idx < len(images_info):
            future_idx = idx
    elif window_seconds is not None:
        for j in range(current_index + 1, len(images_info)):
            if (images_info[j]['timestamp'] - cur['timestamp']).total_seconds() >= window_seconds:
                future_idx = j
                break
    # If no future data, return nulls
    if future_idx is None:
        return None, None, None
    fut_price = images_info[future_idx]['price']
    if cur_price is None or fut_price is None:
        return None, None, None
    # Percent change
    pct_change = round((fut_price - cur_price) / cur_price * 100, 3)
    # Average percent change over each intermediate step
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


def process_images(source_dir, target_session_dir, future_horizons):
    """
    Main pipeline: parse, interpolate, compute changes & future features, copy, metadata.
    """
    files = get_all_image_files(source_dir)
    if not files:
        logging.error("No image files found in source directory.")
        return []
    images_info = []
    for fp in files:
        info = parse_filename(fp)
        if info:
            info['original_filepath'] = fp
            images_info.append(info)
    images_info.sort(key=lambda x: x['timestamp'])

    # Interpolate missing prices
    for i, inf in enumerate(images_info):
        if inf['price'] is None:
            interp = interpolate_price(i, images_info)
            inf['price'] = interp
            inf['interpolated'] = interp is not None
        else:
            inf['interpolated'] = False

    # Compute step change & historic window change
    prev_price = None
    for i, inf in enumerate(images_info):
        ch = compute_change_info(inf['price'], prev_price)
        inf['change'] = ch['encoded']
        inf['change_percent_step'] = ch['change_percent']
        # You can configure custom historic windows similarly by calling compute_time_window_change
        inf['change_percent_window'] = compute_time_window_change(inf, images_info, i, window_seconds=86400)
        prev_price = inf['price'] if inf['price'] is not None else prev_price

    # Compute future prediction features
    for i, inf in enumerate(images_info):
        for cfg in future_horizons:
            label = cfg['label']
            fut_price, fut_pct, avg_pct = compute_future_info(
                i, images_info,
                step_count=cfg.get('step_count'),
                window_seconds=cfg.get('window_seconds')
            )
            inf[f'future_{label}_price'] = fut_price
            inf[f'future_{label}_change_percent'] = fut_pct
            inf[f'avg_{label}_change_percent'] = avg_pct

    # Add UNIX timestamp
    for inf in images_info:
        inf['unix_timestamp'] = int(inf['timestamp'].timestamp())

    # Prepare output folder
    imgs_dir = os.path.join(target_session_dir, 'images')
    if os.path.exists(imgs_dir): shutil.rmtree(imgs_dir)
    os.makedirs(imgs_dir, exist_ok=True)

    metadata = []
    for idx, inf in enumerate(images_info, start=1):
        price_str = f"{inf['price']:.2f}".replace('.', '') if inf['price'] is not None else 'NA'
        new_name = (
            f"{idx}_{inf['timestamp'].strftime('%Y%m%d_%H%M%S')}_"
            f"{inf['currency']}_{price_str}_{inf['change']}"
        )
        dst = os.path.join(imgs_dir, new_name + '.png')
        shutil.copy2(inf['original_filepath'], dst)

        entry = {
            'id': idx,
            'original_filepath': inf['original_filepath'],
            'new_filename': new_name,
            'target_filepath': dst,
            'date_str': inf['date_str'],
            'time_str': inf['time_str'],
            'timestamp': inf['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            'unix_timestamp': inf['unix_timestamp'],
            'currency': inf['currency'],
            'price': inf['price'],
            'change': inf['change'],
            'change_percent_step': inf['change_percent_step'],
            'change_percent_window': inf['change_percent_window'],
            'interpolated': inf['interpolated']
        }
        # Include future fields
        for cfg in future_horizons:
            l = cfg['label']
            entry[f'future_{l}_price'] = inf.get(f'future_{l}_price')
            entry[f'future_{l}_change_percent'] = inf.get(f'future_{l}_change_percent')
            entry[f'avg_{l}_change_percent'] = inf.get(f'avg_{l}_change_percent')
        metadata.append(entry)

    return metadata


def write_metadata_json(metadata, session_dir):
    """
    Write metadata list to dataset_info.json in session_dir.
    """
    path = os.path.join(session_dir, 'dataset_info.json')
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=4, default=str)
    logging.info(f"Wrote metadata JSON to {path}")


def parse_future_horizons(arg_str):
    """
    Parse CLI string like "lab1:step=1;lab2:window=86400" into list of configs.
    Defaults to FUTURE_HORIZONS if unspecified.
    """
    if not arg_str:
        return FUTURE_HORIZONS
    out = []
    for part in arg_str.split(';'):
        label, spec = part.split(':', 1)
        key, val = spec.split('=', 1)
        if key == 'step':
            out.append({'label': label, 'step_count': int(val)})
        elif key == 'window':
            out.append({'label': label, 'window_seconds': int(val)})
    return out


def main(args):
    source_dir = os.path.expanduser(args.source_dir) if args.source_dir else DEFAULT_SOURCE_DIR
    ai_dir = os.path.expanduser(args.ai_process_dir) if args.ai_process_dir else DEFAULT_AI_PROCESS_DIR
    session = args.session if args.session else DEFAULT_SESSION_NAME
    futures = parse_future_horizons(args.future_horizons)
    target_dir = os.path.join(ai_dir, session)
    if os.path.exists(target_dir): shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    logging.info(f"Processing {source_dir} into {target_dir} with future horizons: {futures}")
    metadata = process_images(source_dir, target_dir, futures)
    write_metadata_json(metadata, target_dir)
    logging.info("Preprocessing with future prediction features completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess heatmap images with flexible future prediction features."
    )
    parser.add_argument('--source_dir', type=str, help='Path to heatmap images source directory')
    parser.add_argument('--ai_process_dir', type=str, help='Base output directory for AI processing')
    parser.add_argument('--session', type=str, help='Session name (subfolder)')
    parser.add_argument('--future_horizons', type=str,
                        help='Future horizons spec, e.g.: "one:step=1;day:window=86400"')
    args = parser.parse_args()
    main(args)
