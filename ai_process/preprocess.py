#!/usr/bin/env python3
"""
Preprocessor for Liquidity Heatmap Image Dataset

This script processes heatmap images stored in a source directory by:
  - Parsing filenames to extract date, time, currency, and price.
  - Interpolating missing price values (marked as 'NA') using linear interpolation
    based on the previous and next valid images. For the first image, if missing,
    it uses the next valid price.
  - Computing the percentage change relative to the previous valid price ("step change")
    and also the percentage change over an hourly window ("hourly change"). 
    Both are calculated to three decimal places.
    The "step" change is encoded as a string without decimals (e.g., "p0342" for +0.342%),
    and stored as a numeric value in the field "change_percent_step".
    The "hourly" change is stored as a numeric value in the field "change_percent_hour".
    If no previous price exists, these default to 0.
  - Adding an incremental ID (starting at 1) to each image; this ID is included as the first field
    in the JSON metadata and prefixed in the new filename.
  - Copying images into a target session folder (e.g., "test1") under an "images" subfolder,
    with new filenames that include the ID, timestamp, currency, price (formatted without a decimal point),
    and encoded step change.
  - Generating a JSON file (dataset_info.json) within the session folder containing metadata for all processed images,
    including the UNIX timestamp.

If target folders or files already exist, they will be replaced.

Usage:
    python preprocess.py --session test1 [--source_dir <path>] [--ai_process_dir <path>]

Global parameters are defined below and can be modified directly; command-line arguments override these defaults.

Author: Alireza Ghaderi
Date: 2025-04-02
"""

import os
import re
import glob
import json
import shutil
import argparse
import logging
from datetime import datetime
from pathlib import Path

# =============================================================================
# Global parameters (modifiable here; command-line arguments override these)
# =============================================================================
DEFAULT_SOURCE_DIR = os.path.expanduser("~/liquidLapse/heatmap_snapshots")
DEFAULT_AI_PROCESS_DIR = os.path.expanduser("~/liquidLapse/ai_process")
DEFAULT_SESSION_NAME = "test1"
HOUR_WINDOW_SECONDS = 36000  # Configurable hourly window in seconds

# =============================================================================
# Logging configuration for detailed output
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# Helper Functions
# =============================================================================

def parse_filename(filename):
    """
    Parse the filename to extract date, time, currency, and price.
    
    Expected filename format:
      heatmap_YYYY-MM-DD_HH-MM-SS_CURRENCY-price.png
    where price may be numeric or 'NA'.
    
    Returns:
        dict: {
            'date_str': "YYYY-MM-DD",
            'time_str': "HH-MM-SS",
            'currency': e.g., "BTC",
            'price': float value if valid, else None,
            'timestamp': datetime object
        }
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
        price = None  # Mark as missing if conversion fails (e.g., "NA")
    dt_str = f"{date_str} {time_str.replace('-', ':')}"
    try:
        timestamp = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        logging.error(f"Error parsing datetime from {dt_str}: {e}")
        return None
    return {
        'date_str': date_str,
        'time_str': time_str,
        'currency': currency,
        'price': price,
        'timestamp': timestamp
    }

def get_all_image_files(source_dir):
    """
    Retrieve all heatmap image file paths from the source directory matching the pattern.
    
    Args:
        source_dir (str): Directory containing heatmap images.
        
    Returns:
        list: Sorted list of image file paths.
    """
    pattern = os.path.join(source_dir, "heatmap_*.png")
    return sorted(glob.glob(pattern))

def interpolate_price(index, images_info):
    """
    Interpolate a missing price for the image at a given index using linear interpolation.
    
    Searches for the closest previous and next images with valid prices.
    If there is no previous valid price (i.e., first image), returns the next valid price.
    
    Args:
        index (int): Index of the image with missing price.
        images_info (list): List of dictionaries containing parsed image info.
        
    Returns:
        float or None: Interpolated price or None if interpolation is not possible.
    """
    # Find previous valid price
    prev_index = index - 1
    while prev_index >= 0 and images_info[prev_index]['price'] is None:
        prev_index -= 1
    # Find next valid price
    next_index = index + 1
    while next_index < len(images_info) and images_info[next_index]['price'] is None:
        next_index += 1
    if prev_index < 0 and next_index < len(images_info):
        # No previous price: use next valid price
        return images_info[next_index]['price']
    if prev_index < 0 or next_index >= len(images_info):
        return None
    prev_info = images_info[prev_index]
    next_info = images_info[next_index]
    total_seconds = (next_info['timestamp'] - prev_info['timestamp']).total_seconds()
    if total_seconds == 0:
        return prev_info['price']
    elapsed = (images_info[index]['timestamp'] - prev_info['timestamp']).total_seconds()
    interpolated_price = prev_info['price'] + (next_info['price'] - prev_info['price']) * (elapsed / total_seconds)
    return interpolated_price

def compute_change_info(current_price, previous_price):
    """
    Compute the percentage change between current and previous prices.
    
    Returns:
        dict: {
            'encoded': A string without decimal points (e.g., "p0342" for +0.342% or "n0050" for -0.050%),
            'change_percent': A float representing the percentage change (e.g., 0.342 or -0.050)
        }
    If previous_price is None, defaults to 0 change.
    """
    if previous_price is None:
        return {"encoded": "p0000", "change_percent": 0.0}
    change_percent = round((current_price - previous_price) / previous_price * 100, 3)
    formatted = f"{abs(change_percent):06.3f}"  # e.g. "0.342"
    encoded = ("p" if change_percent >= 0 else "n") + formatted.replace(".", "")
    return {"encoded": encoded, "change_percent": change_percent}

def compute_hourly_change(current_info, images_info, current_index, window_seconds=HOUR_WINDOW_SECONDS):
    """
    Compute the hourly percentage change for the current image by finding the price 
    from at least 'window_seconds' (default 3600 seconds) earlier.
    
    Args:
        current_info (dict): Metadata for the current image.
        images_info (list): List of dictionaries for all images.
        current_index (int): Index of the current image.
        window_seconds (int): Time window in seconds (default: 3600 seconds for 1 hour).
        
    Returns:
        float: Hourly percentage change (0 if no earlier image exists).
    """
    current_timestamp = current_info['timestamp']
    # Initialize candidate to None
    candidate = None
    # Iterate backwards from current_index-1
    for i in range(current_index - 1, -1, -1):
        candidate_info = images_info[i]
        if (current_timestamp - candidate_info['timestamp']).total_seconds() >= window_seconds:
            candidate = candidate_info
            break
    if candidate is None or candidate['price'] is None:
        return 0.0
    hourly_change = round((current_info['price'] - candidate['price']) / candidate['price'] * 100, 3)
    return hourly_change

def process_images(source_dir, target_session_dir):
    """
    Process all images from the source directory:
      - Parse filenames.
      - Interpolate missing prices.
      - Compute step change (relative to previous valid price) and hourly change.
      - Copy images to the target session's "images" folder with new informative filenames.
      - Collect metadata for each image.
      
    Args:
        source_dir (str): Directory containing original heatmap images.
        target_session_dir (str): Session directory (e.g., liquidLapse/ai_process/test1).
        
    Returns:
        list: List of metadata dictionaries for each processed image.
    """
    image_files = get_all_image_files(source_dir)
    if not image_files:
        logging.error("No image files found in the source directory.")
        return []
    
    images_info = []
    for filepath in image_files:
        info = parse_filename(filepath)
        if info is not None:
            info['original_filepath'] = filepath
            images_info.append(info)
    
    # Sort images by timestamp
    images_info.sort(key=lambda x: x['timestamp'])
    
    # Interpolate missing prices where necessary
    for i, info in enumerate(images_info):
        if info['price'] is None:
            interp_price = interpolate_price(i, images_info)
            if interp_price is not None:
                logging.info(f"Interpolated price for {info['original_filepath']}: {interp_price:.2f}")
                info['price'] = interp_price
                info['interpolated'] = True
            else:
                logging.warning(f"Could not interpolate price for {info['original_filepath']}")
                info['interpolated'] = False
        else:
            info['interpolated'] = False

    # Compute step change (relative to previous valid price) and hourly change
    previous_price = None
    for i, info in enumerate(images_info):
        step_change_info = compute_change_info(info['price'], previous_price)
        info['change'] = step_change_info['encoded']
        info['change_percent_step'] = step_change_info['change_percent']
        # Compute hourly change using the current index
        info['change_percent_hour'] = compute_hourly_change(info, images_info, i, window_seconds=HOUR_WINDOW_SECONDS)
        if info['price'] is not None:
            previous_price = info['price']
    
    # Add UNIX timestamp for each image
    for info in images_info:
        info['unix_timestamp'] = int(info['timestamp'].timestamp())
    
    # Prepare the target "images" folder within the session folder
    images_target_dir = os.path.join(target_session_dir, "images")
    if os.path.exists(images_target_dir):
        shutil.rmtree(images_target_dir)
    os.makedirs(images_target_dir, exist_ok=True)
    
    # Copy images with new informative names including an incremental ID and update metadata.
    # The metadata for each image will have the 'id' field as the first field.
    new_metadata = []
    for idx, info in enumerate(images_info, start=1):
        # Format price: remove decimal point (e.g., 8163.10 becomes "816310")
        price_str = f"{info['price']:.2f}".replace('.', '')
        new_filename = (
            f"{idx}_{info['timestamp'].strftime('%Y%m%d_%H%M%S')}_"
            f"{info['currency']}_{price_str}_"
            f"{info['change']}"
        )
        target_filepath = os.path.join(images_target_dir, new_filename + ".png")
        shutil.copy2(info['original_filepath'], target_filepath)
        
        # Build metadata entry with id as the first field
        metadata_entry = {
            "id": idx,
            "original_filepath": info.get('original_filepath'),
            "new_filename": new_filename,
            "target_filepath": target_filepath,
            "date_str": info.get('date_str'),
            "time_str": info.get('time_str'),
            "timestamp": info.get('timestamp').strftime("%Y-%m-%d %H:%M:%S"),
            "unix_timestamp": info.get('unix_timestamp'),
            "currency": info.get('currency'),
            "price": info.get('price'),
            "change": info.get('change'),
            "change_percent_step": info.get('change_percent_step'),
            "change_percent_hour": info.get('change_percent_hour'),
            "interpolated": info.get('interpolated')
        }
        new_metadata.append(metadata_entry)
        # Update original info dictionary with id and new filename
        info['id'] = idx
        info['new_filename'] = new_filename
        info['target_filepath'] = target_filepath
    
    return new_metadata

def write_metadata_json(metadata, session_dir):
    """
    Write metadata to a JSON file in the session folder.
    
    The JSON file (dataset_info.json) contains detailed information for each processed image.
    If the file exists, it will be replaced.
    
    Args:
        metadata (list): List of metadata dictionaries.
        session_dir (str): Session folder where the JSON file will be saved.
    """
    json_filepath = os.path.join(session_dir, "dataset_info.json")
    with open(json_filepath, "w") as f:
        json.dump(metadata, f, indent=4, default=str)
    logging.info(f"Metadata JSON written to {json_filepath}")

# =============================================================================
# Main Function
# =============================================================================

def main(args):
    """
    Main function to execute the preprocessing pipeline.
    
    Global parameters can be modified at the top of the script.
    Command-line arguments override these defaults.
    
    Args:
        args: Parsed command-line arguments.
    """
    # Use command-line arguments if provided; otherwise use global defaults
    source_dir = os.path.expanduser(args.source_dir) if args.source_dir else DEFAULT_SOURCE_DIR
    ai_process_dir = os.path.expanduser(args.ai_process_dir) if args.ai_process_dir else DEFAULT_AI_PROCESS_DIR
    session_name = args.session if args.session else DEFAULT_SESSION_NAME

    # Define target session directory (e.g., liquidLapse/ai_process/test1)
    target_session_dir = os.path.join(ai_process_dir, session_name)
    if os.path.exists(target_session_dir):
        shutil.rmtree(target_session_dir)
    os.makedirs(target_session_dir, exist_ok=True)
    
    logging.info(f"Processing images from {source_dir} into session folder {target_session_dir}")
    
    # Process images and collect metadata
    metadata = process_images(source_dir, target_session_dir)
    
    # Write metadata JSON into the session folder
    write_metadata_json(metadata, target_session_dir)
    
    logging.info("Preprocessing completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess heatmap images for AI model training.")
    parser.add_argument("--source_dir", type=str, default=DEFAULT_SOURCE_DIR,
                        help="Directory containing original heatmap images. Default: %(default)s")
    parser.add_argument("--ai_process_dir", type=str, default=DEFAULT_AI_PROCESS_DIR,
                        help="Base directory for AI processing outputs. Default: %(default)s")
    parser.add_argument("--session", type=str, default=DEFAULT_SESSION_NAME,
                        help="Session name (subfolder within ai_process). Default: %(default)s")
    args = parser.parse_args()
    main(args)
