#!/usr/bin/env python3
"""
Preprocessor for Liquidity Heatmap Image Dataset

This script processes heatmap images stored in a source directory by:
  - Parsing filenames to extract date, time, currency, and price.
  - Interpolating missing price values (marked as 'NA') using linear interpolation
    based on the previous and next valid images.
  - Computing the percentage change (without a decimal point) relative to the previous valid price.
    The change is encoded as a string (e.g., "p0250" for a +2.50% change, "n0050" for a -0.50% change).
  - Copying images into a target session folder (e.g., "test1") under an "images" subfolder,
    with new filenames that include the timestamp, currency, price, and change.
  - Generating a JSON file (dataset_info.json) within the session folder that stores metadata for all processed images.

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
# Global parameters (can be modified directly)
# =============================================================================
DEFAULT_SOURCE_DIR = os.path.expanduser("~/liquidLapse/heatmap_snapshots")
DEFAULT_AI_PROCESS_DIR = os.path.expanduser("~/liquidLapse/ai_process")
DEFAULT_SESSION_NAME = "test1"

# Configure logging for detailed output
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
        price = None  # Mark as missing if conversion fails
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

def compute_price_change(current_price, previous_price):
    """
    Compute the percentage change between the current and previous prices.
    
    The change is formatted without a decimal point.
    Example: if the change is +2.50%, returns "p0250"; if -0.50%, returns "n0050".
    If the previous price is None, returns "na".
    
    Args:
        current_price (float): Current price value.
        previous_price (float): Previous price value.
        
    Returns:
        str: Formatted percentage change.
    """
    if previous_price is None:
        return "na"
    change = (current_price - previous_price) / previous_price * 100
    # Multiply by 100 and format as a 4-digit integer string without decimal point
    if change >= 0:
        return f"p{int(round(change * 100)):04d}"
    else:
        return f"n{int(round(abs(change) * 100)):04d}"

def process_images(source_dir, target_session_dir):
    """
    Process all images from the source directory:
      - Parse filenames.
      - Interpolate missing prices.
      - Compute price changes.
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
    
    # Interpolate missing prices
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

    # Compute price change relative to the previous valid price
    previous_price = None
    for info in images_info:
        if previous_price is not None:
            change_str = compute_price_change(info['price'], previous_price)
        else:
            change_str = "na"
        info['change'] = change_str
        if info['price'] is not None:
            previous_price = info['price']
    
    # Prepare the target "images" folder within the session folder
    images_target_dir = os.path.join(target_session_dir, "images")
    if os.path.exists(images_target_dir):
        shutil.rmtree(images_target_dir)
    os.makedirs(images_target_dir, exist_ok=True)
    
    # Copy images with new informative names and update metadata
    for info in images_info:
        new_filename = (
            f"{info['timestamp'].strftime('%Y%m%d_%H%M%S')}_"
            f"{info['currency']}_"
            f"{info['price']:.2f}_"
            f"{info['change']}"
        )
        # Remove dot from the price part by replacing the decimal point
        new_filename = new_filename.replace('.', '')
        target_filepath = os.path.join(images_target_dir, new_filename + ".png")
        shutil.copy2(info['original_filepath'], target_filepath)
        info['new_filename'] = new_filename
        info['target_filepath'] = target_filepath
    
    return images_info

def write_metadata_json(metadata, session_dir):
    """
    Write metadata to a JSON file in the session directory.
    
    The JSON file (dataset_info.json) contains information about each processed image.
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
    Main function to execute the preprocessing.
    
    Global parameters can be modified at the top of the script.
    Command-line arguments override these global defaults.
    
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
