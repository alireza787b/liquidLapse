#!/usr/bin/env python3
"""
Sequence Generator for Liquidity Heatmap Dataset

This script takes the preprocessed dataset JSON (with timestamps, prices, change features,
and built-in future-prediction fields) and builds clean, sliding-window sequences of metadata
for model training (e.g., RNNs, Transformers), ensuring no null entries.

Each sequence metadata includes:
  - Sequence-level fields: sequence_id, start_time, end_time, last_item_id, last_timestamp, last_price, and future-prediction labels from the last item.
  - An ordered list of items, each with index_in_sequence, id, timestamp, price, and image_path.

Features:
  - Configurable parameters at the top (paths, sequence length, save_images flag).
  - Sliding-window generation with overlap: sequences start at each possible offset where a full window exists.
  - Filters out any sequences containing invalid entries (missing id, price, or future labels).
  - Automatically detects future-prediction fields in the dataset JSON.
  - Optionally copies images into each sequence folder when `--save_images` is enabled; otherwise no directories are created.
  - Outputs a master JSON (`sequences_info.json`) containing only complete sequences.

Usage:
    python generate_sequences.py \
        --session test1 \
        [--base_dir ~/liquidLapse] \
        [--seq_len 110] \
        [--save_images]
"""

import os
import json
import argparse
from datetime import datetime
from shutil import copy2

# =============================================================================
# Configurable Parameters
# =============================================================================
DEFAULT_BASE_DIR       = os.path.expanduser("~/liquidLapse")
DEFAULT_SESSION_NAME   = "test1"

DATASET_JSON_REL       = "ai_process/{session}/dataset_info.json"
SEQ_FOLDER_REL         = "ai_process/{session}/sequences"
SEQUENCES_JSON_NAME    = "sequences_info.json"

DEFAULT_SEQ_LEN        = 100        # Number of timesteps per sequence
DEFAULT_SAVE_IMAGES    = False     # If False, no image folders or copies are created

# =============================================================================
# Helper Functions
# =============================================================================

def read_dataset_info(path):
    """Load JSON and convert timestamp strings to datetime objects."""
    with open(path, 'r') as f:
        data = json.load(f)
    for e in data:
        e['timestamp'] = datetime.strptime(e['timestamp'], "%Y-%m-%d %H:%M:%S")
    return data


def find_future_keys(sample):
    """Return list of future-prediction keys in a sample entry."""
    return [k for k in sample.keys() if k.startswith('future_') or k.startswith('avg_')]


def is_valid_entry(e, future_keys):
    """Check if entry has id, price, and all future keys non-null."""
    if e.get('id') is None or e.get('price') is None:
        return False
    for fk in future_keys:
        if e.get(fk) is None:
            return False
    return True


def make_sequences(data, seq_len, save_images, future_keys, output_dir):
    """
    Generate sliding-window sequences of length seq_len.
    Only include sequences where every item is valid and last item has future labels.

    Args:
        data (list): list of entries
        seq_len (int): window size
        save_images (bool): if True, create per-sequence folders and copy images
        future_keys (list): list of future label keys
        output_dir (str): base output directory for sequences
    Returns:
        list of sequence metadata dicts, (skipped_count, total_windows)
    """
    sequences = []
    skipped = 0
    total_windows = 0
    total = len(data)
    for start in range(0, total - seq_len + 1):
        total_windows += 1
        block = data[start:start + seq_len]
        if not all(is_valid_entry(item, future_keys) for item in block):
            skipped += 1
            continue
        last = block[-1]
        # Build sequence-level metadata
        seq_id = len(sequences) + 1
        seq_meta = {
            'sequence_id': seq_id,
            'start_time': block[0]['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            'end_time':   last['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            'last_item_id': last['id'],
            'last_timestamp': last['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            'last_price': last['price']
        }
        for fk in future_keys:
            seq_meta[fk] = last.get(fk)
        seq_meta['future_filename'] = last.get('new_filename')
        seq_meta['future_original_path'] = last.get('target_filepath')

        # Prepare sequence folders if needed
        if save_images:
            seq_folder = os.path.join(output_dir, f"seq_{seq_id:03d}")
            img_folder = os.path.join(seq_folder, 'images')
            os.makedirs(img_folder, exist_ok=True)
        else:
            seq_folder = None
            img_folder = None

        # Build items list
        items = []
        for idx, item in enumerate(block):
            img_src = item.get('target_filepath')
            if save_images and img_src and img_folder:
                try:
                    dst = os.path.join(img_folder, os.path.basename(img_src))
                    copy2(img_src, dst)
                    img_path = dst
                except Exception:
                    img_path = None
            else:
                img_path = img_src
            items.append({
                'index_in_sequence': idx,
                'id': item['id'],
                'timestamp': item['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                'price': item['price'],
                'image_path': img_path
            })
        seq_meta['items'] = items
        sequences.append(seq_meta)

    return sequences, skipped, total_windows

# =============================================================================
# Main
# =============================================================================

def main(args):
    base = os.path.expanduser(args.base_dir or DEFAULT_BASE_DIR)
    session = args.session or DEFAULT_SESSION_NAME
    src = os.path.join(base, DATASET_JSON_REL.format(session=session))
    out = os.path.join(base, SEQ_FOLDER_REL.format(session=session))
    seq_json = os.path.join(out, SEQUENCES_JSON_NAME)

    os.makedirs(out, exist_ok=True)

    data = read_dataset_info(src)
    if not data:
        print("[ERROR] No data loaded.")
        return
    future_keys = find_future_keys(data[0])

    sequences, skipped, total_windows = make_sequences(
        data=data,
        seq_len=args.seq_len,
        save_images=args.save_images,
        future_keys=future_keys,
        output_dir=out
    )

    # Write JSON
    with open(seq_json, 'w') as jf:
        json.dump(sequences, jf, indent=4)

    print(f"Total windows: {total_windows}")
    print(f"Sequences generated: {len(sequences)}")
    print(f"Sequences skipped: {skipped}")
    print(f"Metadata saved to {seq_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate clean sliding-window sequences from preprocessed heatmap dataset."
    )
    parser.add_argument('--base_dir',    type=str,
                        help=f"Base directory (default: {DEFAULT_BASE_DIR})")
    parser.add_argument('--session',     type=str,
                        help=f"Session name (default: {DEFAULT_SESSION_NAME})")
    parser.add_argument('--seq_len',     type=int,
                        help=f"Sequence length (default: {DEFAULT_SEQ_LEN})")
    parser.add_argument('--save_images', action='store_true',
                        help="Copy images into each sequence folder.")
    args = parser.parse_args()
    args.seq_len = args.seq_len or DEFAULT_SEQ_LEN
    main(args)