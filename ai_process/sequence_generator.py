import os
import json
from datetime import datetime, timedelta
from shutil import copyfile

# Parameters
session_name = "test1"
data_source_path = "../heatmap_snapshots"  # Adjust path as needed
sequence_length = 10  # Number of images per sequence
sequence_folder = f"ai_process/{session_name}/sequences"  # Path to save sequences

# Function to read dataset information
def read_dataset_info(dataset_file):
    with open(dataset_file, 'r') as f:
        dataset_info = json.load(f)
    return dataset_info

# Function to handle gaps in timestamps
def handle_gaps(dataset_info):
    timestamps = [entry['timestamp'] for entry in dataset_info]
    start_time = min(timestamps)
    end_time = max(timestamps)
    
    expected_times = [start_time + timedelta(minutes=i*5) for i in range(len(dataset_info))]
    
    # Check for gaps
    gaps = []
    for i in range(1, len(expected_times)):
        if expected_times[i] != timestamps[i]:
            gaps.append((expected_times[i-1], expected_times[i]))
    
    # Interpolate or pad gaps
    for gap_start, gap_end in gaps:
        # Example: Linear interpolation
        gap_duration = (gap_end - gap_start).total_seconds() / 60  # in minutes
        num_steps = int(gap_duration / 5)  # Assuming 5 minutes interval
        
        interpolated_times = [gap_start + timedelta(minutes=i*5) for i in range(1, num_steps)]
        
        # Insert interpolated entries into dataset_info
        for time in interpolated_times:
            dataset_info.append({
                'filename': None,  # Placeholder for missing filename
                'timestamp': time,
                'price': None  # Placeholder for missing price
            })
    
    # Sort dataset_info by timestamp again
    dataset_info.sort(key=lambda x: x['timestamp'])
    
    return dataset_info

# Function to generate sequences
def generate_sequences(dataset_info):
    num_sequences = len(dataset_info) // sequence_length
    
    for i in range(num_sequences):
        sequence_start = i * sequence_length
        sequence_end = sequence_start + sequence_length
        sequence_data = dataset_info[sequence_start:sequence_end]
        
        # Create sequence folder
        sequence_folder_path = os.path.join(sequence_folder, f"sequence_{i+1}")
        os.makedirs(sequence_folder_path, exist_ok=True)
        
        # Save images (copy from data_source_path to sequence_folder_path/images)
        for data_entry in sequence_data:
            filename = data_entry['filename']
            if filename:
                source_image_path = os.path.join(data_source_path, filename)
                target_image_path = os.path.join(sequence_folder_path, "images", filename)
                os.makedirs(os.path.dirname(target_image_path), exist_ok=True)
                copyfile(source_image_path, target_image_path)
        
        # Save metadata JSON
        metadata_file = os.path.join(sequence_folder_path, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(sequence_data, f, default=str, indent=4)

# Main function
def main():
    # Read dataset info
    dataset_info_file = os.path.join(DEFAULT_AI_PROCESS_DIR, session_name, "dataset_info.json")
    if not os.path.exists(dataset_info_file):
        raise FileNotFoundError(f"Dataset info file '{dataset_info_file}' not found.")
    
    dataset_info = read_dataset_info(dataset_info_file)
    
    # Handle gaps in timestamps
    dataset_info = handle_gaps(dataset_info)
    
    # Generate sequences
    generate_sequences(dataset_info)

if __name__ == "__main__":
    main()
