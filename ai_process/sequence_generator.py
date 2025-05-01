import os
import json
from datetime import datetime, timedelta
from shutil import copyfile

# Parameters
session_name = "test1"
base_directory = os.path.expanduser("~/liquidLapse")
data_source_path = os.path.join(base_directory, "heatmap_snapshots")
sequence_length = 10  # Number of images per sequence
sequence_folder = os.path.join(base_directory, f"ai_process/{session_name}/sequences")
sequences_json_path = os.path.join(sequence_folder, "sequences_info.json")

# Function to read dataset information
def read_dataset_info(dataset_file):
    dataset_file_path = os.path.join(base_directory, dataset_file)
    with open(dataset_file_path, 'r') as f:
        dataset_info = json.load(f)
    
    # Convert timestamp strings to datetime objects
    for entry in dataset_info:
        entry['timestamp'] = datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S')
    
    return dataset_info

# Function to handle gaps in timestamps
def handle_gaps(dataset_info):
    timestamps = [entry['timestamp'] for entry in dataset_info]
    timestamps.sort()  # Ensure timestamps are in chronological order
    
    start_time = timestamps[0]
    end_time = timestamps[-1]
    
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
                'future_action': None  # Placeholder for future action
            })
    
    dataset_info.sort(key=lambda x: x['timestamp'])  # Sort by timestamp
    return dataset_info

# Function to generate sequences and update sequences_info.json
def generate_sequences(dataset_info):
    if not os.path.exists(sequence_folder):
        os.makedirs(sequence_folder)
    
    sequences_info = []
    sequence_counter = 1
    
    for i in range(0, len(dataset_info), sequence_length):
        sequence_entries = dataset_info[i:i+sequence_length]
        
        # Determine future action for the sequence
        future_action = None
        if i + sequence_length < len(dataset_info):
            future_action = dataset_info[i + sequence_length]['change_percent_step']
        
        # Generate sequence folder
        sequence_folder_path = os.path.join(sequence_folder, f"sequence_{sequence_counter}")
        if not os.path.exists(sequence_folder_path):
            os.makedirs(sequence_folder_path)
        
        # Copy images and update sequences_info
        for data_entry in sequence_entries:
            filename = data_entry['new_filename']
            original_filepath = data_entry['original_filepath']
            target_filepath = os.path.join(sequence_folder_path, f"{filename}.png")
            
            copyfile(original_filepath, target_filepath)
            
            # Add entry to sequences_info
            sequences_info.append({
                'id': data_entry['id'],
                'start_time': data_entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': (data_entry['timestamp'] + timedelta(minutes=(sequence_length-1)*5)).strftime('%Y-%m-%d %H:%M:%S'),
                'items': data_entry['new_filename'],
                'prices': data_entry['price'],
                'path_to_image': target_filepath,
                'session_folder_paths': sequence_folder_path,
                'future_action': future_action
            })
        
        sequence_counter += 1
    
    # Write sequences_info to sequences_json_path
    with open(sequences_json_path, 'w') as f:
        json.dump(sequences_info, f, indent=4)

# Main script execution
if __name__ == "__main__":
    dataset_info = read_dataset_info(os.path.join("ai_process",session_name, "dataset_info.json"))
    dataset_info = handle_gaps(dataset_info)
    generate_sequences(dataset_info)
