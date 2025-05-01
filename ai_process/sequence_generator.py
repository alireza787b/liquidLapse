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
                'id': None,        # Placeholder for ID or other necessary data
                # Add other required fields as needed
            })
    
    # Sort dataset_info again after interpolation
    dataset_info.sort(key=lambda x: x['timestamp'])
    
    return dataset_info

# Function to generate sequences and update JSON metadata
def generate_sequences(dataset_info):
    sequences_info = []
    num_sequences = len(dataset_info) // sequence_length
    
    for i in range(num_sequences):
        sequence_data = dataset_info[i * sequence_length : (i + 1) * sequence_length]
        
        # Generate sequence folder path
        sequence_folder_path = os.path.join(sequence_folder, f"sequence_{i + 1}")
        os.makedirs(sequence_folder_path, exist_ok=True)
        
        # Copy images to sequence folder
        for data_entry in sequence_data:
            original_filepath = data_entry['original_filepath']
            target_filepath = os.path.join(sequence_folder_path, os.path.basename(data_entry['target_filepath']))
            copyfile(original_filepath, target_filepath)
            
        # Calculate future action after the sequence
        future_action = None
        if (i + 1) * sequence_length < len(dataset_info):
            next_data_entry = dataset_info[(i + 1) * sequence_length]
            future_action = next_data_entry['change_percent_step']  # Adjust as per your data structure
        
        # Prepare sequence metadata
        sequence_info = {
            'id': i + 1,
            'start_time': sequence_data[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': sequence_data[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'items': [data_entry['filename'] for data_entry in sequence_data],
            'prices': [data_entry['price'] for data_entry in sequence_data],
            'path_to_images': [os.path.abspath(os.path.join(sequence_folder_path, os.path.basename(data_entry['target_filepath']))) for data_entry in sequence_data],
            'future_action': future_action,
            # Add any other required fields
        }
        
        sequences_info.append(sequence_info)
    
    # Write sequences metadata to JSON file
    with open(sequences_json_path, 'w') as f:
        json.dump(sequences_info, f, indent=4)

# Main script execution
if __name__ == "__main__":
    # Read dataset information
    dataset_info = read_dataset_info(os.path.join("ai_process",session_name, "dataset_info.json"))
    
    # Handle gaps in timestamps
    dataset_info = handle_gaps(dataset_info)
    
    # Generate sequences and update JSON metadata
    generate_sequences(dataset_info)
