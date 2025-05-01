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
                # Add other necessary fields here
            })
    
    return dataset_info

# Function to generate sequences and update sequences JSON
def generate_sequences(dataset_info):
    if not os.path.exists(sequence_folder):
        os.makedirs(sequence_folder)
    
    sequences = []
    for i in range(0, len(dataset_info), sequence_length):
        sequence_items = dataset_info[i:i + sequence_length]
        
        # Gather information for the sequence
        sequence_start_time = sequence_items[0]['timestamp']
        sequence_end_time = sequence_items[-1]['timestamp']
        future_action = get_future_action(sequence_end_time)  # Function to get future action
        
        # Create sequence folder if not exists
        sequence_folder_path = os.path.join(sequence_folder, f"sequence_{i // sequence_length + 1}")
        if not os.path.exists(sequence_folder_path):
            os.makedirs(sequence_folder_path)
        
        # Copy images and update dataset_info with paths
        for item in sequence_items:
            original_filepath = item['original_filepath']
            filename = os.path.basename(original_filepath)
            target_filepath = os.path.join(sequence_folder_path, filename)
            
            # Copy image to sequence folder
            copyfile(original_filepath, target_filepath)
            
            # Update dataset_info with target filepath
            item['target_filepath'] = target_filepath
        
        # Add sequence information to sequences list
        sequences.append({
            'id': i // sequence_length + 1,
            'start_time': sequence_start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': sequence_end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'items': [item['filename'] for item in sequence_items],
            'prices': [item['price'] for item in sequence_items],
            'path_to_images': [item['target_filepath'] for item in sequence_items],
            'future_action': future_action,  # Add future action here
        })
    
    # Save sequences information to JSON
    with open(sequences_json_path, 'w') as json_file:
        json.dump(sequences, json_file, indent=4)

# Function to get future action based on the end time of the sequence
def get_future_action(end_time):
    # Example function to get future action, replace with your logic
    future_action = {
        'timestamp': end_time + timedelta(minutes=5),  # Example: Next timestamp after 5 minutes
        'percent_change': 0.5  # Example: Positive or negative percent change
    }
    return future_action

# Main script execution
if __name__ == "__main__":
    dataset_info = read_dataset_info(os.path.join("ai_process",session_name, "dataset_info.json"))
    dataset_info = handle_gaps(dataset_info)
    generate_sequences(dataset_info)
