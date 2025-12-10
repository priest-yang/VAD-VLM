import os
import re
import pickle
import json
import numpy as np
from datasets.arrow_dataset import Dataset
from vlmdatagen.pipeline.utils import load_dataset
from collections import defaultdict
import multiprocessing as mp
import logging
import argparse
import torch

# Set up logging to log warnings into a file
logging.basicConfig(filename='warnings.log', level=logging.WARNING, format='%(message)s')

def extract_json(text):
    import json
    import re

    # Check if the text contains JSON markers
    if '```json' in text:
        # Find the positions of JSON markers
        json_start = text.find('```json') + len('```json')
        json_end = text.find('```', json_start)
        
        # Extract the JSON string
        json_str = text[json_start:json_end].strip()
    else:
        # Assume the entire text is JSON
        json_str = text.strip()

    # Remove any leading/trailing characters that may prevent parsing
    json_str = json_str.strip('`')

    # Parse the JSON content
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix common JSON issues
        json_str_fixed = re.sub(r'\\n', '', json_str)  # Remove escaped newlines
        json_str_fixed = re.sub(r'\\', '', json_str_fixed)  # Remove backslashes
        return json.loads(json_str_fixed)

def extract_duration(duration_str):
    """
    Extract duration in seconds from a duration string.
    The duration string can be in formats like:
    - '3s', '3 seconds', '3 sec'
    - '3m', '3 minutes', '3 min'
    - '0s-8s', '0 seconds - 8 seconds'
    - '1 min - 2 min'
    """
    # Remove any whitespace and lowercase the string
    duration_str = duration_str.strip().lower()
    
    try:
        # Units mapping to seconds
        time_units = {
            's': 1,
            'sec': 1,
            'secs': 1,
            'second': 1,
            'seconds': 1,
            'm': 60,
            'min': 60,
            'mins': 60,
            'minute': 60,
            'minutes': 60
        }
        
        # Function to extract number and unit
        def parse_time_part(time_part):
            match = re.search(r'(\d+(\.\d+)?)\s*([a-z]*)', time_part)
            if match:
                value = float(match.group(1))
                unit = match.group(3)
                if unit == '':
                    # Default to seconds if unit is missing
                    unit = 's'
                if unit in time_units:
                    multiplier = time_units[unit]
                    return value * multiplier
                else:
                    logging.warning(f"Unknown time unit '{unit}' in duration '{duration_str}'")
                    return None
            else:
                logging.warning(f"Could not parse time part '{time_part}' in duration '{duration_str}'")
                return None
        
        # Check if it's a range
        if '-' in duration_str:
            # Split by '-'
            start_str, end_str = duration_str.split('-')
            start_seconds = parse_time_part(start_str)
            end_seconds = parse_time_part(end_str)
            if start_seconds is not None and end_seconds is not None:
                duration_seconds = end_seconds - start_seconds
                if duration_seconds < 0:
                    logging.warning(f"Negative duration in range '{duration_str}'")
                    return None
            else:
                return None
        else:
            # Single duration
            duration_seconds = parse_time_part(duration_str)
            if duration_seconds is None:
                return None
        return duration_seconds
    except Exception as e:
        logging.warning(f"Error parsing duration '{duration_str}': {e}")
        return None


def process_file(args):
    file_name, entry_indices, pkl_dir, output_dir, overwrite, index_root_path, split = args

    # Load the dataset within the child process
    dataset = load_dataset(index_root_path, split, dataset_scale=1)

    # Retrieve the entries for this file_name
    index_entries = [dataset[idx] for idx in entry_indices]
    for entry in index_entries:
        if isinstance(entry['timestamp'], torch.Tensor):
            entry['timestamp'] = entry['timestamp'].item()

    pkl_path = os.path.join(pkl_dir, file_name + '.pkl')
    output_path = os.path.join(output_dir, file_name + '.pkl')

    if os.path.exists(output_path) and not overwrite:
        logging.info(f"File already exists and overwrite is False: {output_path}")
        return

    if not os.path.exists(pkl_path):
        logging.warning(f"Pickle file not found: {pkl_path}")
        return

    # Load the pickle file
    with open(pkl_path, 'rb') as f:
        frames = pickle.load(f)

    # Ensure frames is a list of dictionaries
    if not isinstance(frames, list):
        frames = list(frames)
    if len(frames) == 0:
        logging.warning(f"Empty frames for file {file_name}. Skipping.")
        return

    # Extract frame timestamps
    frame_timestamps = np.array([frame['timestamp'] for frame in frames])

    # Initialize arrays for annotations
    frame_annotated = np.zeros(len(frames), dtype=bool)
    frame_meta_actions = [None] * len(frames)

    # Process each index entry
    for _, entry in enumerate(index_entries):
        index_timestamp = entry['timestamp'] # .item()  # Assuming it's tensor([value])
        meta_actions_data = extract_json(entry['hierarchical_planning'])
        if meta_actions_data is None:
            continue  # Skip this entry
        meta_actions = meta_actions_data.get('Meta_Actions', [])

        # Find the closest frame timestamp
        time_diffs = np.abs(frame_timestamps - index_timestamp)
        min_diff = np.min(time_diffs)
        min_idx = np.argmin(time_diffs)

        if min_diff > 0.25e6:  # 0.25 seconds in microseconds
            logging.warning(f"Timestamp difference too large for file {file_name}: {min_diff/1e6}s")
            continue

        # Map Meta_Actions to frames based on duration
        current_frame_idx = min_idx

        # validate duration, total duration should be 8 seconds
        total_duration = 0
        for action in meta_actions:
            duration = action['Duration']
            duration_seconds = extract_duration(duration)
            if duration_seconds is None:
                logging.warning(f"Invalid duration in Meta_Actions for file {file_name}: {duration} \n Meta_Actions: {meta_actions}")
                continue
            total_duration += duration_seconds

        # duration_seconds = [extract_duration(action['Duration']) for action in meta_actions]
        if len(index_entries) - _ <= 1:
            if total_duration != 8:
                logging.warning(f"Total duration is not 8 seconds for file {file_name}: {total_duration}")
                logging.warning(f"Meta_Actions: {meta_actions}")
            continue

        for action in meta_actions:
            duration = action['Duration']
            # Extract duration in seconds
            duration_seconds = extract_duration(duration)
            num_frames = int(round(duration_seconds / 0.5))  # Since each frame is 0.5s

            for _ in range(num_frames):
                if current_frame_idx >= len(frames):
                    break
                frame_meta_actions[current_frame_idx] = action
                frame_annotated[current_frame_idx] = True
                current_frame_idx += 1

    # Collect annotated frames and their original indices
    annotated_frames = []
    annotated_indices = []
    for idx, (frame, annotated, meta_action) in enumerate(zip(frames, frame_annotated, frame_meta_actions)):
        if annotated:
            frame['meta_action'] = meta_action  # Assign the meta_action
            annotated_frames.append(frame)
            annotated_indices.append(idx)

    if not annotated_frames:
        logging.warning(f"No annotated frames in file {file_name}. Skipping.")
        return

    # Split annotated frames into continuous segments
    segments = []
    current_segment = []
    last_index = None
    for frame, idx in zip(annotated_frames, annotated_indices):
        if last_index is None or idx == last_index + 1:
            # Continuation of current segment
            current_segment.append(frame)
        else:
            # Start of new segment
            segments.append(current_segment)
            current_segment = [frame]
        last_index = idx

    # Add the last segment
    if current_segment:
        segments.append(current_segment)

    # Assign new scene_tokens to each segment
    import uuid
    log_file_name = file_name  # Assuming log_file_name is the same as file_name

    for segment_index, segment_frames in enumerate(segments, start=1):
        scene_token = uuid.uuid5(uuid.NAMESPACE_DNS, f"{log_file_name}_{segment_index}").hex
        for frame in segment_frames:
            frame['scene_token'] = scene_token

    # annotated_rate = np.mean(frame_annotated)
    # if annotated_rate < 0.1:  # Threshold for minimum annotated frames
    #     logging.warning(f"Too few annotated frames in file {file_name}: {annotated_rate*100:.2f}%")
    #     return

    # Save the new pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(annotated_frames, f)


def main():
    parser = argparse.ArgumentParser(description='Process the dataset and map Meta_Actions to frames.')

    parser.add_argument('--index_root_path', type=str, default='/data/ceph/data/DriveVLA/generated_data/', help='Path to the index dataset.')
    parser.add_argument('--split', type=str, default='vegas2_train_with_route_info', help='Dataset split to use.')
    parser.add_argument('--pkl_dir', type=str, default='/home/shaoze.yang/data/nuplan_to_vad/output/nuplan_trainval_1010', help='Directory containing the pickle files.')
    parser.add_argument('--output_dir', type=str, default='data/nuplan/vlm_mapped', help='Directory to save the output pickle files.')
    parser.add_argument('--use_multiprocessing', type=bool, default=True, help='Use multiprocessing to process files in parallel.')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of processes to use.')
    parser.add_argument('--overwrite', type=bool, default=False, help='Overwrite existing files in the output directory.')

    args = parser.parse_args()

    index_root_path = args.index_root_path
    split = args.split
    pkl_dir = args.pkl_dir
    output_dir = args.output_dir
    use_multiprocessing = args.use_multiprocessing
    overwrite = args.overwrite

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the index dataset
    dataset = load_dataset(index_root_path, split, dataset_scale=1)

    # Build a mapping from file_name to indices of entries
    file_name_to_indices = defaultdict(list)
    for idx, entry in enumerate(dataset):
        file_name = entry['file_name'][0]  # Assuming it's a list with one item
        file_name_to_indices[file_name].append(idx)

    # Prepare arguments for processing
    args_list = []
    for file_name, indices in file_name_to_indices.items():
        args_tuple = (file_name, indices, pkl_dir, output_dir, overwrite, index_root_path, split)
        args_list.append(args_tuple)

    if use_multiprocessing:
        # Use multiprocessing to process files in parallel
        if args.num_processes is None:
            num_processes = max(mp.cpu_count() - 10, 1)  # Leave one CPU free
        else:
            num_processes = args.num_processes
        num_processes = min(num_processes, len(args_list))
        with mp.Pool(num_processes) as pool:
            pool.map(process_file, args_list)
    else:
        # Process files sequentially
        for args_tuple in args_list:
            process_file(args_tuple)

if __name__ == '__main__':
    main()


"""
python tools/data_ann/merge_vlm.py \
  --index_root_path /data/ceph/data/DriveVLA/generated_data/ \
  --split pittburg_train_with_route_info \
  --pkl_dir /home/shaoze.yang/data/nuplan_to_vad/output/nuplan_trainval_1010 \
  --output_dir data/nuplan/vlm_mapped \
  --use_multiprocessing True \
  --num_processes 100 \
  --overwrite False
"""