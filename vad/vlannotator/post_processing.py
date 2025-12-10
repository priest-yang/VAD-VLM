# #### This notebook is to combined the annotated pickles together. Will do the following:
# 
# - Exception handling
# - Mapping GT actions to meta actions by design
# - Mapping further to defined categories (if necessary)
# - One hot encoder the meta action (for VAD)
# 
# The meta actions by design:
# 
# | **Category**          | **Meta-actions**                                                                                     |
# |-----------------------|-----------------------------------------------------------------------------------------------------|
# | **Speed-control actions** | Speed up, Slow down, Slow down rapidly, Go straight slowly, Go straight at a constant speed, Stop, Wait, Reverse |
# | **Turning actions**    | Turn left, Turn right, Turn around                                                                  |
# | **Lane-control actions** | Change lane to the left, Change lane to the right, Shift slightly to the left, Shift slightly to the right      |
# 
# notice: `numpy` should be the same version with VAD, i.e. `1.21.6`

import pandas as pd
import os
import pickle
import sys
import numpy as np
import argparse
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process and combine annotated pickles with meta-action mapping')
    parser.add_argument('--input_dir', type=str, default="/data/ceph/data/nuplan/ann_files/splited_pickle/vlm_mapped_1119", 
                       help='Root path to mapped pickle files')
    parser.add_argument('--output_path', type=str, default="/data/ceph/data/nuplan/ann_files/trainval_meta_action_mapped_1119.pkl",
                       help='Path to save the combined pickle file')
    return parser.parse_args()

# Meta actions defined in VLM
META_ACTIONS = {
    # speed control
    "speed up", 
    "slow down", 
    "slow down rapidly", 
    "go straight slowly", 
    "go straight at a constant speed", 
    "stop", 
    "wait", 
    "reverse", 
    # turn action
    "turn left", 
    "turn right", 
    "turn around", 
    # lane change
    "change lane to the left", 
    "change lane to the right", 
    "shift slightly to the left", 
    "shift slightly to the right", 
}


# Handle exceptions in LLM annotation
EXCEPTION_MAP = {
    # Speed Control
    "accelerate": "speed up",
    "accelerate and continue left turn": "speed up",
    "accelerate slightly": "speed up",
    "adjust speed": "speed up",  # General speed control
    "continue at constant speed": "go straight at a constant speed",
    "continue at current speed": "go straight at a constant speed",
    "maintain constant speed": "go straight at a constant speed",
    "maintain speed": "go straight at a constant speed",
    "maintain speed and lane": "go straight at a constant speed",
    "maintain speed and shift slightly to the left": "shift slightly to the left",
    "maintain speed and shift slightly to the right": "shift slightly to the right",
    "slightly slow down": "slow down",
    "speed up and shift slightly to the left": "shift slightly to the left",
    "speed up and shift slightly to the right": "shift slightly to the right",
    "speed up slightly": "speed up", 
    "maintain current speed": "go straight at a constant speed",

    # Turn Action
    "continue turning left": "turn left",
    "slight turn to the right": "turn right",
    "slightly turn left": "turn left",
    "slightly turn right": "turn right",
    "turn more left": "turn left",
    "turn more sharply right": "turn right",
    "turn right slightly": "turn right",
    "turn sharp right": "turn right",
    "turn sharply left": "turn left",
    "turn sharply right": "turn right",
    "turn sharply to the right": "turn right",
    "turn slight left": "turn left",
    "turn slight right": "turn right",
    "turn slightly left": "turn left",
    "turn slightly right": "turn right",
    "turn slightly to the left": "turn left",
    "turn slightly to the right": "turn right",
    "turn to the right": "turn right",

    # Lane Change
    "adjust to the center of the lane": "go straight at a constant speed",
    # "change lane": "change lane to the right",  # need extra handling
    "change lane slightly to the right": "change lane to the right",
    "maintain lane": "go straight at a constant speed",
    "maintain lane and speed": "go straight at a constant speed",
    "maintain lane position": "go straight at a constant speed",
    "maintain lane with slight adjustments": "go straight at a constant speed",
    "shift more to the left": "shift slightly to the left",
    "shift significantly to the left": "shift slightly to the left",
    "shift significantly to the right": "shift slightly to the right",
    "shift slightly left": "shift slightly to the left",
    "shift slightly right": "shift slightly to the right",
    "shift to the left": "shift slightly to the left",
    "shift to the right": "shift slightly to the right",
    "shift to the right lane": "change lane to the right",
    "slight left shift": "shift slightly to the left",
    "slight right shift": "shift slightly to the right",
    "slight shift right": "shift slightly to the right",
    "slight shift to the right": "shift slightly to the right",
    "slightly adjust to the left": "shift slightly to the left",
    "slightly shift left": "shift slightly to the left",
    "slightly shift right": "shift slightly to the right",
    "slightly shift to the left": "shift slightly to the left",
    "slightly shift to the right": "shift slightly to the right",

    # Continue/Position (Closest Meta Actions)
    "adjust course": "go straight at a constant speed",
    "continue forward": "go straight at a constant speed",
    "continue straight": "go straight at a constant speed",
    "go straight": "go straight at a constant speed",
    "maintain current lane": "go straight at a constant speed",
    "maintain current position": "wait",
    "maintain position": "wait",
    "maintain straight": "go straight at a constant speed",
    "move forward": "go straight at a constant speed",
    "move forward slightly to the right": "shift slightly to the right",
    "move straight": "go straight at a constant speed",
    "stabilize": "go straight at a constant speed",
    "stay in lane": "go straight at a constant speed"
}


# ## Mapping to target categories

# mapping to target categories
TARGET_ACTION_MAP = {
    "speed up": "FORWARD",
    "slow down": "FORWARD",
    "slow down rapidly": "FORWARD",
    "go straight slowly": "FORWARD",
    "go straight at a constant speed": "FORWARD",
    "stop": "FORWARD",
    "wait": "FORWARD",
    "reverse": "FORWARD", 
    "turn left": "LEFT",
    "turn right": "RIGHT",
    "turn around": "LEFT",
    "change lane to the left": "CHANGE_LANE_LEFT",
    "change lane to the right": "CHANGE_LANE_RIGHT",
    "shift slightly to the left": "CHANGE_LANE_LEFT",
    "shift slightly to the right": "CHANGE_LANE_RIGHT",
}

TARGET_ACTIONS = [
"FORWARD", #: np.array([1, 0, 0, 0, 0]),
"LEFT", #: np.array([0, 1, 0, 0, 0]),
"RIGHT", #: np.array([0, 0, 1, 0, 0]),
"CHANGE_LANE_LEFT", #: np.array([0, 0, 0, 1, 0]),
"CHANGE_LANE_RIGHT", #: np.array([0, 0, 0, 0, 1]),
]


if __name__ == "__main__":
    args = parse_args()

    root_path = args.input_dir
    output_path = args.output_path

    all_data = []
    for file in tqdm(os.listdir(root_path), total=len(os.listdir(root_path)), desc="Loading data"):
        if file.endswith(".pkl"):
            file_path = os.path.join(root_path, file)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                assert type(data) == list
                all_data += data


    # ## Handle Exception Values in annotation
    for data in tqdm(all_data, total=len(all_data), desc="Processing data"):
        data['meta_action']['Action'] = data['meta_action']['Action'].lower()
        if data['meta_action']['Action'] not in META_ACTIONS:
            if data['meta_action']['Action'] in EXCEPTION_MAP.keys():
                data['meta_action']['Action'] = EXCEPTION_MAP[data['meta_action']['Action']]
            elif data['meta_action']['Action'] == "change lane":
                subject = data['meta_action']['Subject'].lower()
                if "left" in subject:
                    data['meta_action']['Action'] = "change lane to the left"
                elif "right" in subject:
                    data['meta_action']['Action'] = "change lane to the right"
                else:
                    print("Failed to map action: ", data['meta_action'], " Use default action: go straight at a constant speed")
                    data['meta_action']['Action'] = "go straight at a constant speed"
            else:
                print(data['meta_action'])
                print(f"Action {data['meta_action']['Action']} not in META_ACTIONS or EXCEPTION_MAP")
                print("Use default action: go straight at a constant speed")
                data['meta_action']['Action'] = "go straight at a constant speed"
        

    num_actions = len(TARGET_ACTIONS)
    actions2ohe = {action: np.zeros(num_actions, dtype=np.float32) for action in TARGET_ACTIONS}

    for mapping in actions2ohe.keys():
        actions2ohe[mapping][TARGET_ACTIONS.index(mapping)] = 1


    print(f"Mapped {len(actions2ohe.keys())} actions to one-hot encoded vectors")

    for data in all_data:
        if "gt_ego_fut_cmd_old" not in data.keys():
            data["gt_ego_fut_cmd_old"] = data['gt_ego_fut_cmd']
        data['gt_ego_fut_cmd'] = actions2ohe[TARGET_ACTION_MAP[data['meta_action']['Action']]]

    # ## Saving to Modified Pickle
    # Create the mmcv_data dictionary
    print(f"Saving to {output_path}...")
    mmcv_data = {
        "infos": all_data,
        'metadata': {'version': '1.0-trainval'},
    }

    # Save the mmcv_data to the specified output file
    with open(output_path, "wb") as f:
        pickle.dump(mmcv_data, f)



