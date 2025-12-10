import os
import base64
from datasets.arrow_dataset import Dataset
from vlannotator.api import *
import pickle
from functools import partial
import time
import random
import json

from vlannotator.prompt.HierarchicalPlanning import HierarchicalPlanning
from vlannotator.prompt.SceneAnalysis import SceneAnalysis
from vlannotator.prompt.SceneDescription import SceneDescription
from vlannotator.prompt.ScenePrompt import ScenePrompt
from vlannotator.utils import pose_v_describe, prepare_future_trajectories_with_x_forward
from vlannotator.utils import SCENARIO_FILTER_TYPES

def load_indexes(save_root):
    return Dataset.load_from_disk(save_root)

import numpy as np
import os.path as osp
import argparse
    

def data_annotation(meta_data_path, save_root, model="gpt-4o"):
    dataset = pickle.load(open(meta_data_path, "rb"))
    dataset = dataset["infos"] if "infos" in dataset else dataset
    base_name = osp.basename(meta_data_path).replace(".pkl", "")
    
    assert isinstance(dataset, list), "dataset must be a list"
    if len(dataset) == 0:
        print(f"no data in {meta_data_path}")
        return

    api_func = gpt if "gpt" in model else qwen_api if "qwen" in model else zhipuai_api
    assert model in ["gpt-4o", "gpt-4o-mini"] if "gpt" in model else True
    APIConfig.OPENAI_MODEL = model if "gpt" in model else APIConfig.OPENAI_MODEL

    try:
        indexes = load_indexes(os.path.join(save_root, f"{base_name}"))
        indexes = [index for index in indexes]
        print(f"load {len(indexes)} indexes from {os.path.join(save_root, f'{base_name}')}")
    except:
        indexes = []
        print(f"no indexes found in {os.path.join(save_root, f'{base_name}')}")
    
    print(f"start to annotate {len(dataset)} data")
    frame_idx = None if len(indexes) == 0 else indexes[-1]["frame_idx"]
    num_data = len(indexes)
    print("the length of dataset is ", len(dataset))


    for i, sample in enumerate(dataset):
        # skip the already annotated data

        # if sample["scenario_type"][0] not in SCENARIO_FILTER_TYPES:
        #     continue
        if frame_idx is not None and (sample["frame_idx"]-frame_idx) < args.downsample_rate:
            continue

        num_data += 1
        print(f"{sample['tags']} is annotated as the {num_data}th data")

        # scene description 
        frame_idx = sample["frame_idx"]
        img_path = os.path.join(NUPLAN_SENSOR_ROOT, sample["cams"]["CAM_F0"]['data_path'])
        scene_description = api_func(img_path, SceneDescription())
        
        description = ""
        if len(sample["gt_boxes"]) > 0:
            poses = sample["gt_boxes"][:, :3]
            velocities = sample["gt_velocity"]
            gt_names = sample["gt_names"]
            agents_description = []
            for name, pose, velocity in zip(gt_names, poses, velocities):
            # filter agent in +- 30m range
                if np.linalg.norm(pose[:2]) <= 30 and name in ["vehicle", "pedestrian"]:
                    agents_description.append((name, pose, velocity))
                else: continue

            # scene analysis
            if len(agents_description) > 0:
                description = pose_v_describe(agents_description)
            else:
                print("Warning: other_agent_position or other_agent_v is not in the sample")
            
        prompt = SceneAnalysis(description, scene_description)
        raster_path = os.path.join(args.raster_path, base_name, f"{sample['frame_idx']}.png")
        scene_analysis = api_func(raster_path, prompt)

        # hierarchical planning
        ego_fut_traj = prepare_future_trajectories_with_x_forward(dataset, i)
        ego_fut_traj = np.vstack((np.array([0,0]), ego_fut_traj))
        ego_fut_traj = ego_fut_traj[::2, :][1:9, :] # 1s, 2s, 3s, 4s, 5s, 6s, 7s, 8s
        ego_his_trajs = np.vstack([sample["gt_ego_his_trajs"], np.array([0,0])]) #-1s, -0.5s, 0s
        hierarchicalPrompt = HierarchicalPlanning(ego_his_trajs, ego_fut_traj)
        meta_actions = api_func(raster_path, hierarchicalPrompt)

        # add the sample to the indexes
        annotation = {}
        annotation["scene_analysis"] = scene_analysis
        annotation["scene_description"] = scene_description
        annotation["hierarchical_planning"] = meta_actions
        annotation["frame_idx"] = frame_idx
        annotation["tags"] = sample["tags"]
        annotation["timestamp"] = sample["timestamp"]
        print(meta_actions)
                
        indexes.append(annotation)
        if args.text_path is not None:
            os.makedirs(os.path.join(args.text_path, base_name), exist_ok=True)
            with open(os.path.join(args.text_path, base_name, f"cache_{i}.txt"), "w") as file:
                file.write(scene_analysis)
                file.write("\n")
                file.write(scene_description)
                file.write("\n")
                file.write(meta_actions)
                
        # if num_data % 2 == 0:
        new_dataset = Dataset.from_list(indexes)
        new_dataset.save_to_disk(os.path.join(save_root, f"{base_name}"))
    

parser = argparse.ArgumentParser(description="Data annotation script")
parser.add_argument('--metadata_root', type=str, default="data/nuplan/ann_files/splited_pickle/nuplan_trainval", help='Path to the index root')
parser.add_argument('--save_root', type=str, default="/data/ceph/data/nuplan/dataset/vlm_ann_data", help='Directory to save new data')
parser.add_argument('--model', type=str, default="gpt-4o", choices=['qwen', 'gpt-4o', 'gpt-4o-mini', 'glm'], 
                    help='Model to use (qwen or gpt-4o or glm)')
parser.add_argument('--nuplan_root', type=str, default="/data/ceph/data/nuplan/dataset", help='Path to the nuplan root')
parser.add_argument('--raster_path', type=str, default="/data/ceph/data/nuplan/dataset/raster/gt", help='Path to the raster root')
parser.add_argument('--text_path', type=str, default="/data/ceph/data/nuplan/cache/text", help='Path to the text root')
parser.add_argument('--nuplan_sensor_root', type=str, default="/data/ceph/", help='Path to the nuplan sensor root')
parser.add_argument("--use_multiprocessing", action="store_true", help="Use multiprocessing to process the annotation files.")
parser.add_argument("--downsample_rate", type=int, default=10, help="The gap between the frame indices of the annotated data") # default 2Hz to 0.2Hz

args = parser.parse_args()
random.seed(42)

if __name__ == "__main__":
    assert args.nuplan_root is not None, "Please specify the nuplan root"
    NUPLAN_DATA_ROOT = args.nuplan_root
    NUPLAN_MAPS_ROOT = osp.join(NUPLAN_DATA_ROOT, "maps")
    NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"
    NUPLAN_SENSOR_ROOT = args.nuplan_sensor_root #osp.join(NUPLAN_DATA_ROOT, f"nuplan-v1.1/sensor_blobs")

    # set the environment variable
    os.environ["NUPLAN_DATA_ROOT"] = NUPLAN_DATA_ROOT
    os.environ["NUPLAN_MAPS_ROOT"] = NUPLAN_MAPS_ROOT
    os.environ["NUPLAN_MAP_VERSION"] = NUPLAN_MAP_VERSION
    os.environ["NUPLAN_SENSOR_ROOT"] = NUPLAN_SENSOR_ROOT
    os.environ["SENSOR_BLOBS_ROOT"] = NUPLAN_SENSOR_ROOT # for load images
    
    files = os.listdir(args.metadata_root)

    if args.use_multiprocessing:
        from joblib import Parallel, delayed
        import multiprocessing
        from tqdm import tqdm
        print("Using multiprocessing to process the annotation files...")
        available_cpu = multiprocessing.cpu_count()
        print(f"Available CPU: {available_cpu}")
        jobs_needed = len(files)
        used_cpu = max(1, min(available_cpu - 1, jobs_needed))
        Parallel(n_jobs=used_cpu)(
            delayed(data_annotation)(meta_data_path=os.path.join(args.metadata_root, file),
                                    save_root=args.save_root, 
                                    model=args.model)
            for file in tqdm(files)
        )
    else:
        for file in files:
            data_annotation(meta_data_path=os.path.join(args.metadata_root, file),
                        save_root=args.save_root, 
                        model=args.model)
    

"""
python data_annotation.py \
    --metadata_root /data/ceph/data/nuplan/ann_files/splited_pickle/nuplan_test \
    --save_root /data/ceph/data/nuplan/dataset/vlm_ann_data/test/gpt4o \
    --nuplan_root /data/ceph/data/nuplan/dataset\
    --raster_path /data/ceph/data/nuplan/dataset/raster/gt \
    --text_path /data/ceph/data/nuplan/cache/text/test/gpt4o/ \
    --nuplan_sensor_root /data/ceph \
    --downsample_rate 10 \
    --model gpt-4o \
    --use_multiprocessing
"""