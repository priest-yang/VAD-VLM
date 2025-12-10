import os
import pickle
import argparse
import convert_map_json as cmj

def parse_args():
    parser = argparse.ArgumentParser(description="Split a pickle file based on tags and convert map")
    parser.add_argument('--data_root', type=str, help='Data root path', default='/data/ceph/data/nuplan/dataset')
    parser.add_argument('--pkl_path', type=str, help='Path to the input pickle file', default='/data/ceph/data/nuplan/ann_files/test/test_1010.pkl')
    parser.add_argument('--pkl_re_path', type=str, help='Path to the splited result file', default='/data/ceph/data/nuplan/ann_files/test/')
    parser.add_argument('--is_overwrite_map', type=bool, help='Whether overwrite the previously generated files', default=False)
    args = parser.parse_args()
    return args

def split_pkl_based_tag(pkl_path, pkl_re_path):
    if not os.path.exists(pkl_re_path):
        os.mkdir(pkl_re_path)
        print(f"Folder {pkl_re_path} create success")
    else:
        print(f"Folder {pkl_re_path} already exists")

    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    infos = pkl_data["infos"]
    n = len(infos)

    # Staristical Different tags
    all_tags = []
    for item in infos:
        tags = item["tags"]
        for tag in tags:
            all_tags.append(tag)
    
    all_tags = set(all_tags)
    
    # Divide data besed on tag 
    for tag in all_tags:
        filtered_data = []
        for index,data in enumerate(infos):
            if len(data['tags']) == 0:
                continue
            if tag in data['tags']:
                target_index = index
                start_index = max(0, target_index - 5)
                end_index = min(n, target_index + 5)
                filtered_data.extend(infos[start_index:end_index])
                # filtered_data.append(data)

        mmcv_data = {
        "infos": filtered_data,
        'metadata': {'version': '1.0-trainval'},
        }
        # save the data corresponding to each tag to different pkl files       
        with open(pkl_re_path+f"{tag}.pkl", "wb") as f:
            pickle.dump(mmcv_data, f)
    print("complete split")

def merge_same_scene_pkl(pkl_re_path):
    root_path = pkl_re_path
    scenes = ["accelerating","behind","changing_lane", "following_lane", "high", "near", "on", "starting", "stationary", "stopping", "traversing", "low_magnitude_speed", "medium_magnitude_speed", "waiting_for_pedestrian_to_cross"]
    # scenes = ["accelerating","behind"]
    # Loop through the files in the directory
    for scene in scenes:
        merge_path = f"{root_path}{scene}/{scene}_merge.pkl"
        directory = os.path.dirname(merge_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Folder {scene} create success")
        else:
            print(f"Folder {scene} already exists")
        # Loop through the files in the directory
        all_data = []
        for file in os.listdir(root_path):
            if file.startswith(f'{scene}') and file.endswith('.pkl'):
                file_path = os.path.join(root_path, file)
                # Load data from the pickle file
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
                infos = data["infos"]
                all_data += infos
                # all_data.append(data)

        # Create the mmcv_data dictionary
        mmcv_data = {
            "infos": all_data,
            'metadata': {'version': '1.0-trainval'},
        }

        # Save the mmcv_data to the specified output file
        with open(merge_path, "wb") as f:
            pickle.dump(mmcv_data, f)
        print("complete merge!")    

def merge_convert_gt_map_json(data_root, pkl_re_path):
    for subdir_name in os.listdir(pkl_re_path):
        subdir_path = os.path.join(pkl_re_path, subdir_name)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith('.pkl'):
                    merge_pkl_path = os.path.join(subdir_path, filename)
                    merge_pkl_dir = os.path.dirname(merge_pkl_path)
                    file_name = os.path.basename(merge_pkl_path).split('.')[0]
                    save_path = os.path.join(merge_pkl_dir, f'{file_name}_map.json')
                    cmj.convert(data_root, merge_pkl_path, save_path)
                    print(f'complete convert {file_name}_map.json')

                
def main():
    args = parse_args()

    # Call the main function with the arguments
    
    split_pkl_based_tag(args.pkl_path, args.pkl_re_path)
    merge_same_scene_pkl(args.pkl_re_path)
    if args.is_overwrite_map:
        merge_convert_gt_map_json(args.data_root, args.pkl_re_path)


if __name__ == "__main__":
    main()


    