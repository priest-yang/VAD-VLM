import pickle
import argparse
import numpy as np

def main(input_pkl_path, output_pkl_path):
    # Load data from the input pickle file
    data = pickle.load(open(input_pkl_path, "rb"))["infos"]
    
    selected_data = []

    # Process each frame in the data
    for frame in data:
        if 'on_pickup_dropoff' in frame['tags'] or frame['map_location'] == "sg-one-north":
            continue
        if np.max(frame['gt_ego_fut_trajs']) > 20:
            continue
        selected_data.append(frame)

    # Create the mmcv_data dictionary
    mmcv_data = {
        "infos": selected_data,
        'metadata': {'version': '1.0-trainval'},
    }

    # Save the selected data to the output pickle file
    with open(output_pkl_path, "wb") as f:
        pickle.dump(mmcv_data, f)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Filter and save selected data from a pickle file")
    parser.add_argument("input_pkl_path", help="Path to the input pickle file")
    parser.add_argument("output_pkl_path", help="Path to the output pickle file")

    args = parser.parse_args()

    # Call the main function with the arguments
    main(args.input_pkl_path, args.output_pkl_path)
