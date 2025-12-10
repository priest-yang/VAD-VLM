import subprocess
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Redirect ann_file_test(map)')
    parser.add_argument('--script_name',help='path to test.py')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--path_to_test_pkl', type=str)
    parser.add_argument('--nproc_per_node', type=int)
    parser.add_argument('--master_port', type=int)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    return args

def main():
    args = parse_args() 
    script_name = args.script_name

    for subdir_name in os.listdir(args.path_to_test_pkl):
        subdir_path = os.path.join(args.path_to_test_pkl, subdir_name)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith('.pkl'):
                    test_pkl_path = os.path.join(subdir_path, filename)
                    test_pkl_dir = os.path.dirname(test_pkl_path)
                    file_name = os.path.basename(test_pkl_path).split('.')[0]
                    test_map_path = os.path.join(test_pkl_dir, f'{file_name}_map.json')
                    result_json_path = os.path.join(test_pkl_dir)
                    print(f"test {file_name}")

                    process = subprocess.Popen(
                    ["torchrun", "--nproc_per_node", str(args.nproc_per_node), "--master_port", str(args.master_port), script_name, args.config, args.checkpoint,
                    "--modify_ann_file_test", test_pkl_path, "--modify_ann_file_map", test_map_path, "--json_dir", result_json_path, "--launcher", args.launcher,
                    "--eval", args.eval[0]],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )

                    # Real time reading and outputting standard output and standard errors
                    while True:
                        output = process.stdout.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output:
                            print(output.strip())

                    error_output = process.stderr.read()
                    if error_output:
                        print(f"Error: {error_output.strip()}")


    
if __name__ == '__main__':
    main()