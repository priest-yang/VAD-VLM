import os
import json
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize evaluation indicators for different scenarios')
    parser.add_argument('--path_to_test_pkl', help='directory_path_to_store_pickles')
    parser.add_argument('--save_image_path', help='path_to_save_the_plot_image')
    parser.add_argument('--eval_metrics', type=str, help='Evaluation metrics to be compared')
    args = parser.parse_args()
    return args

def plot_plan_L2_1s_bars(path_to_test_pkl, save_image_path, eval_metrics):
    # 存储数据的列表
    scenario_names = []
    metrics_values = []
    
    # 遍历所有子文件夹
    for folder_name in os.listdir(path_to_test_pkl):
        folder_path = os.path.join(path_to_test_pkl, folder_name)
        
        # 检查是否为文件夹
        if not os.path.isdir(folder_path):
            continue
            
        json_path = os.path.join(folder_path, 'evaluation_results.json')
        
        # 检查json文件是否存在
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                
                # 获取 plan_L2_1s 值
                metrics = data['metric_dict'][eval_metrics]
                
                # 保存数据
                scenario_names.append(folder_name)
                metrics_values.append(metrics)
    
    # 创建条形图
    plt.figure(figsize=(12, 6))
    plt.bar(scenario_names, metrics_values)
    
    # 设置图表属性
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('scenario')
    plt.ylabel(eval_metrics)
    plt.title(f'Compare {eval_metrics} in different scenarios')
    
    # 调整布局以防止标签被切掉
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_image_path, f'{eval_metrics}_comparison.png')
    plt.savefig(save_path)
    print(f'{eval_metrics} comparison image saved to {save_path}')

def main():
    args = parse_args()
    plot_plan_L2_1s_bars(args.path_to_test_pkl, args.save_image_path, args.eval_metrics)

if __name__ == '__main__':
    main()

