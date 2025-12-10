# 可视化工具

可视化工具的存储文件夹路径为 `tools/visualizer` ，共包括三个脚本，包括`bbox_viz.py`，`map_viz.py`，和`visualizer.py`。

## 3D边界框可视化

`bbox_viz.py`脚本用于将3D边界框的点云标注投影到nuplan的八个摄像头图像上。
参数说明如下：

- pkl_path：要可视化的pkl路径
- output_dir：输出可视化结果的路径
- num_samples：300

终端使用方法如下:

```bash
cd tools/
python vis_tools/bbox_viz.py \
--pkl_path tools/vis_tools/demo/gt_sampled_300.pkl \
--output_dir vis_out \
--num_samples 300
```

bbox的颜色映射定义如下，可以在脚本里自行修改：

```python
self.color_map = {
    'vehicle': 'r',             # red
    'bicycle': 'g',             # green
    'traffic_cone': 'b',        # blue
    'barrier': 'm',             # magenta
    'czone_sign': 'y',          # yellow
    'generic_object': 'c',      # cyan
    'pedestrian': 'k'           # black
}
```

##  BEV+地图可视化
map_viz.py脚本用于可视化BEV视角下的地图和真值/预测情况，其中自车和自车真值轨迹为红色，其他真值元素（包括地图和其他bbox）的透明度为0.3，预测元素不透明，可以将可视化结果保存为gif格式的连续帧或者多张图像。

参数说明如下：

- data_root: 数据集的根目录路径
- ann_data_path: 标注数据的路径（pickle 文件）
- pred_pickle_path: 预测数据的路径（pickle 文件）
- output_type: 输出类型，可选 'image' 或 'gif'
- output_dir: 可视化结果的输出目录
- num_frames: 要可视化的帧数
- gif_fps: 如果输出为 GIF，设置每秒帧数

终端使用方法如下：

```bash
python vis_tools/map_viz.py \
--data_root /path/to/nuplan/data \
--ann_data_path tools/vis_tools/demo/gt_sampled_300.pkl \
--pred_pickle_path tools/vis_tools/demo/pred_sampled_300.pkl \
--output_type gif \
--output_dir tools/vis_tools/demo/vis_out \
--num_frames 300 \
--gif_fps 2
```

## 总体可视化

visualizer.py脚本用于将BEV+地图和3D边界框总体进行可视化，实现更加直观的单帧或连续帧的可视化，采用九图子宫格的方式，中间显示为BEV地图，周边为八个摄像头以及3d bbox的结果。
参数说明如下：

- data_root: 数据根目录路径
- ann_data_path: 标注数据文件路径（pickle 文件）
- pred_pickle_path: 预测数据文件路径（pickle 文件）
- output_dir: 输出目录
- start_frame: 开始处理的帧索引（默认为0）
- num_frames: 要处理的帧数量（如果不指定，将处理所有帧）
- duration: GIF 中每帧的持续时间（秒），仅在 GIF 模式下使用
- save_mode: 保存模式，可选 'gif' 或 'image'

终端使用方法如下：

```bash
python vis_tools/visualizer.py \
--data_root /path/to/nuplan/data \
--ann_data_path vis_tools/demo/gt_sampled_300.pkl \
--pred_pickle_path vis_tools/demo/pred_sampled_300.pkl \
--output_dir vis_tools/demo \
--start_frame 0 \
--num_frames 300 \
--duration 0.5 \
--save_mode gif
```