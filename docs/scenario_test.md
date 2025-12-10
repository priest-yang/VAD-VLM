# 分场景开环评测

## 1.自动划分场景和转换地图

为了了解不同场景下评测的结果，需要先提取出测试集中的不同场景，生成对应场景的pkl格式的文件，再提取出不同场景下的真值地图信息，即从pkl格式转为json格式的真值地图信息。   

自动划分场景和转换地图的工具的存储文件夹路径为 `tools/scenario_eval`，共包括两个脚本，包括`split_scene_and_merge.py`和`convert_map_json.py`。
   
在`/lwad/lwad`路径下，执行以下命令

```bash
python tools/scenario_eval/split_scene_and_merge.py \
    --data_root='/data/ceph/data/nuplan/dataset' \
    --pkl_path='/data/ceph/data/nuplan/ann_files/test/test_1010.pkl' \
    --pkl_re_path='directory_path_to_store_pickles' \
    --is_overwrite_map=False
```

参数说明如下:
- `--data_root`: nuplan数据集的路径
- `--pkl_path`: 要划分的pkl文件的路径
- `--pkl_re_path`: 划分完成的结果保存的路径
- `--is_overwrite_map`: 可选，是否覆写已经生成map (默认False, 因为生成map的时间比较长)   

## 2.分场景评测

命令行执行以下命令，自动完成所有场景的评测，并将评测结果保存在`Path/to/scenario_eval_pkl`路径之下的`metric_record.json`中。   

在`lwad/lwad`目录下，执行

```bash
python ./adbase/vad/test_scenario.py \
    --script_name=./adbase/vad/test.py \
    ./adbase/vad/configs/VAD/VAD_lightwheel_config.py \
    /data/ckpts/model.pth \
    --path_to_test_pkl='directory_path_to_store_pickles' \ --nproc_per_node=2 \
    --master_port=29303 \
    --launcher=pytorch \
    --eval="bbox"
```   

参数说明如下:   
- `--script_name`: `test.py`的路径, 单个pkl的开环测评脚本
- `--path_to_test_pkl`: 上一步划分完成的结果保存的路径

## 3.绘制不同场景下的指标对比图

在`/lwad/lwad`路径下，执行以下命令，绘制不同场景下的指标对比图

```bash
python ./tools/scenario_eval/scenarios_eval_res_vis.py \
    --path_to_test_pkl='directory_path_to_store_pickles' \
    --save_image_path='directory_path_to_save_the_plot_image' \
    --eval_metrics='Evaluation_metrics_to_be_compared'
```

参数说明如下:   
- `--path_to_test_pkl`: 第一步划分完成的结果保存的路径
- `--save_image_path`: 绘制完成的结果保存的路径
- `--eval_metrics`: 需要对比的指标

## 4.场景划分说明

### 4.1 场景划分思路

- 对于评测集`pkl文件`，统计infos中的tags类别，总共有66类，将其保存在一个集合中，记为all_tags
- 遍历all_tags，对于all_tags中的每一个值记为tag，遍历所有的infos，infos中关键帧的tags键对应的value是一个集合，在该集合中查找是否有tag，如果有tag，则将该关键帧及其前5帧和后5帧加入到`tag.pkl`
- 对于得到的66个`tag.pkl`，根据预设的scenes合并pkl得到对应场景划分

### 4.2 场景释义

| 场景划分                            | 对应tags                                                   | 释义               | 描述                                                                                                              |
| ------------------------------- | ------------------------------------------------------ | ---------------- | --------------------------------------------------------------------------------------------------------------- |
| accelerating                    | accelerating_at_crosswalk                              | 在人行横道上加速         | 车辆在接近或处于人行横道区域时进行加速的情况。可能是车辆在等待行人通过后开始加速前行，或者由于某种原因在人行横道附近开始加速。                                                 |
|                                 | accelerating_at_stop_sign_no_crosswalk                 | 在没有人行横道的停止标志处加速  | 当车辆到达一个没有人行横道的停止标志位置，停下后又开始加速的情况。例如在一些路口只有停止标志而没有人行横道，车辆观察交通状况后决定加速通过                                           |
|                                 | accelerating_at_stop_sign                              | 在停止标志处加速         | 广义上指车辆在任何有停止标志的地方停下后开始加速，可能有或没有人行横道                                                                             |
|                                 | accelerating_at_traffic_light_with_lead                | 在有前车的交通灯处加速      | 车辆在交通信号灯处等待，当前方有引导车辆（前车）并且信号灯变为允许通行状态时，车辆开始加速。例如在路口等待绿灯时，前方有一辆车先启动，本车随后加速跟随                                     |
|                                 | accelerating_at_traffic_light_without_lead             | 在没有前车的交通灯处加速     | 车辆在交通信号灯处等待，当信号灯变为允许通行状态且前方没有引导车辆时，车辆开始加速。比如在路口独自等待绿灯，绿灯亮后自行加速通过                                                |
|                                 | accelerating_at_traffic_light                          | 在交通灯处加速          | 较为宽泛地表示车辆在交通信号灯处由静止变为加速的各种情况，不具体区分前方是否有引导车辆等条件                                                                  |
| behind                          | behind_bike                                            | 在自行车后面           | 自车处于一辆自行车的后方位置。可能是在道路上行驶过程中，自车跟随着自行车行驶，保持一定的距离。例如在城市道路中，自车在自行车后面行驶，需要根据自行车的速度和行驶状态来调整自己的速度和行驶策略，同时要注意避免与自行车发生碰撞 |
|                                 | behind_long_vehicle                                    | 在长型车辆后面          | 当自车位于一辆长度较长的车辆（如卡车、大巴车等）的后方。在这种情况下，自车的视线可能会受到长型车辆的遮挡，影响对前方道路状况的观察                                               |
|                                 | behind_pedestrian_on_driveable                         | 在可行驶区域上的行人后面     | 自车在可行驶的区域上处于一个行人的后方。可能是在一些低速行驶区域，如小区道路、停车场等地方，自车发现前方有行人在行走，需要缓慢跟行或者等待合适的时机超越行人                                  |
| changing_lane                   | changing_lane_to_right                                 | 向右变道             | 车辆从当前所在车道向右侧相邻车道进行变道                                                                                            |
|                                 | changing_lane                                          | 变道               | 更宽泛的场景，涵盖了车辆向任何方向进行变道的情况，包括向左变道、向右变道以及在多车道道路上的连续变道等                                                             |
| following_lane                  | following_lane_with_lead                               | 跟随车道且有前车         | 在这种场景下，车辆正在沿着既定的车道行驶，并且前方有一辆引导车辆（前车）。车辆需要根据前车的速度、行驶状态和距离来调整自己的速度，以保持安全的跟车距离                                     |
|                                 | following_lane_with_slow_lead                          | 跟随车道且前车行驶缓慢      | 车辆在车道上行驶，且前方有一辆行驶速度较慢的引导车辆，这里特别强调了前车行驶缓慢的情况。车辆可能需要考虑是否超车，或者在不超车的情况下，如何保持安全距离并适应前车的低速行驶                          |
|                                 | following_lane_without_lead                            | 跟随车道且无前车         | 车辆在车道上行驶，但前方没有引导车辆                                                                                              |
| high                            | high_lateral_acceleration                              | 高横向加速度           | 可能的情况包括高速转弯、紧急避让操作或行驶在弯道半径较小的道路上                                                                                |
|                                 | high_magnitude_jerk                                    | 高加加速度            | 可能是由于车辆突然加速、急刹车、频繁换挡或遇到颠簸路面等情况引起的                                                                               |
|                                 | high_magnitude_speed                                   | 高速度量级            | 车辆以较高的速度行驶                                                                                                      |
| near                            | near_barrier_on_driveable                              | 在可行驶区域附近有障碍物     | 车辆在可行驶的道路区域附近存在某种障碍物的情况。障碍物可能是道路施工设置的临时屏障、防撞栏等                                                                  |
|                                 | near_construction_zone_sign                            | 在施工区域标志附近        | 表示车辆靠近施工区域的标志                                                                                                   |
|                                 | near_high_speed_vehicle                                | 靠近高速行驶的车辆        | 车辆处于与一辆高速行驶的车辆较近的位置                                                                                             |
|                                 | near_long_vehicle                                      | 靠近长型车辆           | 车辆靠近长度较长的车辆，如卡车、大巴车等                                                                                            |
|                                 | near_multiple_pedestrians                              | 靠近多个行人           | 车辆周围有多个行人的场景                                                                                                    |
|                                 | near_multiple_vehicles                                 | 靠近多辆车辆           | 车辆处于多辆其他车辆附近的情况                                                                                                 |
|                                 | near_pedestrian_at_pickup_dropoff                      | 在行人上下车地点附近       | 车辆靠近行人上下车的地点，如公交车站、出租车停靠点等                                                                                      |
|                                 | near_pedestrian_on_crosswalk_with_ego                  | 自车在人行横道上靠近行人     | 自车处于人行横道上并且靠近行人的场景                                                                                              |
|                                 | near_pedestrian_on_crosswalk                           | 靠近在人行横道上的行人      | 车辆靠近有人行横道且上面有行人的位置                                                                                              |
|                                 | near_trafficcone_on_driveable                          | 在可行驶区域附近有交通锥     | 车辆在可行驶的道路区域附近有交通锥的情况                                                                                            |
| on                              | on_all_way_stop_intersection                           | 在全向停车路口          | 这种场景表示车辆处于一个所有方向来车都必须先完全停车，然后观察交通状况，按照一定顺序依次通过的交叉路口。在这个路口，没有交通信号灯或其他优先控制设备，完全依靠驾驶员的判断和礼让来确保安全通行                 |
|                                 | on_carpark                                             | 在停车场             | 车辆位于停车场区域内。可能处于寻找停车位、缓慢行驶在停车通道中、倒车入库或者准备驶出停车场等状态                                                                |
|                                 | on_intersection                                        | 在交叉路口            | 广义地表示车辆处于任何类型的交叉路口处，包括有交通信号灯控制的路口、有停车标志控制的路口以及没有任何控制措施的路口                                                       |
|                                 | on_pickup_dropoff                                      | 在上下客区域           | 车辆处于专门用于乘客上下车的地点，比如出租车候车点、公交车站、机场或火车站的接送区域等                                                                     |
|                                 | on_stopline_crosswalk                                  | 在人行横道前的停止线处      | 车辆停在或靠近有人行横道的道路上的停止线位置                                                                                          |
|                                 | on_stopline_stop_sign                                  | 在有停止标志的停止线处      | 当车辆行驶到有停止标志的路口时，必须在相应的停止线前完全停下。车辆需要观察路口的交通状况，确认安全后才能继续行驶                                                        |
|                                 | on_stopline_traffic_light                              | 在有交通信号灯的停止线处     | 车辆位于有交通信号灯控制的路口的停止线处                                                                                            |
|                                 | on_traffic_light_intersection                          | 在有交通信号灯的交叉路口     | 车辆处于有交通信号灯控制的交叉路口                                                                                               |
| starting                        | starting_high_speed_turn                               | 开始高速转弯           | 车辆从相对较高的速度开始进行转弯操作。可能是在高速公路匝道或类似场景中，车辆以较快的速度进入转弯状态                                                              |
|                                 | starting_right_turn                                    | 开始右转             | 车辆开始执行向右侧转弯的动作。可以在路口、 driveway 等位置                                                                              |
|                                 | starting_left_turn                                     | 开始左转             | 车辆开始向左转弯。通常在路口处                                                                                                 |
|                                 | starting_straight_stop_sign_intersection_traversal     | 在有停止标志的路口开始直行通过  | 车辆在有停止标志的交叉路口处，准备直行通过。车辆首先要在停止线前完全停下，观察路口各个方向的交通情况，确认安全后开始直行                                                    |
|                                 | starting_low_speed_turn                                | 开始低速转弯           | 车辆以较低的速度开始进行转弯。可能在狭窄道路、居民区或需要谨慎驾驶的区域                                                                            |
|                                 | starting_straight_traffic_light_intersection_traversal | 在有交通信号灯的路口开始直行通过 | 车辆在有交通信号灯控制的交叉路口，当信号灯为绿灯时，车辆开始直行通过路口                                                                            |
|                                 | starting_unprotected_cross_turn                        | 开始无保护交叉路口转弯      | 在没有交通信号灯或交通标志明确保护的交叉路口开始转弯                                                                                      |
|                                 | starting_protected_cross_turn                          | 开始有保护交叉路口转弯      | 在有交通信号灯或交通标志明确给予保护的交叉路口开始转弯                                                                                     |
|                                 | starting_unprotected_noncross_turn                     | 开始无保护非交叉路口转弯     | 在不是传统交叉路口（如 driveway 接入主路等）且没有保护措施的地方开始转弯                                                                       |
|                                 | starting_protected_noncross_turn                       | 开始有保护非交叉路口转弯     | 在非交叉路口但有一定保护措施（如减速让行标志等）的地方开始转弯                                                                                 |
|                                 | starting_u_turn                                        | 开始掉头             | 车辆开始进行掉头操作                                                                                                      |
| stationary                      | stationary_at_traffic_light_with_lead                  | 在有前车的交通灯处静止      | 车辆在交通信号灯前处于静止状态，并且前方有一辆引导车辆（前车）                                                                                 |
|                                 | stationary_in_traffic                                  | 在交通流中静止          | 车辆在正常的交通流中处于静止状态，可能是由于交通拥堵、道路施工或其他原因导致车辆无法前进                                                                    |
|                                 | stationary                                             | 静止               | 较为宽泛的描述，表示车辆处于静止不动的状态，但没有具体说明静止的位置或原因                                                                           |
|                                 | stationary_at_traffic_light_without_lead               | 在没有前车的交通灯处静止     | 车辆在交通信号灯前静止，且前方没有引导车辆                                                                                           |
| stopping                        | stopping_at_crosswalk                                  | 在人行横道处停车         | 车辆行驶到人行横道前，为了让行人优先通过而停车                                                                                         |
|                                 | stopping_at_stop_sign_without_lead                     | 在没有前车的停止标志处停车    | 车辆到达有停止标志的位置，且前方没有引导车辆                                                                                          |
|                                 | stopping_merge                                         | 在汇入车道处停车         | 当车辆准备从匝道或其他道路汇入主路时，可能需要在汇入点停车等待合适的时机                                                                            |
|                                 | stopping_at_stop_sign_no_crosswalk                     | 在没有人行横道的停止标志处停车  | 车辆到达有停止标志的位置停车，强调了该停止标志处没有人行横道                                                                                  |
|                                 | stopping_at_traffic_light_with_lead                    | 在有前车的交通信号灯处停车    | 车辆在交通信号灯前停车，并且前方有一辆引导车辆（前车）                                                                                     |
|                                 | stopping_with_lead                                     | 在有前车的情况下停车       | 广义上表示车辆在有其他车辆作为引导的情况下停车                                                                                         |
|                                 | stopping_at_stop_sign_with_lead                        | 在有前车的停止标志处停车     | 车辆在有停止标志的位置停车，且前方有一辆引导车辆                                                                                        |
|                                 | stopping_at_traffic_light_without_lead                 | 在没有前车的交通信号灯处停车   | 车辆在交通信号灯前停车，且前方没有引导车辆                                                                                           |
| traversing                      | traversing_crosswalk                                   | 穿过人行横道           | 车辆正在通过人行横道                                                                                                      |
|                                 | traversing_merge                                       | 穿过汇入车道           | 当车辆从匝道或其他道路并入主路时，处于正在穿过汇入区域的状态                                                                                  |
|                                 | traversing_traffic_light_intersection                  | 穿过有交通信号灯的交叉路口    | 车辆在有交通信号灯控制的交叉路口行驶通过                                                                                            |
|                                 | traversing_intersection                                | 穿过交叉路口           | 广义上表示车辆正在通过任何类型的交叉路口，包括有交通信号灯控制、有停车标志控制或无控制的交叉路口                                                                |
|                                 | traversing_pickup_dropoff                              | 穿过上下客区域          | 车辆在乘客上下车的区域行驶通过                                                                                                 |
| low_magnitude_speed             | low_magnitude_speed                                    | 低速度量级            | 车辆以相对较低的速度行驶。车辆可能在特定区域行驶或交通状况导致低速行驶                                                                             |
| waiting_for_pedestrian_to_cross | waiting_for_pedestrian_to_cross                        | 等待行人通过           | 车辆在道路上行驶时遇到行人准备穿越马路的情况，可能包括人行横道处/无交通信号控制的路口或路段等                                                                 |
| medium_magnitude_speed          | medium_magnitude_speed                                 | 中等速度量级           | 车辆以既不是特别快也不是特别慢的速度行驶，车辆可能位于城区道路                                                                                 |


