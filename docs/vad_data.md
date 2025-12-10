Here is the content formatted in markdown:

# 光轮VAD训练数据详解

## pkl格式源数据

### 初始化

```python
info = {
    "lidar_path": lidar_path,
    "token": uuid.uuid5(uuid.NAMESPACE_DNS, lidar_pc_token).hex,
    "prev": uuid.uuid5(uuid.NAMESPACE_DNS, prev_token).hex if image_f0.prev_token else None,
    "next": uuid.uuid5(uuid.NAMESPACE_DNS, next_token).hex if image_f0.next_token else None,
    "can_bus": can_bus,
    "frame_idx": frame_idx,  # temporal related info
    "sweeps": [],
    "cams": dict(),
    "scene_token": scene_token, 
    "scene_token_lidarpc": image_f0.lidar_pc.scene_token, 
    "lidar2ego_translation": list(cs_record.translation_np), 
    "lidar2ego_rotation": list(cs_record.quaternion), 
    "ego2global_translation": list(pose_record.translation_np), 
    "ego2global_rotation": list(pose_record.quaternion), 
    "timestamp": image_f0.timestamp,
    "fut_valid_flag": fut_valid_flag,
    "map_location": map_location,
}
```

### 基本信息数据

1. **lidar_path**
   - **描述**: lidar pcd点云的原始数据路径
   - **来源**: `os.path.join(os.environ["NUPLAN_SENSOR_ROOT"], image_f0.lidar_pc.filename)`
   - **示例**: `./data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-16-03-27-0400__LIDAR_TOP__1533153857947444.pcd.bin`
   - **注意事项**: 在VAD实际的训练中并没有用到（但不能没有这一项）

2. **token, prev, next**
   - **描述**: 当前帧、上一帧、下一帧的专属token
   - **来源**: 通过`uuid5`生成，`uuid.uuid5(uuid.NAMESPACE_DNS, lidar_pc_token).hex`
   - **示例**: `1b9a789e08bb4b7b89eacb2176c70840`
   - **注意事项**: 第一帧的`prev`为`None`，最后一帧的`next`为`None`

3. **scene_token**
   - **描述**: 与`log_file`一一对应的场景token，用于标识整个场景
   - **来源**: 通过`uuid5`生成，`uuid.uuid5(uuid.NAMESPACE_DNS, log_file_name).hex`
   - **示例**: `bc6a757d637f4832be68986833ec17ac`

4. **frame_idx**
   - **描述**: 当前帧在整个场景中的索引
   - **来源**: 在处理每个场景时递增的计数器
   - **示例**: `0, 1, 2, ...`

5. **timestamp**
   - **描述**: 当前帧的时间戳
   - **来源**: 从NuPlan数据库中获取，`image_f0.timestamp`
   - **示例**: `1533153857947444`
   - **注意事项**: 微秒级时间戳

6. **map_location**
   - **描述**: 当前场景的地图位置
   - **来源**: 从NuPlan数据库中直接获取，`nuplan_db.log.location`
   - **示例**: NuPlan总共有四种地图，分别为：`sg-one-north`, `us-ma-boston`, `us-nv-las-vegas-strip`, `us-pa-pittsburgh-hazelwood`

7. **tag**
   - **描述**: 当前场景的标签，比如直行、静止、红绿灯等等
   - **来源**: 从NuPlan数据库中直接获取，`tags = set([tag.type for tag in lidar_pc.scenario_tags])`

### 坐标系转换数据

7. **lidar2ego_translation**
   - **描述**: LiDAR坐标系到车辆坐标系的平移向量
   - **来源**: 从NuPlan数据库中获取，`list(cs_record.translation_np)`
   - **示例**: `[0 0 0]`
   - **注意事项**: `[x, y, z]`，NuPlan默认lidar坐标系和自车坐标系是重合的。

8. **lidar2ego_rotation**
   - **描述**: LiDAR坐标系到车辆坐标系的旋转四元数
   - **来源**: 从NuPlan数据库中获取，`list(cs_record.quaternion)`
   - **示例**: `[1.0, 0.0, 0.0, 0.0]`
   - **注意事项**: 四元数格式`[w, x, y, z]`，NuPlan默认lidar坐标系和自车坐标系是重合的。

9. **ego2global_translation**
   - **描述**: 车辆坐标系到全局坐标系的平移向量
   - **来源**: 从NuPlan数据库中获取，`list(pose_record.translation_np)`
   - **示例**: `[664458.7055365173, 3998147.2296558036, 614.9258117050194]`
   - **注意事项**: NuPlan数据集使用的是UTM全球坐标系，所以这个值很大。

10. **ego2global_rotation**
    - **描述**: 车辆坐标系到全局坐标系的旋转四元数
    - **来源**: 从NuPlan数据库中获取，`list(pose_record.quaternion)`
    - **示例**: `[0.7052555420356911, 0.0145322582806032, 0.004299058719078649, 0.7087911906840684]`

### 传感器信息

11. **cams**
    - **描述**: 包含所有相机的信息
    - **来源**: 从NuPlan数据库中获取并处理
    - **示例**: `{'CAM_F0': {...}, 'CAM_B0': {...}, ...}`
      - **处理结果**:
      ```python
      cam = {
        'data_path': f"./data/nuplan/dataset/nuplan-v1.1/sensor_blobs/{image.filename}", 
        'type': camera.channel,
        'sample_data_token':  image.token, 
        'sensor2ego_translation': np.array(sensor2ego_translation), 
        'sensor2ego_rotation': sensor2ego_rotation.elements, 
        'ego2global_translation': [ego.x, ego.y, ego.z],
        'ego2global_rotation': [ego.qw, ego.qx, ego.qy, ego.qz], 
        'timestamp': image.timestamp, 
        'sensor2lidar_rotation': sensor2lidar_rotation.rotation_matrix, 
        'sensor2lidar_translation': np.array(sensor2lidar_translation.tolist()),
        'cam_intrinsic': np.array(camera.intrinsic) # 相机内参
      }
      ```
    - **注意事项**: 每个相机包含数据路径、类型、内参、外参等信息

12. **sweeps**
    - **描述**: 历史帧的LiDAR扫描信息
    - **来源**: 直接置空了，VAD训练不需要
    - **示例**: `[]`
    - **注意事项**: 由于VAD不需要用到LiDAR信息，所以直接置空

### CAN总线数据

13. **can_bus**
    - **描述**: 车辆CAN总线数据，包含位置、姿态、加速度等信息
    - **来源**: 从NuPlan数据库中获取并计算得到
    - **示例**: `[412.0, 1179.0, 0, 0.9998, 0, 0, 0.0175, 0, 0, 9.81, 0, 0, 0, 5.0, 0, 0, 0.0175, 1]`
    - **注意事项**: 包含19个元素，依次为`[x, y, z, qx, qy, qz, qw, ax, ay, az, ang_rate_x, ang_rate_y, ang_rate_z, vx, vy, vz, yaw_rad, yaw_deg]`

### 自车数据

14. **gt_ego_his_trajs**
    - **描述**: 自车的历史轨迹
    - **来源**: 通过处理NuPlan数据库中的历史帧计算得到
    - **示例**: `array([[ 8.2715256e-03, -3.9512994e-05], [ 2.7000766e-02, -2.5819591e-04]], dtype=float32)`
    - **注意事项**: 包含`his_ts`个时间步的轨迹。需要进行差分操作，将绝对坐标转换为相对于前一时刻的偏移量

15. **gt_ego_fut_trajs**
    - **描述**: 自车的未来轨迹
    - **来源**: 通过处理NuPlan数据库中的未来帧计算得到
    - **示例**: `array([[ 1.4763416e-01,  4.2693401e-03], [ 2.6370573e-01,  2.0201351e-03], [ 3.5134003e-01,  7.8743597e-04], [ 4.6239963e-01,  5.9338484e-04], [ 6.2846434e-01,  1.4426851e-03], [ 7.4899316e-01, -4.1283248e-03]], dtype=float32)`

16. **gt_ego_lcf_feat**
    - **描述**: 自车的局部坐标系特征
    - **来源**: 从NuPlan数据库中获取并处理
    - **示例**: `array([1.21327981e-01, -7.07906438e-03, 2.26566270e-01, -2.76316460e-02, -3.49283160e-04, 4.08400011e+00, 1.85000002e+00, 1.21534325e-01, -2.87394668e-03], dtype=float32)`
    - **注意事项**: `[vx, vy, ax, ay, w, length, width, vel, steer]`

17. **gt_ego_fut_cmd**
    - **描述**: 自车的未来行驶指令的one-hot编码
    - **来源**: 根据未来轨迹计算得到
    - **示例**: `[0, 1, 0]`
    - **注意事项**: 分为右转、左转、直行三种command。如果未来最后一帧的y轴偏移量小于-2，则为右转；如果y轴偏移量大于2，则为左转；否则为直行。

18. **gt_agent_fut_yaw**
    - **描述**: 场景中所有目标的未来朝向角
    - **来源**: 通过处理NuPlan数据库中的未来帧计算得到
    - **示例**: `[[yaw1, yaw2, ...], ...]`
    - **注意事项**: 用下一帧的yaw减当前yaw

19. **gt_agent_fut_goal**
    - **描述**: 场景中所有目标的未来目标方向类别
    - **来源**: 通过处理未来轨迹计算得到
    - **示例**: `[0, 1, ..., 9]`
    - **注意事项**: `0-8`表示8个方向，`9`表示静止

---

### Agent数据

1. **gt_boxes**
    - **描述**: 场景中所有目标的3D边界框
    - **来源**: 从NuPlan数据库中获取并处理
    - **示例**: `array([[15.7386601, -0.270137411, 0.698836231, 2.11421824, 5.08443737, 2.11082649, -0.0375901999], ...])`
    - **注意事项**: 每个边界框包含7个元素`[[x, y, z, l, w, h, yaw], ...]`，为**自车系**下的坐标

2. **gt_names**
    - **描述**: 场景中所有目标的类别名称
    - **来源**: 从NuPlan数据库中获取
    - **示例**: `['vehicle', 'pedestrian', ...]`
    - **注意事项**: 与`gt_boxes`一一对应

3. **gt_velocity**
    - **描述**: 场景中所有目标的速度
    - **来源**: 从NuPlan数据库中获取并处理
    - **示例**: `[[vx, vy], ...]`
    - **注意事项**: 与`gt_boxes`一一对应

4. **gt_agent_fut_trajs**
    - **描述**: 场景中所有目标的未来轨迹
    - **来源**: 通过处理NuPlan数据库中的未来帧计算得到
    - **示例**: `[[x1, y1, x2, y2, ...], ...]`
    - **注意事项**: 每个目标包含`fut_ts`个时间步的轨迹。需要进行差分操作，将绝对坐标转换为相对于前一时刻的偏移量

5. **gt_agent_fut_masks**
    - **描述**: 标识未来轨迹中每个时间步是否有效
    - **来源**: 全部置1
    - **示例**: `[[1, 1, 1, ...], ...]`
    - **注意事项**: 全部置1

6. **gt_agent_fut_yaw**
    - **描述**: 场景中所有目标的未来朝向角
    - **来源**: 通过处理NuPlan数据库中的未来帧计算得到
    - **示例**: `[[yaw1, yaw2, ...], ...]`
    - **注意事项**: 与`gt_agent_fut_trajs`对应

7. **num_lidar_pts**
    - **描述**: 每个目标的LiDAR点数
    - **来源**: 全部置1
    - **示例**: `[1, 1, 1, ...]`
    - **注意事项**: 与`gt_bboxes`一一对应。目前全部置1了，否则为0的话会在训练过程中`bbox`被过滤掉

8. **num_radar_pts**
    - **描述**: 每个目标的雷达点数
    - **来源**: 全部置1
    - **示例**: `[1, 1, 1, ...]`
    - **注意事项**: 与`gt_bboxes`一一对应

9. **valid_flag**
    - **标识**: 每个目标是否有效
    - **来源**: 全部置True
    - **示例**: `[True, True, ...]`
    - **注意事项**: 与`gt_boxes`一一对应。在当前实现中，我们默认所有目标都有效

---

### json格式地图数据

#### 初始化

```python
out_json_data = {
    "version": "1.3",
    "polygon": [],
    "line": [],
    "node": [],
    "drivable_area": [],           # Polygons
    "ped_crossing": [],            # Polygon
    "road_divider": [],            # LineString
    "lane_divider": [],            # LineString
    "road_segment": [],            # Polygon
    "lane": [],                    # Polygon
    "traffic_light": [],
    "road_block": [],
    "walkway": [],
    "stop_line": [],
    "connectivity": {},
    "arcline_path_3": {},
    "lane_connector": [],
    "carpark_area": [],
    "canvas_edge": []
}
```

目标是输出一个与Nuscenes对齐的NuPlan源地图，用到的有效数据为`ped_crossing`, `road_divider`, `lane_divider`, `road_segment`, `lane`，其余元素可以置为空。

---

### 地图元素映射

将NuPlan数据集中的地图数据从`GPKG`（GeoPackage）格式转换为`JSON`格式，并对类型名进行映射。具体的类型映射定义如下：

| 类型名         | 源类型             | 描述         | 数据类型 |
| -------------- | ------------------ | ------------ | -------- |
| **ped_crossing** | crosswalks          | 人行横道      | Polygon  |
| **road_segment** | road segments       | 道路段       | Polygon  |
| **lane**        | generic_drivable_areas | 车道         | Polygon  |
| **lane_divider** | boundaries[0]       | 车道分割线    | LineString |
| **road_divider** | boundaries[2]       | 道路分割线    | LineString |

---

### 数据格式示例

每一个类别会有一个独立的`token`，以及一个`polygon_token`或`line_token`用于索引一个polygon或者line，一个polygon或者line又会索引若干个node，一个node是一个`[x, y]`坐标，具体的类别定义示例如下：

- **ped_crossing**
    ```json
    "ped_crossing": [
        {
          "token": "36d4809d-9b21-4843-8fd1-709c4e4f2d75",
          "polygon_token": "92aad748-b432-4e20-9e09-94b82befbadd",
          "road_segment_token": null
        },
        ...
    ]
    ```

- **road_segment**
    ```json
    "road_segment": [
        {
          "token": "c4572886-9dc2-4456-a48e-ba84a7d58a3a",
          "polygon_token": "8a45b36b-7f4d-4c99-9102-cf272604f337",
          "is_intersection": false
        },
        ...
    ]
    ```

- **lane**
    ```json
    "lane": [
        {
          "token": "6f8e6dd0-a277-4377-b5fa-2a223a486c42",
          "polygon_token": "0a45f8f8-21f2-438c-9e20-6619c5ce7ec2",
          "lane_type": "CAR",
          "left_lane_divider_segments": [],
          "right_lane_divider_segments": []
        },
        ...
    ]
    ```
