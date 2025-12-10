from tqdm import tqdm
import numpy as np
from datasets.arrow_dataset import _concatenate_map_style_datasets
from pyquaternion import Quaternion
import numpy.typing as npt


SCENARIO_FILTER_TYPES = set([
    "waiting_for_pedestrian_to_cross",
    "near_multiple_vehicles",
    "changing_lane",
    "starting_right_turn",
    "behind_long_vehicle",
    "high_magnitude_speed",
    "stationary_in_traffic",
    "following_lane_with_lead",
    "traversing_pickup_dropoff",
    "stopping_with_lead",
    "starting_straight_intersection_traversal",
    "high_lateral_acceleration",
    "starting_left_turn",
    "near_pedestrian_on_crosswalk",
    "on_stopline_traffic_light",
    "on_intersection",
    "on_traffic_light_intersection",
    "traversing_intersection",
    "traversing_traffic_light_intersection",
    "near_construction_zone_sign",
    "on_pickup_dropoff",
    "near_pedestrian_at_pickup_dropoff"
])
        
        
        
def pose_v_describe(agents_description):
    description=""
    for name, pose, velocity in agents_description:
        description += f"the agent {name} is at the relative place of {pose}(x, y, z, yaw) with a velocity of {velocity}. \n"
    return description


from PIL import Image
def concatenate_images(img_path1, img_path2, save_path):
    """
    拼接两张图片，img_path1 在左，img_path2 在右，先调整两张图片的高度使其相同，然后保存到 save_path。
    """
    # 打开两张图片
    image1 = Image.open(img_path1)
    image2 = Image.open(img_path2)

    # 获取两张图片的尺寸
    width1, height1 = image1.size
    width2, height2 = image2.size

    # 确定新的高度为两者之中的最大值
    new_height = max(height1, height2)

    # 如果两张图片的高度不同，调整高度较小的图片
    if height1 != new_height:
        # 计算新的宽度，保持原始宽高比
        new_width1 = int(new_height * width1 / height1)
        image1 = image1.resize((new_width1, new_height), Image.Resampling.LANCZOS)
    if height2 != new_height:
        # 计算新的宽度，保持原始宽高比
        new_width2 = int(new_height * width2 / height2)
        image2 = image2.resize((new_width2, new_height), Image.Resampling.LANCZOS)

    # 获取调整大小后的宽度
    width1, height1 = image1.size
    width2, height2 = image2.size

    # 创建一个新的画布，宽度为两张图片宽度之和，高度为调整后的统一高度
    new_width = width1 + width2
    new_image = Image.new('RGB', (new_width, new_height))

    # 将两张图片粘贴到新图片上
    new_image.paste(image1, (0, 0))  # 第一张图片粘贴在左边
    new_image.paste(image2, (width1, 0))  # 第二张图片粘贴在右边

    # 保存到指定路径
    new_image.save(save_path)

def _restore_trajectory(trajectory_deltas: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Restore the trajectory deltas to get the future trajectory."""
    initial_position = trajectory_deltas[0]
    recovered_ego_fut_trajs = np.vstack(
        [initial_position, np.cumsum(trajectory_deltas, axis=0)]
    )
    return recovered_ego_fut_trajs

def _transform_to_global(
        points: npt.NDArray[np.float32],
        x: float,
        y: float,
        z: float,
        q: Quaternion,
        inv: bool = False,
        ) -> npt.NDArray[np.float32]:
        
    """Transform points to global coordinate frame."""
    if inv:
        return (points - np.array([x, y, z])) @ q.rotation_matrix
    else:
        return points @ q.rotation_matrix.T + np.array([x, y, z])

def prepare_future_trajectories(key_frame_data, frame_idx: int) -> npt.NDArray[np.float32]:
    """Prepare future trajectories for ego vehicle."""
    ego_fut_trajs_restored = []
    for future_offset in [0, 6, 12]:
        future_idx = frame_idx + future_offset
        scene_token = key_frame_data[frame_idx]["scene_token"] if frame_idx < len(key_frame_data) else None
        if future_idx < len(key_frame_data):
            future_data = key_frame_data[future_idx]
            if future_data["scene_token"] != scene_token:
                continue
            trajectory_deltas = future_data["gt_ego_fut_trajs"]
            future_positions = _restore_trajectory(trajectory_deltas)
            x, y, z = future_data["can_bus"][0:3]
            qx, qy, qz, qw = future_data["can_bus"][3:7]
            q = Quaternion([qw, qx, qy, qz])
            future_positions = np.c_[
                future_positions, np.zeros(future_positions.shape[0])
            ]
            future_positions_global = _transform_to_global(
                future_positions, x, y, z, q
            )
            ego_fut_trajs_restored.append(future_positions_global[1:, :])
    # Transform to ego coordinate frame
    x, y, z = key_frame_data[frame_idx]["can_bus"][0:3]
    qx, qy, qz, qw = key_frame_data[frame_idx]["can_bus"][3:7]
    q = Quaternion([qw, qx, qy, qz])
    ego_future_positions = []
    for traj in ego_fut_trajs_restored:
        traj_in_ego = _transform_to_global(traj, x, y, z, q, inv=True)
        traj_in_ego = traj_in_ego @ np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]) # rotate 90 degrees, make y-axis the forward direction

        ego_future_positions.append(traj_in_ego[:, :2])
    if ego_future_positions:
        return np.vstack(ego_future_positions)
    else:
        return np.zeros((2, 0))
    
def prepare_future_trajectories_with_x_forward(key_frame_data, frame_idx: int) -> npt.NDArray[np.float32]:
    """Prepare future trajectories for ego vehicle."""
    ego_fut_trajs_restored = []
    for future_offset in [0, 6, 12]:
        future_idx = frame_idx + future_offset
        scene_token = key_frame_data[frame_idx]["scene_token"]
        if future_idx < len(key_frame_data):
            future_data = key_frame_data[future_idx]
            if future_data["scene_token"] != scene_token:
                continue
            trajectory_deltas = future_data["gt_ego_fut_trajs"]
            future_positions = _restore_trajectory(trajectory_deltas)
            x, y, z = future_data["can_bus"][0:3]
            qx, qy, qz, qw = future_data["can_bus"][3:7]
            q = Quaternion([qw, qx, qy, qz])
            future_positions = np.c_[
                future_positions, np.zeros(future_positions.shape[0])
            ]
            future_positions_global = _transform_to_global(
                future_positions, x, y, z, q
            )
            ego_fut_trajs_restored.append(future_positions_global[1:, :])
    # Transform to ego coordinate frame
    x, y, z = key_frame_data[frame_idx]["can_bus"][0:3]
    qx, qy, qz, qw = key_frame_data[frame_idx]["can_bus"][3:7]
    q = Quaternion([qw, qx, qy, qz])
    ego_future_positions = []
    for traj in ego_fut_trajs_restored:
        traj_in_ego = _transform_to_global(traj, x, y, z, q, inv=True)
        # traj_in_ego = traj_in_ego @ np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]) # rotate 90 degrees, make y-axis the forward direction

        ego_future_positions.append(traj_in_ego[:, :2])
    if ego_future_positions:
        return np.vstack(ego_future_positions)
    else:
        return np.zeros((2, 0))