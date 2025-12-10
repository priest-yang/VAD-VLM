import os
import cv2
import pickle
import argparse
import numpy as np
import multiprocessing
from tqdm import tqdm
import numpy.typing as npt
from pyquaternion import Quaternion
from joblib import Parallel, delayed
from typing import Tuple, List, Optional

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.scene_object import SceneObject
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    StateVector2D,
    TimePoint,
)
from nuplan.planning.training.preprocessing.features.raster_utils import (
    get_agents_raster,
    get_baseline_paths_raster,
    get_roadmap_raster,
    get_ego_raster,
    _get_layer_coords,
    _draw_polygon_image,
    _draw_linestring_image,
    SemanticMapLayer,
)


"""
python tools/nuplan/raster_navigation.py \
    --ann_dir /data/ceph/data/nuplan/ann_files/nuplan_trainval_1111 \
    --save_dir /data/ceph/data/nuplan/dataset/raster \
    --nuplan_data_root /data/ceph/data/nuplan/dataset \
    --use_multiprocessing
"""


parser = argparse.ArgumentParser(description="Draw raster for a given frame index.")
parser.add_argument("--ann_dir", type=str, help="Path to the annotation directory.", default="/data/ceph/data/nuplan/ann_files/temp")
parser.add_argument("--save_dir", type=str, help="Path to save the rasters.", default="/data/ceph/data/nuplan/dataset/vlm_ann_data/raster")
parser.add_argument(
    "--nuplan_data_root",
    type=str,
    help="Path to the nuplan data root.",
    default="/data/ceph/data/nuplan/dataset",
)
parser.add_argument(
    "--save_root",
    type=str,
    help="Path to save the rasters.",
    default="/data/ceph/data/nuplan/dataset/raster",
)
parser.add_argument(
    "--use_multiprocessing",
    action="store_true",
    help="Use multiprocessing to process the annotation files.",
)
parser.add_argument("--overwrite", type=bool, default=True)
args = parser.parse_args()


def get_drivable_area_raster(
    ego_state: EgoState,
    map_api: NuPlanMap,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    raster_shape: Tuple[int, int],
    resolution: float,
    roadblock_ids: Optional[List[int]] = None,
) -> npt.NDArray[np.float32]:
    """
    Construct the map layer of the raster by converting vector map to raster map.
    """
    x_length = x_range[1] - x_range[0]
    y_length = y_range[1] - y_range[0]
    assert x_length == y_length, (
        f"Raster shape is assumed to be square but got width: {x_length} "
        f"and height: {y_length}"
    )

    radius = x_length / 2
    drivable_area_raster = np.zeros(raster_shape, dtype=np.float32)
    coords, lane_ids = [], []

    for map_name in ["ROADBLOCK", "ROADBLOCK_CONNECTOR"]:
        layer_coords, layer_lane_ids = _get_layer_coords(
            ego_state, map_api, SemanticMapLayer[map_name], "polygon", radius
        )
        coords.extend(layer_coords)
        lane_ids.extend(layer_lane_ids)

    if roadblock_ids is not None:
        drivable_lane_ids = set(str(roadblock_id) for roadblock_id in roadblock_ids)
        drivable_coords = [
            coord
            for coord, lane_id in zip(coords, lane_ids)
            if lane_id in drivable_lane_ids
        ]
    else:
        drivable_coords = coords

    drivable_area_raster = _draw_polygon_image(
        drivable_area_raster, drivable_coords, radius, resolution, 0.5
    )

    drivable_area_raster = np.flip(drivable_area_raster, axis=0)
    return np.ascontiguousarray(drivable_area_raster, dtype=np.float32)


class DrawRaster:
    def __init__(self, ann_path: str = None, maps_db: GPKGMapsDB = None):
        """Initialize raster drawing parameters and load map and annotation data."""
        self.x_range = [-51.2, 51.2]
        self.y_range = [-51.2, 51.2]
        self.raster_shape = (512, 512)
        self.resolution = (self.x_range[1] - self.x_range[0]) / self.raster_shape[1]
        self.thickness = 2

        self.map_features = {
            "LANE": 1.0,
            "INTERSECTION": 1.0,
            "STOP_LINE": 0.5,
            "CROSSWALK": 0.5,
        }

        self.ego_para = VehicleParameters(
            vehicle_name="pacifica",
            vehicle_type="gen1",
            width=1.1485 * 2.0,
            front_length=4.049,
            rear_length=1.127,
            wheel_base=3.089,
            cog_position_from_rear_axle=1.67,
            height=1.777,
        )

        # if ann_path is None:
        #     ann_path = os.path.join(NUPLAN_ANN_ROOT, "trainval_filtered.pkl")
        assert ann_path is not None, "ann_path is not provided"
        with open(ann_path, "rb") as f:
            print(f"Loaded annotation data from {ann_path}, please wait...")
            self.key_frame_data = pickle.load(f)
        self.key_frame_data = (
            self.key_frame_data["infos"]
            if "infos" in self.key_frame_data
            else self.key_frame_data
        )
        init_data = self.key_frame_data[0] if self.key_frame_data else None
        if init_data is None:
            print(f"No data found in {ann_path}")
            return

        self.location = init_data["map_location"].replace(".gpkg", "")
        self.maps_db = maps_db
        self.map_api = NuPlanMap(maps_db, self.location)

        ego_longitudinal_offset = 0.0
        ego_width_pixels = int(self.ego_para.width / self.resolution)
        ego_front_length_pixels = int(self.ego_para.front_length / self.resolution)
        ego_rear_length_pixels = int(self.ego_para.rear_length / self.resolution)

        self.ego_raster = get_ego_raster(
            self.raster_shape,
            ego_longitudinal_offset,
            ego_width_pixels,
            ego_front_length_pixels,
            ego_rear_length_pixels,
        )
        self.ego_raster = self._to_rgb_channel(self.ego_raster, "red")

    def draw_frame_raster(self, frame_idx: int) -> npt.NDArray[np.float32]:
        """Draw raster for a given frame index."""
        data = self.key_frame_data[frame_idx]
        self._update_map_api(data["map_location"])

        # Prepare future trajectories
        ego_future_positions = self._prepare_future_trajectories(frame_idx)

        # Get ego state
        self.ego_state = self._get_ego_state(data)

        # Prepare tracked objects
        self.tracked_objects = self._prepare_tracked_objects(data)

        # Generate raster layers
        gt_raster, navigation_raster = self._generate_raster_layers(
            data, ego_future_positions
        )

        # Combine raster layers
        final_gt_raster, final_navigation_raster = self._combine_rasters(
            gt_raster
        ), self._combine_rasters(navigation_raster)
        return np.clip(final_gt_raster, 0, 1), np.clip(final_navigation_raster, 0, 1)

    def draw_sequence_raster(
        self, save_dir: str, log_name: str, start_frame: int, end_frame: int
    ):
        """Draw a sequence of rasters and save as images and video."""
        gt_save_path = os.path.join(save_dir, "gt", log_name)
        navigation_save_path = os.path.join(save_dir, "navigation", log_name)
        os.makedirs(gt_save_path, exist_ok=True)
        os.makedirs(navigation_save_path, exist_ok=True)

        # Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # or 'H264'
        video = cv2.VideoWriter(
            os.path.join(gt_save_path, "0_video.mp4"),
            fourcc,
            10,  # fps
            (self.raster_shape[1], self.raster_shape[0]),
            isColor=True
        )
        
        if not video.isOpened():
            # Fallback to other codecs if H.264 is not available
            video = cv2.VideoWriter(
                os.path.join(gt_save_path, "0_video.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'),
                10,
                (self.raster_shape[1], self.raster_shape[0]),
                isColor=True
            )
        
        for frame_idx in tqdm(
            range(start_frame, min(end_frame, len(self.key_frame_data))),
            desc="Drawing Rasters",
            total=min(end_frame, len(self.key_frame_data)) - start_frame,
        ):
            final_gt_raster, final_navigation_raster = self.draw_frame_raster(frame_idx)
            
            # Convert RGBA to BGR for OpenCV
            gt_img = (final_gt_raster[..., :3] * 255).astype(np.uint8)
            navigation_img = (final_navigation_raster[..., :3] * 255).astype(np.uint8)
            
            # Save images with alpha channel as PNG
            cv2.imwrite(
                os.path.join(gt_save_path, f"{frame_idx}.png"),
                cv2.cvtColor((final_gt_raster * 255).astype(np.uint8), cv2.COLOR_RGBA2BGRA)
            )
            cv2.imwrite(
                os.path.join(navigation_save_path, f"{frame_idx}.png"),
                cv2.cvtColor((final_navigation_raster * 255).astype(np.uint8), cv2.COLOR_RGBA2BGRA)
            )
            
            # Write BGR frame to video
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGBA2BGR)
            video.write(gt_img)
        
        video.release()
        
        # Use ffmpeg to convert the video to a more compatible format (if ffmpeg is available)
        try:
            import subprocess
            input_video = os.path.join(gt_save_path, "0_video.mp4")
            output_video = os.path.join(gt_save_path, "video_h264.mp4")
            subprocess.run([
                'ffmpeg', '-i', input_video,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-y',  # Overwrite output file if it exists
                output_video
            ], check=True)
            # Replace original video with the converted one
            os.replace(output_video, input_video)
            # # Remove the original video file
            # os.remove(input_video)
        except (ImportError, subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"Warning: Could not convert video using ffmpeg: {e}")
        
        print(
            f"Saved rasters and video to {gt_save_path} {navigation_save_path} over {start_frame}-{end_frame}."
        )

    def _update_map_api(self, map_location: str):
        """Update the map API if the location has changed."""
        current_location = map_location.replace(".gpkg", "")
        if self.location != current_location:
            self.map_api = NuPlanMap(self.maps_db, current_location)
            self.location = current_location

    def _transform_to_global(
        self,
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

    def _prepare_future_trajectories(self, frame_idx: int) -> npt.NDArray[np.float32]:
        """Prepare future trajectories for ego vehicle."""
        ego_fut_trajs_restored = []
        for future_offset in [0, 6, 12]:
            future_idx = frame_idx + future_offset
            if future_idx < len(self.key_frame_data):
                future_data = self.key_frame_data[future_idx]
                trajectory_deltas = future_data["gt_ego_fut_trajs"]
                future_positions = self._restore_trajectory(trajectory_deltas)
                x, y, z = future_data["can_bus"][0:3]
                qx, qy, qz, qw = future_data["can_bus"][3:7]
                q = Quaternion([qw, qx, qy, qz])
                future_positions = np.c_[
                    future_positions, np.zeros(future_positions.shape[0])
                ]
                future_positions_global = self._transform_to_global(
                    future_positions, x, y, z, q
                )
                ego_fut_trajs_restored.append(future_positions_global[1:, :])
        # Transform to ego coordinate frame
        x, y, z = self.key_frame_data[frame_idx]["can_bus"][0:3]
        qx, qy, qz, qw = self.key_frame_data[frame_idx]["can_bus"][3:7]
        q = Quaternion([qw, qx, qy, qz])
        ego_future_positions = []
        for traj in ego_fut_trajs_restored:
            traj_in_ego = self._transform_to_global(traj, x, y, z, q, inv=True)
            traj_in_ego = traj_in_ego @ np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])

            ego_future_positions.append(traj_in_ego[:, :2])
        if ego_future_positions:
            return np.vstack(ego_future_positions)
        else:
            return np.zeros((2, 0))

    def _get_ego_state(self, data) -> EgoState:
        """Get the ego state from the data."""
        can_bus = data["can_bus"]
        x, y, z = can_bus[0], can_bus[1], can_bus[2]
        qx, qy, qz, qw = can_bus[3:7]
        vx, vy = can_bus[13], can_bus[14]
        acceleration_x, acceleration_y = can_bus[7], can_bus[8]
        timestamp = data["timestamp"]

        q = Quaternion([qw, qx, qy, qz])
        ego_orientation = q.yaw_pitch_roll[0]
        return EgoState.build_from_rear_axle(
            StateSE2(x, y, ego_orientation),
            tire_steering_angle=0.0,
            vehicle_parameters=self.ego_para,
            time_point=TimePoint(timestamp),
            rear_axle_velocity_2d=StateVector2D(vx, vy),
            rear_axle_acceleration_2d=StateVector2D(acceleration_x, acceleration_y),
        )

    def _prepare_tracked_objects(self, data) -> DetectionsTracks:
        """Prepare tracked objects from data."""
        timestamp = data["timestamp"]
        x, y, z = data["can_bus"][0:3]
        qx, qy, qz, qw = data["can_bus"][3:7]
        q = Quaternion([qw, qx, qy, qz])

        track_gt_boxes = np.array(data["gt_boxes"])
        if track_gt_boxes.size == 0:
            return DetectionsTracks(TrackedObjects([]))
        track_gt_boxes[:, :3] = self._transform_to_global(
            track_gt_boxes[:, :3], x, y, z, q
        )
        # Transform headings to the global frame
        track_gt_boxes[:, -1] = q.yaw_pitch_roll[0] - track_gt_boxes[:, -1]
        # Normalize headings to be within [-π, π]
        track_gt_boxes[:, -1] = np.arctan2(
            np.sin(track_gt_boxes[:, -1]), np.cos(track_gt_boxes[:, -1])
        )

        scene_objects = [
            SceneObject.from_raw_params(
                token=str(i),
                track_token=str(i),
                timestamp_us=timestamp,
                track_id=i,
                center=StateSE2(box[0], box[1], box[-1]),
                size=(box[3], box[4], box[5]),
            )
            for i, box in enumerate(track_gt_boxes)
        ]
        return DetectionsTracks(TrackedObjects(scene_objects))

    def _generate_raster_layers(
        self, data, ego_future_positions
    ) -> Tuple[List[npt.NDArray[np.float32]], List[npt.NDArray[np.float32]]]:
        """Generate the different raster layers."""
        roadmap_raster = get_roadmap_raster(
            self.ego_state,
            self.map_api,
            self.map_features,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.resolution,
        )
        roadmap_raster = self._to_rgb_channel(roadmap_raster, "roadmap_raster")

        agents_raster = get_agents_raster(
            self.ego_state,
            self.tracked_objects,
            self.x_range,
            self.y_range,
            self.raster_shape,
        )
        agents_raster = self._to_rgb_channel(agents_raster, "agents_raster")

        baseline_paths_raster = get_baseline_paths_raster(
            self.ego_state,
            self.map_api,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.resolution,
            self.thickness,
        )
        baseline_paths_raster = self._to_rgb_channel(baseline_paths_raster, "baseline_paths_raster")

        drivable_area_raster = get_drivable_area_raster(
            self.ego_state,
            self.map_api,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.resolution,
            data.get("roadblock_ids"),
        )
        drivable_area_raster = self._to_rgb_channel(drivable_area_raster, "drivable_area_raster")

        radius = (self.x_range[1] - self.x_range[0]) / 2
        ego_fut_trajs_raster = np.zeros(self.raster_shape)
        ego_fut_trajs_raster = _draw_linestring_image(
            ego_fut_trajs_raster,
            [ego_future_positions],
            radius,
            self.resolution,
            10,
            [1],
        )
        ego_fut_trajs_raster = self._to_rgb_channel(ego_fut_trajs_raster, "ego_fut_trajs_raster")

        gt_raster = [
            roadmap_raster,
            drivable_area_raster, 
            baseline_paths_raster, 
            agents_raster,
            ego_fut_trajs_raster,
            self.ego_raster,
        ]

        navigation_raster = [
            roadmap_raster,
            drivable_area_raster,
            self.ego_raster,
        ]

        return gt_raster, navigation_raster

    def _restore_trajectory(
        self, trajectory_deltas: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Restore the trajectory deltas to get the future trajectory."""
        initial_position = trajectory_deltas[0]
        recovered_ego_fut_trajs = np.vstack(
            [initial_position, np.cumsum(trajectory_deltas, axis=0)]
        )
        return recovered_ego_fut_trajs

    def _to_rgb_channel(
        self, raster: npt.NDArray[np.float32], color: str
    ) -> npt.NDArray[np.float32]:
        """Convert black and white raster to an RGBA channel."""
        color_map = {  # RGBA (Alpha channel added)
            "red": [1, 0, 0, 1],
            "green": [0, 1, 0, 1],
            "blue": [0, 0, 1, 1],
            # Semi-transparent colors for different raster layers
            "roadmap_raster": [0.5, 0.5, 0.5, 0.3],  # Gray with 0.7 opacity
            "drivable_area_raster": [0, 1, 0, 1],  # Green with 0.5 opacity
            "baseline_paths_raster": [0.9, 0.9, 0.9, 0.5],  # Yellow with 0.8 opacity
            "agents_raster": [181, 3, 0, 0.9],  # Blue with 0.9 opacity
            "ego_fut_trajs_raster": [0, 0, 1, 0.6],  # Blue
        }
        color_array = np.array(color_map[color], dtype=np.float32)
        # Expand raster to 4 channels (RGBA)
        return raster[..., np.newaxis] * color_array

    def _combine_rasters(
        self, rasters: List[npt.NDArray[np.float32]]
    ) -> npt.NDArray[np.float32]:
        """Combine multiple rasters into a final raster image with alpha blending."""
        # Initialize with transparent background
        final_raster = np.zeros((*self.raster_shape, 4), dtype=np.float32)
        
        for raster in rasters:
            # Skip if raster is empty
            if not np.any(raster):
                continue
            
            # Get alpha values
            alpha = raster[..., 3:4]
            # Get RGB values
            rgb = raster[..., :3]
            
            # Perform alpha blending
            old_alpha = final_raster[..., 3:4]
            new_alpha = alpha + old_alpha * (1 - alpha)
            
            # Avoid division by zero
            mask = new_alpha > 0
            
            # Blend RGB channels
            final_raster[..., :3][mask[..., 0]] = (
                (alpha * rgb + old_alpha * final_raster[..., :3] * (1 - alpha)) 
                / np.maximum(new_alpha, 1e-8)
            )[mask[..., 0]]
            
            # Update alpha channel
            final_raster[..., 3:4] = new_alpha

        return final_raster


def process_annotation_file(ann_path: str, save_dir: str, maps_db: GPKGMapsDB):
    """Process a single annotation file."""
    draw_raster = DrawRaster(ann_path=ann_path, maps_db=maps_db)
    if (
        draw_raster.key_frame_data is None or len(draw_raster.key_frame_data) == 0
    ):  # no data in the annotation file
        return

    log_name = os.path.basename(ann_path).replace(".pkl", "")
    draw_raster.draw_sequence_raster(
        save_dir, log_name, start_frame=0, end_frame=len(draw_raster.key_frame_data)
    )


def process_annotation_file_with_error_handling(
    ann_path: str, save_dir: str, maps_db: GPKGMapsDB
):
    """Process a single annotation file."""
    try:
        process_annotation_file(ann_path, save_dir, maps_db)
    except Exception as e:
        print(f"Error processing annotation file {ann_path}: {e}")


if __name__ == "__main__":
    ann_dir = args.ann_dir
    save_dir = args.save_dir
    use_multiprocessing = args.use_multiprocessing
    ann_files = [
        os.path.join(ann_dir, file)
        for file in os.listdir(ann_dir)
        if file.endswith(".pkl")
    ]
    NUPLAN_DATA_ROOT = args.nuplan_data_root
    NUPLAN_MAPS_ROOT = os.path.join(NUPLAN_DATA_ROOT, "maps")

    maps_db = GPKGMapsDB(map_root=NUPLAN_MAPS_ROOT, map_version="nuplan-maps-v1.0")

    if use_multiprocessing:
        print("Using multiprocessing to process the annotation files...")
        available_cpu = multiprocessing.cpu_count()
        print(f"Available CPU: {available_cpu}")
        jobs_needed = len(ann_files)
        used_cpu = max(1, min(available_cpu - 1, jobs_needed))
        Parallel(n_jobs=used_cpu)(
            delayed(process_annotation_file)(ann_path, save_dir, maps_db)
            for ann_path in tqdm(ann_files)
        )
    else:
        print("Using single process to process the annotation files...")
        for ann_path in tqdm(ann_files):
            process_annotation_file(ann_path, save_dir, maps_db)


"""
python tools/nuplan/raster_navigation.py \
    --ann_dir /data/ceph/data/nuplan/ann_files/nuplan_trainval_1111 \
    --save_dir /data/ceph/data/nuplan/dataset/raster \
    --nuplan_data_root /data/ceph/data/nuplan/dataset \
    --use_multiprocessing
"""
