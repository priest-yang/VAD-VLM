from pathlib import Path
from typing import Dict
import torch
import numpy as np
from torchvision import transforms
import copy

from lwpybase.math.rotation import Quaternion, Isometry
from lwpybase.sensors.sensor_configs import VehicleType
from uw_engine.nodes.e2e_ad.models.mmcv_model import MMCVModel


meta_keys = ['filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename', 'transformation_3d_flow', 'scene_token', 'can_bus', 'lidar2global', 'lidar2global']
keys = ['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'fut_valid_flag', 'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels', 'can_bus_dict', 'can_bus_dict']


class VAD(MMCVModel):
    default_cfg_name = "VAD/VAD_base_e2e.py"
    default_ckpt_path = Path("/data/nas/main/models/e2e_ad/vad/VAD_base.pth")

    def __init__(
        self,
        logger,
        cfg_name: str,
        ckpt_path: Path,
        device: str,
        fuse_conv_bn: bool = True,
        vehicle: VehicleType = VehicleType.nuscenes
    ):
        super().__init__(logger, cfg_name, ckpt_path, device, fuse_conv_bn, vehicle)
        self.logger.debug("start import model")
        import adbase
        from mmcv import Config
        from mmcv.utils import load_checkpoint, wrap_fp16_model
        from mmcv.models import build_model, fuse_conv_bn
        self.logger.debug("done import model")

        ad_config_base_path = Path(adbase.__file__).parent / "vad" / "configs"
        cfg_name = cfg_name or self.default_cfg_name
        ckpt_path = ckpt_path or self.default_ckpt_path
        cfg = Config.fromfile(ad_config_base_path / cfg_name)
        model = build_model(cfg.model)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, str(ckpt_path), map_location='cpu')
        if fuse_conv_bn:
            model = fuse_conv_bn(model)
        self.vehicle = vehicle

        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
        self.device = device
        self.model = model.to(device).eval()
        self._get_pipeline(cfg)
        self.ego_his_poses = []
        self.cfg = cfg
        self.meta_actions = cfg.meta_actions

    def prepare_data(self, img: torch.Tensor, info: Dict, timestamp: int):
        input_dict = self.get_data_info(info, timestamp)

        input_dict['img'] = img
        example = self.pipeline(input_dict)
        example["img_metas"] = [[meta.data] for meta in example["img_metas"]]
        example["img"] = [img.data.unsqueeze(0) for img in example["img"]]
        # bypass processing in pipeline & add missing keys
        example['ego_fut_cmd'] = input_dict['ego_fut_cmd_']
        example['fut_valid_flag'] = [[flag] for flag in example['fut_valid_flag']]
        example['lidar2global'] = input_dict["lidar2global"]
        example['can_bus_dict'] = input_dict["can_bus_dict"]
        # example = input_dict
        return example

    def get_data_info(self, info: Dict, timestamp: int):
        input_dict = dict(
            ego2global_translation=info["e2g_iso"].translation,
            ego2global_rotation=info["e2g_iso"].quaternion,
            lidar2ego_translation=info["l2e_iso"].translation,
            lidar2ego_rotation=info["l2e_iso"].quaternion,
            can_bus=info["can_bus"],
            can_bus_dict=info["can_bus_dict"],
            scene_token=info["scene_token"],
            timestamp=[int(timestamp) * 1e-9 % 1e6],
            ego_fut_cmd_=info["command"],
        )

        input_dict["fut_valid_flag"] = False

        ego_lcf_feat = np.zeros(9)
        ego_lcf_feat[:2] = input_dict["can_bus"][13:15]
        ego_lcf_feat[2:4] = input_dict["can_bus"][7:9]
        ego_lcf_feat[4] = input_dict["can_bus"][12]
        ego_lcf_feat[5:7] = np.array([4.084, 1.85])  # ego length and width
        ego_lcf_feat[7] = np.sqrt(ego_lcf_feat[0]**2 + ego_lcf_feat[1]**2)
        ego_lcf_feat[8] = 2 * input_dict["can_bus_dict"]["steering"] / 2.588  # compute kappa from steering
        ego_lcf_feat = np.array(ego_lcf_feat)
        input_dict["ego_lcf_feat"] = torch.from_numpy(ego_lcf_feat).unsqueeze(0).unsqueeze(0).unsqueeze(0).float()

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = info["l2e_iso"].quaternion.to_rotation_matrix()
        lidar2ego[:3, 3] = info["l2e_iso"].translation
        input_dict["lidar2ego"] = lidar2ego

        input_dict.update(
            dict(
                lidar2img=info["lidar2img_rts"],
                camera2ego=info["camera2ego"],
                # cam_intrinsic=info["cam_intrinsics"],
                # lidar2cam=info["lidar2cam_rts"],
            )
        )

        ego2global = np.eye(4)
        ego2global[:3, :3] = info["e2g_iso"].quaternion.to_rotation_matrix()
        ego2global[:3, 3] = info["e2g_iso"].translation
        lidar2global = ego2global @ lidar2ego
        input_dict["lidar2global"] = lidar2global

        self.ego_his_poses.append(lidar2global[:3, 3])
        if len(self.ego_his_poses) > 40:
            del self.ego_his_poses[0:5]

        # TODO: prepare ego_his_trajs, which is 2 timestamps before the current timestamp
        # The history trajectories are in the lidar frame of the current vehicle
        his_ts = 2
        ego_his_trajs = np.zeros((his_ts + 1, 3))
        ego_his_trajs_diff = np.zeros((his_ts + 1, 3))
        curr_t = len(self.ego_his_poses) - 1
        dt = 1
        for i in range(his_ts, -1, -1):
            if curr_t < len(self.ego_his_poses):
                ego_his_trajs[i] = self.ego_his_poses[curr_t]
                has_prev = curr_t - dt >= 0
                has_next = curr_t + dt <= len(self.ego_his_poses) - 1
                if has_next:
                    ego_his_trajs_diff[i] = self.ego_his_poses[curr_t + 1] - self.ego_his_poses[curr_t]
                curr_t = curr_t - dt if has_prev else len(self.ego_his_poses)
            else:
                ego_his_trajs[i] = ego_his_trajs[i + 1] - ego_his_trajs_diff[i + 1]
                ego_his_trajs_diff[i] = ego_his_trajs_diff[i + 1]
        # global to ego at lcf
        ego_his_trajs = ego_his_trajs - lidar2global[:3, 3]
        rot_mat = Quaternion.from_rotation_matrix(lidar2global[:3, :3]).inverse().to_rotation_matrix()
        ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
        ego_his_trajs = ego_his_trajs[1:] - ego_his_trajs[:-1]
        input_dict["ego_his_trajs"] = ego_his_trajs[:, :2].astype(np.float32)

        return input_dict

    def forward(self, data):
        with torch.no_grad():
            result = self.model.uw_forward_inference(
                img_metas=data["img_metas"],
                img=data["img"],
                fut_valid_flag=data["fut_valid_flag"],
                ego_his_trajs=data["ego_his_trajs"],
                ego_lcf_feat=data["ego_lcf_feat"],
                ego_fut_cmd=np.eye(len(self.meta_actions))[self.meta_actions.index(data["ego_fut_cmd"])],  # torch.from_numpy(data["ego_fut_cmd"]).to(self.device)],
                # ego_fut_cmd = np.eye(3)[data["ego_fut_cmd"]]#torch.from_numpy(data["ego_fut_cmd"]).to(self.device)
            )
        result[0]["lidar2global"] = data["lidar2global"]
        result[0]["can_bus_dict"] = data["can_bus_dict"]
        return result

    def build_inference(self, model_outputs, infer_dump_keys):
        """
        Build the inference results from ad model outputs for downstream tasks
        Input:
            model_outputs: List[Dict[str, Any]]
        Output:
            outputs: List[Dict[str, Any]]
        """
        # parse input
        traj = model_outputs[0]["pts_bbox"]["trajs_3d"].numpy()  # .tolist()#[100,6,12]
        boxes = model_outputs[0]["pts_bbox"]["boxes_3d"]
        box_labels = model_outputs[0]["pts_bbox"]["labels_3d"].numpy()  # 100
        box_score = model_outputs[0]["pts_bbox"]["scores_3d"].numpy()
        # map_boxes=model_outputs[0]["pts_bbox"]["map_boxes_3d"].numpy().tolist()#[50,4]
        # map_scores=model_outputs[0]["pts_bbox"]["map_scores_3d"].numpy().tolist()#[50]
        # map_labels=model_outputs[0]["pts_bbox"]["map_labels_3d"].numpy().tolist()#[50]0,1,2
        # map_pts_3d=model_outputs[0]["pts_bbox"]["map_pts_3d"].numpy().tolist()#[50,20,2]
        can_bus_dict = model_outputs[0]["can_bus_dict"]
        if self.vehicle.is_nuplan():
            mapped_class_names = [
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone', 'generic_object',
                'czone_sign', 'vehicle'
            ]
        else:   # nuscenes or by default
            mapped_class_names = [
                'car', 'truck', 'construction_vehicle', 'bus',
                'trailer', 'barrier', 'motorcycle', 'bicycle',
                'pedestrian', 'traffic_cone'
            ]
        labels = []
        for box_ind in range(traj.shape[0]):
            if box_score[box_ind] <= 0.6:
                continue
            label = {}
            label["object_id"] = box_ind
            label["object_type"] = str(mapped_class_names[box_labels[box_ind]])
            label['center'] = boxes.center[box_ind].numpy()
            box_q = Quaternion.from_rpy([0, 0, boxes.yaw[box_ind]])
            box_quat = [box_q.w, box_q.x, box_q.y, box_q.z]
            label['orientation'] = box_quat
            # if self.prev_labels != None:
            #     self.prev_labels[]
            # else:
            label['velocity'] = box_q.rotate([1.,0.,0.])
            label['size'] = boxes.dims[box_ind].numpy()
            labels.append(label)
        # self.prev_labels = labels
        ego_fut_preds = model_outputs[0]["pts_bbox"]["ego_fut_preds"].numpy()  # [3,6,2]
        ego_fut_cmd = model_outputs[0]["pts_bbox"]["ego_fut_cmd"].tolist()  # [3]
        pos = model_outputs[0]["pts_bbox"]["pos"]
        angle = model_outputs[0]["pts_bbox"]["angle"]
        ego_fut_cmd_idx = torch.nonzero(torch.Tensor(ego_fut_cmd))
        ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]
        ego_fut_pred = [[*row, 0] for row in ego_fut_pred]
        ego_fut_pred = np.cumsum(ego_fut_pred, axis=0)
        ego_fut_pred_local = copy.deepcopy(ego_fut_pred)
        ego_fut_pred_local = [self.lidar2sae(np.array([x[0], x[1], 0])) for x in ego_fut_pred_local]
        ego_fut_pred = ego_fut_pred + pos

        q = Quaternion.from_rpy([0, 0, angle])
        quat = [q.w, q.x, q.y, q.z]
        # Rz_neg_90 = np.array([
        #             [0, -1, 0],
        #             [1,  0, 0],
        #             [0,  0, 1]
        #         ])
        # M = np.eye(4)  # 单位矩阵
        # M[:3, :3] = Rz_neg_90
        # e2g_iso = Isometry.from_transform(translation=pos, quaternion=quat)
        ego_pred_homo = np.hstack((np.array(ego_fut_pred), np.ones((6, 1))))
        transformed_points = np.dot(ego_pred_homo, model_outputs[0]["lidar2global"].T)  # model_outputs[0]["lidar2global"]
        e2g_fut_pred = transformed_points[:, :3] + pos
        l2g_t = model_outputs[0]["lidar2global"][:3, 3]
        l2g_r_mat = model_outputs[0]["lidar2global"][:3, :3]
        # TODO: parse mapFormer results
        outputs = [{
            # "boxes":boxes,
            "traj": ego_fut_pred_local,
            "command": ego_fut_cmd,
            "traj_g": e2g_fut_pred,
            "labels": labels,
            "pos": pos,
            "quat": quat,
            "l2g_t": l2g_t,
            "l2g_r_mat": l2g_r_mat,
            "can_bus_dict": can_bus_dict
        }]
        return outputs
