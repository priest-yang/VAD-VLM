from pathlib import Path
from typing import Dict
import numpy as np
import torch

from lwpybase.math.rotation import Quaternion, Isometry
from lwpybase.sensors.sensor_configs import VehicleType
from uw_engine.nodes.e2e_ad.models.model import ADModel, NuSCategory


class UniAD(ADModel):
    default_cfg_name = "VAD/VAD_base_stage_2.py"
    default_ckpt_path = Path("/home/lightwheel/vad/ckpts/VAD_base.pth")

    def __init__(self, cfg_name: str, ckpt_path: Path, device: str, fuse_conv_bn: bool = True, vehicle: VehicleType = VehicleType.nuscenes):
        # from projects.mmdet3d_plugin.uniad import UniAD
        import adbase
        from mmcv import Config
        # from mmdet3d.models import build_model
        # from mmcv.runner import load_checkpoint, wrap_fp16_model
        # from mmcv.cnn import fuse_conv_bn
        from mmcv.utils import load_checkpoint, wrap_fp16_model
        from mmcv.models import build_model, fuse_conv_bn
        ad_config_base_path = Path(adbase.__file__).parent / "uniad" / "configs"
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

    def prepare_data(self, img: torch.Tensor, info: Dict, timestamp: int):
        input_dict = self.get_data_info(info, timestamp)
        example = self.pipeline(img, input_dict)
        return example

    def get_data_info(self, info: Dict, timestamp: int):
        # TODO: parse map gt

        input_dict = dict(
            can_bus=info["can_bus"],
            timestamp=[timestamp / 1e9],
            command=[info["command"]],
        )

        l2e_iso = info["l2e_iso"]
        l2e_t = l2e_iso.translation
        l2e_quatd = l2e_iso.quaternion
        l2e_r_mat = l2e_quatd.to_rotation_matrix()
        e2g_iso = info["e2g_iso"]
        e2g_t = e2g_iso.translation
        e2g_r_mat = e2g_iso.quaternion.to_rotation_matrix()
        l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
        l2g_t = l2e_t @ e2g_r_mat.T + e2g_t

        input_dict.update(
            dict(
                l2g_r_mat=l2g_r_mat.astype(np.float32),
                l2g_t=l2g_t.astype(np.float32),
                lidar2img=info["lidar2img_rts"],
                can_bus_dict=info["can_bus_dict"],
                # cam_intrinsic=info["cam_intrinsics"],
                # lidar2cam=info["lidar2cam_rts"],
            )
        )

        return input_dict

    def pipeline(self, img: torch.Tensor, input_dict: Dict):
        result = input_dict
        result['img'] = [img[..., i] for i in range(img.shape[-1])]
        result['img_shape'] = img.shape
        result['ori_shape'] = img.shape
        result['pad_shape'] = img.shape
        result['scale_factor'] = 1.0
        result['img_norm_cfg'] = dict(
            mean=np.array([103.53, 116.28, 123.675], dtype=np.float32),
            std=np.array([1., 1., 1.], dtype=np.float32),
            to_rgb=False)
        input_dict['img'] = [self.normalize_torch(img, **input_dict['img_norm_cfg']) for img in input_dict['img']]
        input_dict['pad_size_divisor'] = 32
        padded_img = [self.impad_to_multiple_torch(img, size_divisor=input_dict['pad_size_divisor'], pad_val=0) for img in input_dict['img']]
        input_dict['ori_shape'] = [img.shape for img in input_dict['img']]
        input_dict['img_shape'] = [img.shape for img in padded_img]
        input_dict['pad_shape'] = [img.shape for img in padded_img]
        padded_img = [img.permute(2, 0, 1) for img in padded_img]
        padded_img = torch.stack(padded_img, dim=0).unsqueeze(0).unsqueeze(0)
        input_dict['img'] = padded_img
        return input_dict

    def forward(self, data):
        with torch.no_grad():
            result = self.model.uw_forward_inference(
                data["img"], data["img_shape"],
                data["lidar2img"], data["can_bus"],
                torch.from_numpy(data["l2g_t"]).unsqueeze(0).to(self.device),
                torch.from_numpy(data["l2g_r_mat"]).unsqueeze(0).to(self.device),
                [torch.tensor(data["timestamp"], device=self.device, dtype=torch.float64)],
                [torch.tensor(data["command"], device=self.device)]
            )
        result[0]["can_bus_dict"] = data["can_bus_dict"]
        result[0]["l2g_t"] = data["l2g_t"]
        result[0]["l2g_r_mat"] = data["l2g_r_mat"]
        result[0]["command"] = data["command"]
        return result

    def build_inference(self, model_outputs, infer_dump_keys):
        model_outputs = model_outputs[0]
        # parse trajectory
        traj = model_outputs["planning"]["result_planning"]["sdc_traj"].tolist()[0]
        traj = [self.lidar2sae(np.array([x[0], x[1], 0])) for x in traj]
        e2g_t = model_outputs["can_bus_dict"]["pose"]["location"]
        e2g_quatd = model_outputs["can_bus_dict"]["pose"]["orientation"]
        e2g_iso = Isometry.from_transform(translation=e2g_t, quaternion=e2g_quatd)
        traj_g = [e2g_iso.apply(p) for p in traj]
        model_outputs["traj"] = traj
        model_outputs["traj_g"] = traj_g

        model_outputs["command"] = model_outputs["command"][0]
        model_outputs["lane"] = model_outputs["pts_bbox"]['lane'].tolist()
        model_outputs["drivable"] = model_outputs["pts_bbox"]['drivable'].tolist()
        model_outputs.pop("pts_bbox")

        # parse object detection results
        lidar_bboxes = model_outputs["boxes_3d"].tensor.numpy()
        labels = []
        scores = model_outputs["scores_3d"].numpy().tolist()
        labels_res = model_outputs["labels_3d"].numpy().tolist()
        track_ids = model_outputs["track_ids"].numpy().tolist()
        for i in range(len(labels_res)):
            yaw = lidar_bboxes[i][6]
            quat = Quaternion.from_rpy([0, 0, -yaw])
            center = lidar_bboxes[i][:3].tolist()
            center[2] += lidar_bboxes[i][5] / 2
            iso = Isometry.from_transform(translation=center, quaternion=quat)
            velocity = [float(lidar_bboxes[i][7]), float(lidar_bboxes[i][8]), 0]
            label_dict = {
                "center": iso.translation.tolist(),
                "orientation": iso.quaternion.to_array().tolist(),
                "size": [float(lidar_bboxes[i][3]), float(lidar_bboxes[i][4]), float(lidar_bboxes[i][5])],
                "object_type": NuSCategory(labels_res[i]).name,
                "object_id": float(track_ids[i]),
                "is_ego": False,
                "velocity": velocity,
                "score": scores[i]
            }
            labels.append(label_dict)
        model_outputs["labels"] = labels

        outputs = [{}]
        for key in infer_dump_keys:
            if key in model_outputs:
                outputs[0][key] = model_outputs[key]
        return outputs
