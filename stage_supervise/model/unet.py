
import torch
import torch.nn as nn
from monai.networks.nets import UNETR

import model.pps_net.utils.optical_flow_funs as OF
from model.pps_net.losses import (
    calculate_per_pixel_lighting,
    MidasLoss, VNL_Loss,
    calculate_pps_supp_loss)
from model.pps_net.utils.utils import get_normals_from_depth


class Unet(nn.Module):
    def __init__(self, input_size=336):
        super().__init__()
        self.input_size = (input_size,)*2
        self.unet = UNETR(
            in_channels=3,
            out_channels=1,
            img_size=self.input_size,
            feature_size=32,
            norm_name='batch',
            spatial_dims=2,
        )

        self.vnl_loss = VNL_Loss(1.0, 1.0, self.input_size)
        self.midas_loss = MidasLoss()

    def pred(self, x, **kwargs):
        pred = self.sforward(x)[:, 0]
        return pred

    def forward(self, batch):
        img_tensor = batch['rgb']
        pred = self.sforward(img_tensor)[:, 0]
        return pred

    def sforward(self, x):
        return torch.clamp(self.unet(x) / 10, 0, 1)

    def train_step(self, batch):
        img_tensor = batch['rgb']
        gt = batch['depth']
        light_data = batch['light_data']
        n_intrinsics = batch['n_intrinsics']

        pred = self.sforward(img_tensor)[:, 0]
        ref_dirs = OF.get_camera_pixel_directions(img_tensor.shape[2:4], n_intrinsics,
                                                  normalized_intrinsics=True)
        # if fc is not None:
        #     pred, optimal_scale = scale_predictions(gt[:, 0], pred, return_scale=True)
        #     self.wt.write(f'{float(optimal_scale)}\n')
        #     print(optimal_scale)
        normal_pred, _ = get_normals_from_depth(pred.unsqueeze(1), n_intrinsics)
        normal_gt, _ = get_normals_from_depth(gt, n_intrinsics)

        pc_preds = pred.unsqueeze(3) * ref_dirs
        l_pred, a_pred = calculate_per_pixel_lighting(pc_preds, *light_data)

        pc_gt = gt.squeeze(1).unsqueeze(3) * ref_dirs
        l_gt, a_gt = calculate_per_pixel_lighting(pc_gt, *light_data)

        pps_sup_loss = calculate_pps_supp_loss(l_pred, a_pred, normal_pred, l_gt, a_gt, normal_gt)
        pred = pred.unsqueeze(1)
        vnl_loss = self.vnl_loss(pred, gt)
        mask_valid = torch.ones_like(gt).bool()
        ssi_loss, reg_loss = self.midas_loss(pred, gt, mask_valid)

        l1_loss = torch.abs(gt - pred).mean()
        rmse = torch.sqrt(torch.mean((gt - pred) ** 2))
        epsilon = 1e-8
        abs_rel = torch.abs(gt - pred) / (gt + epsilon)
        sq_rel = torch.mean(((gt - pred) / (gt + epsilon)) ** 2)
        si = torch.max(gt / (pred + epsilon), pred / (gt + epsilon))
        s1 = (si < 1.05).float().mean()
        s2 = (si < (1.05 ** 2)).float().mean()
        s3 = (si < (1.05 ** 3)).float().mean()
        a1 = (abs_rel < 0.1).float().mean()

        loss = pps_sup_loss + 10 * vnl_loss + ssi_loss + 0.1 * l1_loss + 0.1 * reg_loss
        loss_dict = {
            'loss': loss,
            'pps_sup_loss': pps_sup_loss,
            'vnl_loss': vnl_loss,
            'ssi_loss': ssi_loss,
            'reg_loss': reg_loss,
            'l1_loss': l1_loss,
            'abs_rel': abs_rel.mean(),
            'sq_rel': sq_rel,
            'a1': a1,
            's1': s1,
            's2': s2,
            's3': s3,
            'rsme': rmse
        }
        return loss_dict
