# Based on https://github.com/EPFL-VILAB/omnidata

import numpy as np
import torch
import torch.nn as nn


def masked_l1_loss(preds, target, mask_valid):
    element_wise_loss = abs(preds - target)
    element_wise_loss[~mask_valid] = 0
    return element_wise_loss.sum() / mask_valid.sum()


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / (det[valid] + 1e-6)
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / (det[valid] + 1e-6)

    return x_0, x_1


def masked_shift_and_scale(depth_preds, depth_gt, mask_valid):
    depth_preds_nan = depth_preds.clone()
    depth_gt_nan = depth_gt.clone()
    depth_preds_nan[~mask_valid] = torch.nan
    depth_gt_nan[~mask_valid] = torch.nan

    mask_diff = mask_valid.view(mask_valid.size()[:2] + (-1,)).sum(-1, keepdims=True) + 1

    t_gt = depth_gt_nan.view(depth_gt_nan.size()[:2] + (-1,)).nanmedian(-1, keepdims=True)[0].unsqueeze(-1)
    t_gt[torch.isnan(t_gt)] = 0
    diff_gt = torch.abs(depth_gt - t_gt)
    diff_gt[~mask_valid] = 0
    s_gt = (diff_gt.view(diff_gt.size()[:2] + (-1,)).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)
    depth_gt_aligned = (depth_gt - t_gt) / (s_gt + 1e-6)

    t_pred = depth_preds_nan.view(depth_preds_nan.size()[:2] + (-1,)).nanmedian(-1, keepdims=True)[0].unsqueeze(-1)
    t_pred[torch.isnan(t_pred)] = 0
    diff_pred = torch.abs(depth_preds - t_pred)
    diff_pred[~mask_valid] = 0
    s_pred = (diff_pred.view(diff_pred.size()[:2] + (-1,)).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)
    depth_pred_aligned = (depth_preds - t_pred) / (s_pred + 1e-6)

    return depth_pred_aligned, depth_gt_aligned


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class SSIMAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, depth_preds, depth_gt, mask_valid):
        depth_pred_aligned, depth_gt_aligned = masked_shift_and_scale(depth_preds, depth_gt, mask_valid)
        ssi_mae_loss = masked_l1_loss(depth_pred_aligned, depth_gt_aligned, mask_valid)
        return ssi_mae_loss


class GradientMatchingTerm(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class MidasLoss(nn.Module):
    def __init__(self, scales=4, reduction='image-based'):
        super().__init__()

        self.__ssi_mae_loss = SSIMAE()
        self.__gradient_matching_term = GradientMatchingTerm(scales=scales, reduction=reduction)
        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        prediction_inverse = 1 / (prediction.squeeze(1) + 1e-6)
        target_inverse = 1 / (target.squeeze(1) + 1e-6)
        ssi_loss = self.__ssi_mae_loss(prediction, target, mask)

        scale, shift = compute_scale_and_shift(prediction_inverse, target_inverse, mask.squeeze(1))
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction_inverse + shift.view(-1, 1, 1)
        reg_loss = self.__gradient_matching_term(self.__prediction_ssi, target_inverse, mask.squeeze(1))

        return ssi_loss, reg_loss


class VNL_Loss(nn.Module):
    """
    Virtual Normal Loss Function.
    """

    def __init__(self, focal_x, focal_y, input_size,
                 delta_cos=0.867, delta_diff_x=0.01,
                 delta_diff_y=0.01, delta_diff_z=0.01,
                 delta_z=0.0001, sample_ratio=0.15):
        super(VNL_Loss, self).__init__()
        self.fx = torch.tensor([focal_x], dtype=torch.float32)  # .to(cuda0)
        self.fy = torch.tensor([focal_y], dtype=torch.float32)  # .to(cuda0)
        self.input_size = input_size
        self.u0 = torch.tensor(input_size[1] // 2, dtype=torch.float32)  # .to(cuda0)
        self.v0 = torch.tensor(input_size[0] // 2, dtype=torch.float32)  # .to(cuda0)
        self.init_image_coor()
        self.delta_cos = delta_cos
        self.delta_diff_x = delta_diff_x
        self.delta_diff_y = delta_diff_y
        self.delta_diff_z = delta_diff_z
        self.delta_z = delta_z
        self.sample_ratio = sample_ratio

    def init_image_coor(self):
        x_row = np.arange(0, self.input_size[1])
        x = np.tile(x_row, (self.input_size[0], 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy())  # .to(cuda0)
        self.u_u0 = x - self.u0

        y_col = np.arange(0, self.input_size[0])  # y_col = np.arange(0, height)
        y = np.tile(y_col, (self.input_size[1], 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy())  # .to(cuda0)
        self.v_v0 = y - self.v0

    def transfer_xyz(self, depth):
        # print('!!!!!!!!!!!!!!!111111 ', self.u_u0.device, torch.abs(depth).device, self.fx.device)
        x = self.u_u0 * torch.abs(depth) / self.fx
        y = self.v_v0 * torch.abs(depth) / self.fy
        z = depth
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1)  # [b, h, w, c]
        return pw

    def select_index(self):
        valid_width = self.input_size[1]
        valid_height = self.input_size[0]
        num = valid_width * valid_height
        p1 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p1)
        p2 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p2)
        p3 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p3)

        p1_x = p1 % self.input_size[1]
        p1_y = (p1 / self.input_size[1]).astype(int)

        p2_x = p2 % self.input_size[1]
        p2_y = (p2 / self.input_size[1]).astype(int)

        p3_x = p3 % self.input_size[1]
        p3_y = (p3 / self.input_size[1]).astype(int)
        p123 = {'p1_x': p1_x, 'p1_y': p1_y, 'p2_x': p2_x, 'p2_y': p2_y, 'p3_x': p3_x, 'p3_y': p3_y}
        return p123

    def form_pw_groups(self, p123, pw):
        """
        Form 3D points groups, with 3 points in each grouup.
        :param p123: points index
        :param pw: 3D points
        :return:
        """
        p1_x = p123['p1_x']
        p1_y = p123['p1_y']
        p2_x = p123['p2_x']
        p2_y = p123['p2_y']
        p3_x = p123['p3_x']
        p3_y = p123['p3_y']

        pw1 = pw[:, p1_y, p1_x, :]
        pw2 = pw[:, p2_y, p2_x, :]
        pw3 = pw[:, p3_y, p3_x, :]
        # [B, N, 3(x,y,z), 3(p1,p2,p3)]
        pw_groups = torch.cat([pw1[:, :, :, np.newaxis], pw2[:, :, :, np.newaxis], pw3[:, :, :, np.newaxis]], 3)
        return pw_groups

    def filter_mask(self, p123, gt_xyz, delta_cos=0.867,
                    delta_diff_x=0.005,
                    delta_diff_y=0.005,
                    delta_diff_z=0.005):
        pw = self.form_pw_groups(p123, gt_xyz)
        pw12 = pw[:, :, :, 1] - pw[:, :, :, 0]
        pw13 = pw[:, :, :, 2] - pw[:, :, :, 0]
        pw23 = pw[:, :, :, 2] - pw[:, :, :, 1]
        # ignore linear
        pw_diff = torch.cat([pw12[:, :, :, np.newaxis], pw13[:, :, :, np.newaxis], pw23[:, :, :, np.newaxis]],
                            3)  # [b, n, 3, 3]
        m_batchsize, groups, coords, index = pw_diff.shape
        proj_query = pw_diff.view(m_batchsize * groups, -1, index).permute(0, 2,
                                                                           1)  # (B* X CX(3)) [bn, 3(p123), 3(xyz)]
        proj_key = pw_diff.view(m_batchsize * groups, -1, index)  # B X  (3)*C [bn, 3(xyz), 3(p123)]
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize * groups, index, 1), q_norm.view(m_batchsize * groups, 1, index))  # []
        energy = torch.bmm(proj_query, proj_key)  # transpose check [bn, 3(p123), 3(p123)]
        norm_energy = energy / (nm + 1e-8)
        norm_energy = norm_energy.view(m_batchsize * groups, -1)
        mask_cos = torch.sum((norm_energy > delta_cos) + (norm_energy < -delta_cos), 1) > 3  # igonre
        mask_cos = mask_cos.view(m_batchsize, groups)
        # ignore padding and invilid depth
        mask_pad = torch.sum(pw[:, :, 2, :] > self.delta_z, 2) == 3

        # ignore near
        mask_x = torch.sum(torch.abs(pw_diff[:, :, 0, :]) < delta_diff_x, 2) > 0
        mask_y = torch.sum(torch.abs(pw_diff[:, :, 1, :]) < delta_diff_y, 2) > 0
        mask_z = torch.sum(torch.abs(pw_diff[:, :, 2, :]) < delta_diff_z, 2) > 0

        mask_ignore = (mask_x & mask_y & mask_z) | mask_cos
        mask_near = ~mask_ignore
        mask = mask_pad & mask_near

        return mask, pw

    def select_points_groups(self, gt_depth, pred_depth):
        pw_gt = self.transfer_xyz(gt_depth)
        pw_pred = self.transfer_xyz(pred_depth)
        B, C, H, W = gt_depth.shape
        p123 = self.select_index()
        # mask:[b, n], pw_groups_gt: [b, n, 3(x,y,z), 3(p1,p2,p3)]
        mask, pw_groups_gt = self.filter_mask(p123, pw_gt,
                                              delta_cos=0.867,
                                              delta_diff_x=0.005,
                                              delta_diff_y=0.005,
                                              delta_diff_z=0.005)

        # [b, n, 3, 3]
        pw_groups_pred = self.form_pw_groups(p123, pw_pred)
        pw_groups_pred[pw_groups_pred[:, :, 2, :] == 0] = 0.0001
        mask_broadcast = mask.repeat(1, 9).reshape(B, 3, 3, -1).permute(0, 3, 1, 2)
        pw_groups_pred_not_ignore = pw_groups_pred[mask_broadcast].reshape(1, -1, 3, 3)
        pw_groups_gt_not_ignore = pw_groups_gt[mask_broadcast].reshape(1, -1, 3, 3)

        return pw_groups_gt_not_ignore, pw_groups_pred_not_ignore

    def forward(self, gt_depth, pred_depth, select=True):
        """
        Virtual normal loss.
        :param pred_depth: predicted depth map, [B,W,H,C]
        :param data: target label, ground truth depth, [B, W, H, C], padding region [padding_up, padding_down]
        :return:
        """
        device = gt_depth.device
        self.fx = self.fx.to(device)
        self.fy = self.fy.to(device)
        self.u0 = self.u0.to(device)
        self.v0 = self.v0.to(device)
        self.u_u0 = self.u_u0.to(device)
        self.v_v0 = self.v_v0.to(device)
        # print("************ ", self.fx.device, self.u_u0.device)

        gt_points, dt_points = self.select_points_groups(gt_depth, pred_depth)

        gt_p12 = gt_points[:, :, :, 1] - gt_points[:, :, :, 0]
        gt_p13 = gt_points[:, :, :, 2] - gt_points[:, :, :, 0]
        dt_p12 = dt_points[:, :, :, 1] - dt_points[:, :, :, 0]
        dt_p13 = dt_points[:, :, :, 2] - dt_points[:, :, :, 0]

        gt_normal = torch.cross(gt_p12, gt_p13, dim=2)
        dt_normal = torch.cross(dt_p12, dt_p13, dim=2)
        dt_norm = torch.norm(dt_normal, 2, dim=2, keepdim=True)
        gt_norm = torch.norm(gt_normal, 2, dim=2, keepdim=True)
        dt_mask = dt_norm == 0.0
        gt_mask = gt_norm == 0.0
        dt_mask = dt_mask.to(torch.float32)
        gt_mask = gt_mask.to(torch.float32)
        dt_mask *= 0.01
        gt_mask *= 0.01
        gt_norm = gt_norm + gt_mask
        dt_norm = dt_norm + dt_mask
        gt_normal = gt_normal / gt_norm
        dt_normal = dt_normal / dt_norm
        loss = torch.abs(gt_normal - dt_normal)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0)
        if select:
            loss, indices = torch.sort(loss, dim=0, descending=False)
            loss = loss[int(loss.size(0) * 0.25):]
        loss = torch.mean(loss)
        return loss


if __name__ == '__main__':
    vnl_loss = VNL_Loss(1.0, 1.0, (480, 640))
    midas_loss = MidasLoss()
    gt_depth = torch.randn([2, 1, 480, 640]).abs().cuda()
    pred_depth = torch.randn([2, 1, 480, 640]).abs().cuda()
    mask_valid = torch.ones([2, 1, 480, 640], dtype=torch.bool).cuda()
    vnl_loss = vnl_loss(pred_depth, gt_depth)
    ssi_loss, reg_loss = midas_loss(pred_depth, gt_depth, mask_valid)
    print(vnl_loss, ssi_loss, reg_loss)
