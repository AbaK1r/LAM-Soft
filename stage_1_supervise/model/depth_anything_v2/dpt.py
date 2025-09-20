import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from torchmetrics import Accuracy, JaccardIndex
from torchmetrics.segmentation import DiceScore

from model.pps_net.losses import (
    calculate_per_pixel_lighting,
    MidasLoss, VNL_Loss,
    calculate_pps_supp_loss)
from model.pps_net.utils.utils import get_normals_from_depth
from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .dinov2_layers import PatchEmbed


def _make_fusion_block(features, use_bn, size=None, if_resConfUnit1=True):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
        if_resConfUnit1=if_resConfUnit1
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
            self,
            in_channels,
            features=256,
            use_bn=False,
            out_channels=(256, 512, 1024, 1024),
            use_clstoken=False
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn, if_resConfUnit1=False)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()  # 12
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        return out


class DinoVisionClassifier(nn.Module):
    def __init__(
            self,
            num_classes=1000,
            in_chans=3,
            encoder='vits',
            input_size=336,
            pretrain=True
    ):
        super().__init__()
        self.in_chans = in_chans
        self.dinov2 = DINOv2(model_name=encoder)
        self.head = nn.Linear(self.dinov2.embed_dim, num_classes)  # 分类头

        if pretrain:
            logging.info("load pretrain.")
            self.dinov2.load_state_dict(
                torch.load(f'/data/2t/jupyter/wxh/ckpts/dinov2_{encoder}.pth', map_location='cpu'),
                strict=False)
        if in_chans != 3:
            self.dinov2.patch_embed = PatchEmbed(
                img_size=input_size,
                patch_size=self.dinov2.patch_embed.patch_size[0],
                in_chans=in_chans,
                embed_dim=self.dinov2.patch_embed.embed_dim)

    def freeze_blocks(self):
        if self.in_chans != 3:
            for name, param in self.dinov2.named_parameters():
                if 'patch_embed' not in name:
                    param.requires_grad = False
        else:
            for name, param in self.dinov2.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.dinov2.forward_features(x)
        logits = self.head(features["x_norm_clstoken"])
        return logits


class DepthAnythingV2(nn.Module):
    def __init__(
            self,
            encoder='vitl',
            input_size=518,
            use_bn=False,
            use_clstoken=False,
            pretrain=True
    ):
        super(DepthAnythingV2, self).__init__()
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }[encoder]

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.depth_head = DPTHead(
            self.pretrained.embed_dim, model_configs['features'],
            use_bn, out_channels=model_configs['out_channels'], use_clstoken=use_clstoken)
        self.mask_head = DPTHead(
            self.pretrained.embed_dim, model_configs['features'],
            use_bn, out_channels=model_configs['out_channels'], use_clstoken=use_clstoken)
        if pretrain:
            logging.info("load pretrain.")
            sd = torch.load(f'/data/16t/wxh/ckpts/depth_anything_v2_{encoder}.pth', map_location='cpu')
            sd_mask_head = {k.replace('depth_head', 'mask_head'): v for k, v in sd.items() if 'depth_head' in k}
            sd.update(sd_mask_head)
            self.load_state_dict(sd, strict=False)
        self.input_size = (input_size,) * 2
        self.vnl_loss = VNL_Loss(1.0, 1.0, self.input_size)
        self.midas_loss = MidasLoss()
        self.accuracy = Accuracy(task='binary')
        self.dice = DiceScore(1, include_background=False)
        self.iou = JaccardIndex(task="multiclass", num_classes=2)

    def pred(self, x, **kwargs):
        return self.sforward(x)

    def sforward(self, x):
        """

        :param x:
        :return: (bs, 1, H, W), (bs, 1, H, W)
        """
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder],
            return_class_token=True)
        depth = F.relu(self.depth_head(features, patch_h, patch_w)) / 150
        mask = F.sigmoid(self.mask_head([[j.detach() for j in i] for i in features], patch_h, patch_w))

        return depth, mask

    def forward(self, x):
        """

        :param x:
        :return: (bs, 1, H, W), (bs, 1, H, W)
        """
        x = x['rgb']
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder],
            return_class_token=True)

        depth = F.relu(self.depth_head(features, patch_h, patch_w)) / 150
        mask = F.sigmoid(self.mask_head([[j.detach() for j in i] for i in features], patch_h, patch_w))
        return depth, mask

    @torch.no_grad()
    def pred_picture(self, pic_path, cuda=True, return_pic=False):
        image = Image.open(pic_path).resize(self.input_size)
        x = torch.tensor(np.asarray(image)).permute(2, 0, 1)[None].float() / 255.
        if cuda:
            x = x.cuda()
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder],
            return_class_token=True)

        depth = F.relu(self.depth_head(features, patch_h, patch_w)) / 150
        mask = F.sigmoid(self.mask_head([[j.detach() for j in i] for i in features], patch_h, patch_w))

        depth = depth.detach().cpu().numpy()[0, 0]
        mask = mask.detach().cpu().numpy()[0, 0]
        # return depth if not return_pic else (depth, image)
        return (depth, mask) if not return_pic else (depth, mask, image)

    def train_step(self, batch, output_pred=False):
        img_tensor = batch['rgb']
        b = img_tensor.shape[0]
        gt = batch['depth']
        n_intrinsics = batch['n_intrinsics']

        pos = torch.zeros((b, 3)).to(img_tensor)  # light and camera co-located (b,3)
        direction = torch.tensor([[0, 0, 1]]).repeat(b, 1).to(img_tensor)  # light direction straight towards +z (b,3)
        mu = torch.zeros((b,)).to(img_tensor)  # approximate attenuation in air as 0 (b,)
        light_data = (pos, direction, mu)

        pred, mask = self.sforward(img_tensor)

        normal_pred, pc_preds = get_normals_from_depth(pred, n_intrinsics)
        normal_gt, pc_gt = get_normals_from_depth(gt, n_intrinsics)
        l_pred, a_pred = calculate_per_pixel_lighting(rearrange(pc_preds, 'b c x y -> b x y c'), *light_data)
        l_gt, a_gt = calculate_per_pixel_lighting(rearrange(pc_gt, 'b c x y -> b x y c'), *light_data)

        pps_sup_loss = calculate_pps_supp_loss(l_pred, a_pred, normal_pred, l_gt, a_gt, normal_gt)
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

        mask_gt = (si > si.mean()).to(int)
        mask_acc = self.accuracy(mask, mask_gt)
        mask_dice_score = self.dice(mask[:, 0], mask_gt[:, 0])
        mask_iou_score = self.iou(mask[:, 0], mask_gt[:, 0])
        mask_ce_loss = F.binary_cross_entropy(mask, mask_gt.float())

        loss = pps_sup_loss + 10 * vnl_loss + ssi_loss + 0.1 * l1_loss + 0.1 * reg_loss + 1 - mask_dice_score + mask_ce_loss
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
            'rsme': rmse,

            'mask_acc': mask_acc,
            'mask_dice_score': mask_dice_score,
            'mask_iou_score': mask_iou_score,
            'mask_ce_loss': mask_ce_loss,
        }
        if output_pred:
            # (b, 1, h, w)
            loss_dict['pred'] = {
                'gt': gt,
                'pred': pred,
                'mask': mask,
                'mask_gt': mask_gt,
            }

        return loss_dict
