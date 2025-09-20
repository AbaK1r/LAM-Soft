# Our proposed PPSNet, which includes a depth estimation backbone and a depth refinement module
# Part of this code is inspired by the Depth Anything (CVPR 2024) paper's code implementation
# See https://github.com/LiheYoung/Depth-Anything/blob/main/depth_anything/dpt.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import model.pps_net.utils.optical_flow_funs as OF
from model.pps_net.blocks import FeatureFusionBlock, _make_scratch
from model.pps_net.utils.color_convert import hsv_to_rgb, rgb_to_hsv
from model.pps_net.losses import (
    calculate_per_pixel_lighting,
    MidasLoss, VNL_Loss,
    calculate_pps_supp_loss)
from model.pps_net.unet import UNet
from model.pps_net.utils.utils import get_normals_from_depth, normalize_images


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


def _normalize_vectors(v):
    """ Normalize a batch of 3D vectors to unit vectors using PyTorch. """
    norms = v.norm(p=2, dim=1, keepdim=True)
    return v / (norms + 1e-8)  # Adding a small epsilon to avoid division by zero


class CrossAttentionModule(nn.Module):
    def __init__(self, feature_dim=384, heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=heads,
                                               batch_first=True)  # Note the batch_first=True

    def forward(self, queries, keys_values):
        # Since we're ignoring class tokens, the input is directly [B, N, C]
        # Apply multi-head attention; assuming queries and keys_values are prepared [B, N, C]
        attn_output, _ = self.attention(queries, keys_values, keys_values)

        # attn_output is already in the correct shape [B, N, C], so we return it directly
        return attn_output


class FeatureEncoder(nn.Module):
    """Encodes combined features into a lower-dimensional representation."""

    def __init__(self, input_channels, encoded_dim):
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_channels // 2, encoded_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class FiLM(nn.Module):
    """Applies Feature-wise Linear Modulation to condition disparity refinement."""

    def __init__(self, encoded_dim, target_channels):
        super(FiLM, self).__init__()
        self.scale_shift_net = nn.Linear(encoded_dim, target_channels * 2)

    def forward(self, features, disparity):
        # Global average pooling and processing to get scale and shift parameters
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        scale_shift_params = self.scale_shift_net(pooled_features)
        scale, shift = scale_shift_params.chunk(2, dim=1)

        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        # Apply FiLM modulation to disparity
        modulated_disparity = disparity * scale + shift
        return modulated_disparity


class PPSNet_Refinement(nn.Module):
    """Refines disparity map conditioned on encoded features."""

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self, disparity_channels, encoded_dim, input_size):
        super(PPSNet_Refinement, self).__init__()
        self.input_size = input_size
        self.encoder = FeatureEncoder(input_channels=384, encoded_dim=encoded_dim)
        self.film = FiLM(encoded_dim=encoded_dim, target_channels=disparity_channels)
        self.cross_attention = CrossAttentionModule()
        self.refinement_net = UNet(disparity_channels, disparity_channels)

        self.apply(self.init_weights)

    def forward(self, features_rgb, features_colored_dot_product, initial_disparity):

        # Feed in RGB features from before and colored dot product
        combined_features = []
        for features1, features2 in zip(features_rgb, features_colored_dot_product):
            # Extract feature maps
            feature_map1 = features1[0]  # Assuming the first element is the feature map, shape [B, N, C]
            feature_map2 = features2[0]  # Assuming the first element is the feature map, shape [B, N, C]
            dummy_cls_token = torch.zeros((feature_map1.shape[0], 1, feature_map1.shape[-1]),
                                          device=feature_map1.device)  # [B, 1, C]

            # Apply cross-attention directly
            attn_output = self.cross_attention(feature_map1, feature_map2)  # Assume this outputs shape [B, N, C]

            # Append both the attention output and a dummy class token to combined_features
            # This makes combined_features a list of tuples [(feature_map, class_token), ...]
            combined_features.append((attn_output, dummy_cls_token))

        # Reshape combined_features for processing
        combined_features_reshaped = combined_features[3][0].reshape(-1, 384, self.input_size//14, self.input_size//14)
        encoded_features = self.encoder(combined_features_reshaped)

        # Condition disparity refinement on encoded features
        modulated_disparity = self.film(encoded_features, initial_disparity)

        # Refine modulated disparity
        modulated_disparity = F.interpolate(
            modulated_disparity, scale_factor=0.5, mode='bilinear', align_corners=False)
        refined_disparity = self.refinement_net(modulated_disparity)
        refined_disparity = F.interpolate(
            refined_disparity, size=(self.input_size,)*2, mode='bilinear', align_corners=False)
        return (refined_disparity + initial_disparity).squeeze(1)


class DPTHead(nn.Module):
    def __init__(
        self,
        nclass,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=(256, 512, 1024, 1024),
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()

        self.nclass = nclass
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
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1,
                                                  padding=1)

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
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

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

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


class DPT_DINOv2(nn.Module):
    def __init__(
        self,
        encoder='vits',
        features=256,
        out_channels=(256, 512, 1024, 1024),
        use_bn=False,
        use_clstoken=False
    ):
        super(DPT_DINOv2, self).__init__()

        assert encoder in ['vits', 'vitb', 'vitl']
        self.pretrained = torch.hub.load(
            'torchhub/facebookresearch_dinov2_main',
            'dinov2_{:}14'.format(encoder),
            source='local', pretrained=False)

        dim = self.pretrained.blocks[0].attn.qkv.in_features

        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x, ref_dirs, light_pos, light_dir, mu, n_intrinsics):
        h, w = x.shape[-2:]

        # Step 1: Initial depth prediction and normals from depth
        features_rgb = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)

        patch_h, patch_w = h // 14, w // 14

        disparity = self.depth_head(features_rgb, patch_h, patch_w)
        disparity = F.interpolate(disparity, size=(h, w), mode="bilinear", align_corners=True)
        disparity = F.relu(disparity)

        # Get depth from disparity
        depth = 1 / disparity
        depth = torch.clamp(depth, 0, 1)

        normal, _ = get_normals_from_depth(depth, n_intrinsics)

        # Step 2: Get PPL info from initial depth
        pc_preds = depth.squeeze(1).unsqueeze(3) * ref_dirs
        l, a = calculate_per_pixel_lighting(pc_preds, light_pos, light_dir, mu)

        # Ensure L and A are in the same format as RGB (B, C, H, W)
        l = l.permute(0, 3, 1, 2)  # Rearrange l from [B, H, W, C] to [B, C, H, W]
        a = a.permute(0, 3, 1, 2)  # Rearrange a from [B, H, W, C] to [B, C, H, W]

        # Log transformation on A
        a = torch.log(a + 1e-8)

        # Min-Max normalization for A should ideally be based on dataset statistics,
        # Here we normalize based on the min and max of the current batch for simplicity
        # a_min, a_max = a.min(), a.max()
        # a = (a - a_min) / (a_max - a_min + 1e-8)
        a = normalize_images(a)

        # Normalize l and normal
        l_norm = _normalize_vectors(l)
        normal_norm = _normalize_vectors(normal)

        # Compute dot product and apply attenuation
        dot_product = torch.sum(l_norm * normal_norm, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        dot_product_clamped = torch.clamp(dot_product, -1, 1)
        dot_product_attenuated = dot_product_clamped * a

        hsv_albedo = rgb_to_hsv(x)
        hsv_albedo[:, 2] = 1.0  # Set V to 100% brightness
        albedo_tensor = hsv_to_rgb(hsv_albedo)

        dot_product_attenuated = dot_product_attenuated.repeat(1, 3, 1, 1)  # New shape will be [B, 3, H, W]

        # colored_dot_product_attenuated = albedo_tensor.permute(2, 0, 1).unsqueeze(0) * dot_product_attenuated
        colored_dot_product_attenuated = albedo_tensor * dot_product_attenuated
        colored_dot_product_attenuated = normalize_images(colored_dot_product_attenuated)
        # Feature extraction for dot_product_attenuated
        features_colored_dot_product = self.pretrained.get_intermediate_layers(
            colored_dot_product_attenuated, 4, return_class_token=True)

        return disparity, features_rgb, features_colored_dot_product


class PPSNet_Backbone(DPT_DINOv2):
    def __init__(self, config):
        super().__init__(**config)


class PpsNet(nn.Module):
    def __init__(self, encoder_name='vits', input_size=512):
        super().__init__()
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        }
        self.encoder_name = encoder_name
        self.input_size = (input_size,)*2
        self.backbone = PPSNet_Backbone(model_configs[encoder_name])
        self.refinement_model = PPSNet_Refinement(1, 384, input_size)
        self.vnl_loss = VNL_Loss(1.0, 1.0, self.input_size)
        self.midas_loss = MidasLoss()
        self.load_pretrained()

    def load_pretrained(self):
        # self.backbone.load_state_dict(torch.load(f'./pretrained/depth_anything_{self.encoder_name}14.pth'))

        checkpoint = torch.load('/data/2t/jupyter/wxh/datasets/student.ckpt')
        backbone_state_dict = {}
        for k, v in checkpoint['student_state_dict'].items():
            name = k[7:] if k.startswith('module.') else k
            backbone_state_dict[name] = v
        self.backbone.load_state_dict(backbone_state_dict)

        refinement_state_dict = {}
        for k, v in checkpoint['refiner_state_dict'].items():
            name = k[7:] if k.startswith('module.') else k
            refinement_state_dict[name] = v
        self.refinement_model.load_state_dict(refinement_state_dict)

    def pred(self, img_tensor, n_intrinsics):
        pos = torch.zeros(1, 3)  # light and camera co-located (b,3)
        direction = torch.tensor([[0, 0, 1]])  # light direction straight towards +z (b,3)
        mu = torch.tensor([0])  # approximate attenuation in air as 0 (b,)
        light_data = (pos.cuda(), direction.cuda(), mu.cuda())

        ref_dirs = OF.get_camera_pixel_directions(img_tensor.shape[2:4], n_intrinsics,
                                                  normalized_intrinsics=True)
        disparity, rgb_feats, colored_dot_product_feats = self.backbone(
            img_tensor, ref_dirs, *light_data, n_intrinsics)
        disp_preds = self.refinement_model(rgb_feats, colored_dot_product_feats, disparity)

        pred = 1 / disp_preds
        return pred

    def sforward(self, img_tensor, light_data, n_intrinsics):
        ref_dirs = OF.get_camera_pixel_directions(img_tensor.shape[2:4], n_intrinsics,
                                                  normalized_intrinsics=True)
        disparity, rgb_feats, colored_dot_product_feats = self.backbone(
            img_tensor, ref_dirs, *light_data, n_intrinsics)
        disp_preds = self.refinement_model(rgb_feats, colored_dot_product_feats, disparity)

        pred = 1 / disp_preds
        pred = torch.clamp(pred, 0., 1.)
        return pred, ref_dirs

    def forward(self, batch):
        img_tensor = batch['rgb']
        light_data = batch['light_data']
        n_intrinsics = batch['n_intrinsics']
        ref_dirs = OF.get_camera_pixel_directions(img_tensor.shape[2:4], n_intrinsics,
                                                  normalized_intrinsics=True)
        disparity, rgb_feats, colored_dot_product_feats = self.backbone(
            img_tensor, ref_dirs, *light_data, n_intrinsics)
        disp_preds = self.refinement_model(rgb_feats, colored_dot_product_feats, disparity)

        pred = 1 / disp_preds
        pred = torch.clamp(pred, 0., 1.)
        return pred

    def train_step(self, batch):
        img_tensor = batch['rgb']
        gt = batch['depth']
        light_data = batch['light_data']
        n_intrinsics = batch['n_intrinsics']

        pred, ref_dirs = self.sforward(img_tensor, light_data, n_intrinsics)
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
        # pps_sup_loss, mask_valid = calculate_pps_corr_loss(img_tensor, l, a, normal)
        pred = pred.unsqueeze(1)
        vnl_loss = self.vnl_loss(pred, gt)
        mask_valid = torch.ones_like(gt).bool()
        ssi_loss, reg_loss = self.midas_loss(pred, gt, mask_valid)

        l1_loss = torch.abs(gt - pred).mean()
        rmse = torch.sqrt(torch.mean((gt - pred) ** 2))
        epsilon = 1e-6
        abs_rel = torch.abs(gt - pred) / (gt + epsilon)
        sq_rel = torch.mean(((gt - pred) / (gt + epsilon)) ** 2)
        si = torch.max(gt / (pred + epsilon), pred / (gt + epsilon))
        s1 = (si < 1.05).float().mean()
        s2 = (si < (1.05 ** 2)).float().mean()
        s3 = (si < (1.05 ** 3)).float().mean()
        a1 = (abs_rel < 0.1).float().mean()

        loss = pps_sup_loss + 10 * vnl_loss + ssi_loss + 0.1 * reg_loss + 0.1 * l1_loss
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

    # def train_step(self, batch):
    #     img_tensor = batch['rgb']
    #     gt = batch['depth']
    #     light_data = batch['light_data']
    #     n_intrinsics = batch['n_intrinsics']
    #
    #     pred, ref_dirs = self.forward(img_tensor, light_data, n_intrinsics)
    #
    #     normal_pred, _ = get_normals_from_depth(pred.unsqueeze(1), n_intrinsics)
    #     normal_gt, _ = get_normals_from_depth(gt, n_intrinsics)
    #
    #     pc_preds = pred.unsqueeze(3) * ref_dirs
    #     l_pred, a_pred = calculate_per_pixel_lighting(pc_preds, *light_data)
    #
    #     pc_gt = gt.squeeze(1).unsqueeze(3) * ref_dirs
    #     l_gt, a_gt = calculate_per_pixel_lighting(pc_gt, *light_data)
    #
    #     pps_sup_loss = calculate_pps_supp_loss(l_pred, a_pred, normal_pred, l_gt, a_gt, normal_gt)
    #     # pps_sup_loss, mask_valid = calculate_pps_corr_loss(img_tensor, l, a, normal)
    #
    #     vnl_loss = self.vnl_loss(pred.unsqueeze(1), gt)
    #     mask_valid = torch.ones_like(gt).bool()
    #     ssi_loss, reg_loss = self.midas_loss(pred.unsqueeze(1), gt, mask_valid)
    #
    #     l1_loss = torch.abs(gt - pred[:, None]).mean()
    #
    #     loss = pps_sup_loss + 10 * vnl_loss + ssi_loss + 0.1 * reg_loss + 0.5 * l1_loss
    #     loss_dict = {
    #         'loss': loss,
    #         'pps_sup_loss': pps_sup_loss,
    #         'vnl_loss': vnl_loss,
    #         'ssi_loss': ssi_loss,
    #         'reg_loss': reg_loss,
    #         'l1_loss': l1_loss
    #     }
    #     return loss_dict
