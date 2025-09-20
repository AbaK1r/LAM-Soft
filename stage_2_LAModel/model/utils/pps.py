import torch
import torch.nn.functional as F

from .utils import point_cloud_to_normals


def cal_pps(depth, light_data, ref_dirs):
    """

    :param depth: (B, H, W)
    :param light_data:
    :param ref_dirs:
    :return: (B, 1, H, W)
    """
    pc_preds = depth.unsqueeze(3) * ref_dirs  # (B, H, W, 3)
    l, a = calculate_per_pixel_lighting(pc_preds, *light_data)
    l = l.permute(0, 3, 1, 2)  # Rearrange l from [B, H, W, C] to [B, C, H, W]
    a = a.permute(0, 3, 1, 2)  # Rearrange a from [B, H, W, C] to [B, C, H, W]   min 0.6122, max 377.4979
    a = torch.log(a + 1e-8)
    a = (a + 0.49) / (5.94 + 0.49)
    l_norm = _normalize_vectors(l)
    normal_norm = point_cloud_to_normals(pc_preds.permute(0, 3, 1, 2))  # (B, 3, H, W)
    dot_product = torch.sum(l_norm * normal_norm, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
    dot_product_clamped = torch.clamp(dot_product, -1, 1)
    pps_image = dot_product_clamped * a  # (1, 1, ipt_size, ipt_size)
    return pps_image


def calculate_per_pixel_lighting(pc_preds, light_pos, light_dir, mu):
    # pc_gt and pc_preds (b,m,n,3)
    # light_pos (b,3)
    # light_dir (b,3)
    # angular attenuation mu (b,)
    # return (b,m,n,3) (b,m,n,1)

    # Calculate PPL for pc_preds
    to_light_vec_preds = light_pos.unsqueeze(1).unsqueeze(1) - pc_preds
    n_to_light_vec_preds = F.normalize(to_light_vec_preds, dim=3)
    # (b,m,n,1)
    len_to_light_vec_preds = torch.norm(to_light_vec_preds, dim=3, keepdim=True)
    light_dir_dot_to_light_preds = torch.sum(-n_to_light_vec_preds * light_dir.unsqueeze(1).unsqueeze(1), dim=3,
                                             keepdim=True).clamp(min=1e-8)
    numer_preds = torch.pow(light_dir_dot_to_light_preds, mu.view(-1, 1, 1, 1))
    atten_preds = numer_preds / (len_to_light_vec_preds ** 2).clamp(min=1e-8)

    return n_to_light_vec_preds, atten_preds


def _normalize_vectors(v):
    """ Normalize a batch of 3D vectors to unit vectors using PyTorch. """
    norms = v.norm(p=2, dim=1, keepdim=True)
    return v / (norms + 1e-8)  # Adding a small epsilon to avoid division by zero
