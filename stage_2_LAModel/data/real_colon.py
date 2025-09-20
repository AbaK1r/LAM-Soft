import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from model.utils.color_convert import hsv_to_rgb, rgb_to_hsv
import model.utils.optical_flow_funs as OF


class RealColon(Dataset):
    def __init__(self, data_mode='train', input_size=256, **kwargs):
        super().__init__()
        input_size = 336
        self.pic_root = Path('/home/star/datasets/wxh/real_colon_clip_pics')
        self.depth_root = Path('/home/star/datasets/wxh/real_colon_clip_pics_est')
        self.gen_pps_root = Path('/home/star/datasets/wxh/real_colon_gen_pps')

        intrinsics = np.array([[822.78458832, 0., 583.3560641],
                               [0., 822.90873376, 582.9039563],
                               [0., 0., 1.]]).astype(np.float32)
        intrinsics[0, :] = intrinsics[0, :] * (input_size / 1158)
        intrinsics[1, :] = intrinsics[1, :] * (input_size / 1158)
        self.intrinsics = torch.tensor(intrinsics)
        # self.intrinsics_inv = self.intrinsics.inverse()
        self.n_intrinsics = OF.pixel_intrinsics_to_normalized_intrinsics(
            torch.from_numpy(intrinsics).unsqueeze(0).float(), (input_size,) * 2).squeeze(0)
        self.data_mode = data_mode
        self.input_size = input_size
        # self.resize = transforms.Resize((input_size,) * 2, antialias=True)
        if data_mode == 'Train':
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2),
                                       hue=(0.8, 1.2)),
                transforms.RandomAutocontrast(p=0.5),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.ToTensor()

        self.image_path_ls = sorted(self.pic_root.glob('*.jpg'), key=lambda x: int(x.name.split('.')[0]))
        with open('/data/2t/jupyter/wxh/PPSNetSupMask/pred_mask_mean_val.txt', 'r') as f:
            mask_data = {}
            for i in f.readlines():
                _name, _score = i.strip().split(' ')
                mask_data[_name] = float(_score)
        self.image_path_ls = [i for i in self.image_path_ls if i.stem in mask_data.keys() and mask_data[i.stem] < 0.54]
        logging.info(f"size of real colon dataset: {len(self.image_path_ls)}")

        self.depth_path_ls = [self.depth_root / (x.stem + '.tiff') for x in self.image_path_ls]
        for i in self.depth_path_ls:
            assert i.exists()
        self.gen_pps_path_ls = [self.gen_pps_root / (x.stem + '.tiff') for x in self.image_path_ls]
        for i in self.gen_pps_path_ls:
            assert i.exists()
        self.ref_dirs = OF.get_camera_pixel_directions((input_size,)*2, self.n_intrinsics[None],
                                                       normalized_intrinsics=True)  # (1, ipt_size, ipt_size, 3)

    def __len__(self):
        return len(self.image_path_ls)
        # return 200

    def __getitem__(self, idx):
        # if idx > self.__len__():
        #     raise StopIteration
        pic1_raw = Image.open(str(self.image_path_ls[idx]))
        depth_raw = Image.open(str(self.depth_path_ls[idx]))
        # gen_pps_raw = Image.open(str(self.gen_pps_path_ls[idx]))
        rgb = self.transform(pic1_raw)
        depth = torch.tensor(np.asarray(depth_raw, dtype=np.float32) / 65535.)[None]

        gt_pps = Image.open(str(self.gen_pps_path_ls[idx]))
        gt_pps = torch.tensor(np.asarray(gt_pps, dtype=np.float32)) / 65535.
        # -----------------------#
        # pos = torch.zeros(1, 3)  # light and camera co-located (b,3)
        # direction = torch.tensor([[0, 0, 1]])  # light direction straight towards +z (b,3)
        # mu = torch.tensor([0])  # approximate attenuation in air as 0 (b,)
        # pc_preds = depth.unsqueeze(3) * self.ref_dirs
        # l, a = calculate_per_pixel_lighting(pc_preds, pos, direction, mu)
        # l = l.permute(0, 3, 1, 2)  # Rearrange l from [B, H, W, C] to [B, C, H, W]
        # a = a.permute(0, 3, 1, 2)  # Rearrange a from [B, H, W, C] to [B, C, H, W]   min 0.6122, max 377.4979
        # a = torch.log(a + 1e-8)
        # a = (a + 0.49) / (5.94 + 0.49)
        # l_norm = _normalize_vectors(l)
        # normal, _ = get_normals_from_depth(depth[None], self.n_intrinsics[None])
        # normal_norm = _normalize_vectors(normal)
        # dot_product = torch.sum(l_norm * normal_norm, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        # dot_product_clamped = torch.clamp(dot_product, -1, 1)
        # gen_pps = dot_product_clamped * a  # (1, 1, ipt_size, ipt_size)
        # -----------------------#
        hsv_albedo = rgb_to_hsv(rgb[None])
        hsv_albedo[:, 2] = 1.0
        albedo_tensor = hsv_to_rgb(hsv_albedo)[0]
        real_pps = (rgb / (albedo_tensor + 1e-8)).mean(0)
        assert ((real_pps >= 0.) & (
                    real_pps <= 1.001)).all(), f'something must be wrong...{real_pps.max()};{real_pps.min()}'
        # -----------------------#
        # gen_pps = (pps_image - pps_image.mean()) / pps_image.std() * real_pps.std() + real_pps.mean()
        # -----------------------#
        # gen_pps = pps_image
        # -----------------------#
        # gen_pps = (pps_image - pps_image.min()) / (pps_image.max() - pps_image.min())
        # gen_pps = torch.clamp(gen_pps, 0, 1)

        # Constants for Per-Pixel Lighting (PPL)
        pos = torch.zeros(3)  # light and camera co-located (b,3)
        direction = torch.tensor([0, 0, 1])  # light direction straight towards +z (b,3)
        mu = 0  # approximate attenuation in air as 0 (b,)
        light_data = (pos, direction, mu)

        data_dic = {
            'rgb': rgb,  # [3, 84, 84]
            # 'pps': pps_image,  # [84, 84]
            'real_pps': real_pps[None],  # [1, 84, 84]
            # 'gen_pps': gen_pps[0],  # [1, 84, 84]
            # 'a': a,
            'gt_pps': gt_pps[None],  # [1, 84, 84]
            'depth': depth,  # [1, 84, 84]
            'light_data': light_data,
            'n_intrinsics': self.n_intrinsics,
            'filedir': str(self.image_path_ls[idx])
        }
        return data_dic


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



pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    B, H, W = depth.size()
    i_range = torch.arange(0, H).view(1, H, 1).expand(1, H, W).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, W).view(1, 1, W).expand(1, H, W).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, H, W).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    B, H, W = depth.size()
    if pixel_coords is None:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[..., :H, :W].expand(B, 3, H, W).reshape(B, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(B, 3, H, W)
    return cam_coords * depth.unsqueeze(1)


def cloud2ppl(pc, light_pos, light_dir, mu):
    """

    :param pc: (B,3,m,n)
    :param light_pos: (B,3)
    :param light_dir: (B,3)
    :param mu: (B,) angular attenuation
    :return l, a: (B,3,m,n) (B,1,m,n)
    """
    light_pos = light_pos[..., None, None]
    light_dir = light_dir[..., None, None]
    mu = mu.view(-1, 1, 1, 1)
    to_light_vec_preds = light_pos - pc
    l = F.normalize(to_light_vec_preds, dim=1)
    len_to_light_vec_preds = torch.norm(to_light_vec_preds, dim=1, keepdim=True)
    light_dir_dot_to_light_preds = torch.sum(-l * light_dir, dim=1, keepdim=True).clamp(min=1e-8)
    numer_preds = torch.pow(light_dir_dot_to_light_preds, mu)
    a = numer_preds / len_to_light_vec_preds.pow(2).clamp(min=1e-8)
    a = torch.log(a + 1e-8)
    return l, a


def pps(l, a, depth, intrinsics):
    """

    :param l:
    :param a:
    :param depth: (B,1,H,W)
    :param intrinsics: (B,3,3)
    """
    normal, _ = get_normals_from_depth(depth, intrinsics)
    n = _normalize_vectors(normal)
    l = _normalize_vectors(l)
    dot_product = torch.sum(l * n, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
    dot_product_clamped = torch.clamp(dot_product, -1, 1)
    dot_product_attenuated = dot_product_clamped * a
    return dot_product_attenuated


def _normalize_vectors(v):
    """ Normalize a batch of 3D vectors to unit vectors using PyTorch. """
    norms = v.norm(p=2, dim=1, keepdim=True)
    return v / (norms + 1e-8)  # Adding a small epsilon to avoid division by zero


def get_normals_from_depth(
        depth, intrinsics, depth_is_along_ray=False, diff_type='center', normalized_intrinsics=True):
    """

    :param depth: (B,1,m,n)
    :param intrinsics: (B,3,3)
    :param depth_is_along_ray:
    :param diff_type:
    :param normalized_intrinsics:
    :return: (B,3,m,n), (B,3,m,n)
    """
    dirs = OF.get_camera_pixel_directions(depth.shape[2:4], intrinsics, normalized_intrinsics=normalized_intrinsics)
    dirs = dirs.permute(0, 3, 1, 2)
    if depth_is_along_ray:
        dirs = torch.nn.functional.normalize(dirs, dim=1)
    pc = dirs * depth

    normal = point_cloud_to_normals(pc, diff_type=diff_type)
    return normal, pc


def point_cloud_to_normals(pc, diff_type='center'):
    # pc (B,3,m,n)
    # return (B,3,m,n)
    dp_du, dp_dv = image_derivatives(pc, diff_type=diff_type)
    normal = torch.nn.functional.normalize(torch.cross(dp_du, dp_dv, dim=1))
    return normal


def image_derivatives(image, diff_type='center'):
    c = image.size(1)
    if diff_type == 'center':
        sobel_x = 0.5 * torch.tensor(
            [[0.0, 0, 0], [-1, 0, 1], [0, 0, 0]],
            device=image.device
        ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        sobel_y = 0.5 * torch.tensor(
            [[0.0, 1, 0], [0, 0, 0], [0, -1, 0]],
            device=image.device
        ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    elif diff_type == 'forward':
        sobel_x = torch.tensor(
            [[0.0, 0, 0], [0, -1, 1], [0, 0, 0]],
            device=image.device
        ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        sobel_y = torch.tensor(
            [[0.0, 1, 0], [0, -1, 0], [0, 0, 0]],
            device=image.device
        ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    else:
        raise NotImplementedError

    dp_du = torch.nn.functional.conv2d(image, sobel_x, padding=1, groups=3)
    dp_dv = torch.nn.functional.conv2d(image, sobel_y, padding=1, groups=3)
    return dp_du, dp_dv
