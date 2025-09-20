from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from model.utils.color_convert import hsv_to_rgb, rgb_to_hsv
import model.utils.optical_flow_funs as OF


def load_pose(data_seq_path: Path):
    pose_path = Path('/home/star/datasets/wxh/C3VD') / data_seq_path.name / 'pose.txt'
    return np.loadtxt(pose_path, delimiter=',').astype(np.float32).reshape(-1, 4, 4).transpose(0, 2, 1)


class C3vd(Dataset):
    def __init__(self, data_mode='train', input_size=256, **kwargs):
        super().__init__()
        input_size = 336
        intrinsics = np.array([
            [767.3638647587969, 0, 544.0626504570412],
            [0, 767.4796480626248, 543.6409955105457],
            [0, 0, 1]
        ]).astype(np.float32)
        intrinsics[0, :] = intrinsics[0, :] * (input_size / 1080)
        intrinsics[1, :] = intrinsics[1, :] * (input_size / 1080)
        self.intrinsics = torch.tensor(intrinsics.astype(np.float32))
        self.n_intrinsics = OF.pixel_intrinsics_to_normalized_intrinsics(
            torch.from_numpy(intrinsics).unsqueeze(0).float(), (input_size,)*2).squeeze(0)
        self.data_mode = data_mode
        self.input_size = input_size
        self.resize = transforms.Resize((input_size,) * 2, antialias=True)
        self.to_tensor = transforms.ToTensor()
        if self.data_mode == 'train':
            # self.data_enhance = transforms.Compose([
            #     # transforms.RandomEqualize(p=0.9),
            #     # transforms.RandomAutocontrast(p=0.9),
            #     # 随机地改变图像的亮度和色调。亮度因子从[0.5, 1.5]之间均匀地选择，色调因子从[-0.3, 0.3]之间均匀地选择。
            #     # transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=0.5, hue=0.3)], p=0.6),
            #     transforms.RandomApply(transforms=[transforms.GaussianBlur(3, 0.1)], p=0.6),
            # ])
            self.data_enhance = None
        else:
            self.data_enhance = None

        with open(f'task/c3vd/{self.data_mode}.txt', 'r') as f:
            self.img_root_dirs = f.readline().strip().split(' ')

        self.image_path_ls = []
        self.depth_path_ls = []
        self.idx_ls = []
        self.setup_idx()

    #  *0.001525
    def setup_idx(self):
        idx_ls = []
        for idx, subdir in enumerate(self.img_root_dirs):
            subdir = Path('/home/star/datasets/wxh/processed') / subdir
            image_path = sorted(subdir.glob('*.png'), key=lambda x: int(x.name.split('_')[0]))
            depth_path = sorted(subdir.glob('*.tiff'), key=lambda x: int(x.name.split('_')[0]))
            self.image_path_ls.append(image_path)
            self.depth_path_ls.append(depth_path)
            idx_ls += [[idx, i] for i in range(len(image_path))]
        self.idx_ls = idx_ls

    def __len__(self):
        return len(self.idx_ls)

    def __getitem__(self, idx):
        seq_id, idx = self.idx_ls[idx]
        pic1_raw = Image.open(str(self.image_path_ls[seq_id][idx]))
        depth1_raw = Image.open(str(self.depth_path_ls[seq_id][idx]))
        if self.data_enhance:
            pic1_raw = self.data_enhance(pic1_raw)
        if self.input_size != pic1_raw.size[-1]:
            pic1 = self.to_tensor(self.resize(pic1_raw))
            depth1 = torch.tensor(
                np.array(depth1_raw.resize((self.input_size,)*2, 0), dtype=np.float32) * 0.0001525)[None]
        else:
            pic1 = self.to_tensor(pic1_raw)
            depth1 = torch.tensor(np.array(depth1_raw, dtype=np.float32) * 0.0001525)[None]

        pos = torch.zeros(3)  # light and camera co-located (b,3)
        direction = torch.tensor([0, 0, 1])  # light direction straight towards +z (b,3)
        mu = torch.tensor([0])  # approximate attenuation in air as 0 (b,)
        light_data = (pos, direction, mu)

        cam_coords = pixel2cam(depth1, self.intrinsics.inverse()[None])
        l, a = cloud2ppl(cam_coords, *light_data)
        # a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        pps_image = pps(l, a, depth1[:, None], self.n_intrinsics[None])[0, 0]  # (H, W)

        hsv_albedo = rgb_to_hsv(pic1[None])
        hsv_albedo[:, 2] = 1.0
        albedo_tensor = hsv_to_rgb(hsv_albedo)[0]
        real_pps = (pic1 / albedo_tensor).mean(0)
        assert ((real_pps >= 0.) & (real_pps <= 1.001)).all(), f'something must be wrong...{real_pps.max()};{real_pps.min()}'
        # gen_pps = (pps_image - pps_image.mean()) / pps_image.std() * real_pps.std() + real_pps.mean()
        gen_pps = pps_image
        # gen_pps = (pps_image - pps_image.min()) / (pps_image.max() - pps_image.min())
        # gen_pps = torch.clamp(gen_pps, 0, 1)
        data_dic = {
            'rgb': pic1,
            'pps': pps_image,
            'real_pps': real_pps[None],
            'gt_pps': gen_pps[None],
            'depth': depth1 / 10.
        }
        return data_dic


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
