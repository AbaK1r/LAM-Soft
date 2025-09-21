from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import model.pps_net.utils.optical_flow_funs as OF


def load_pose(data_seq_path: Path):
    pose_path = Path('/home/star/datasets/wxh/C3VD') / data_seq_path.name / 'pose.txt'
    return np.loadtxt(pose_path, delimiter=',').astype(np.float32).reshape(-1, 4, 4).transpose(0, 2, 1)


class C3vd(Dataset):
    def __init__(self, data_mode='train', input_size=336, **kwargs):
        super().__init__()
        # intrinsics = np.array([
        #     [767.3638647587969, 0, 544.0626504570412],
        #     [0, 767.4796480626248, 543.6409955105457],
        #     [0, 0, 1]
        # ]).astype(np.float32)
        # intrinsics[0, :] = intrinsics[0, :] * (input_size / 1080)
        # intrinsics[1, :] = intrinsics[1, :] * (input_size / 1080)
        intrinsics = np.array([
            [0.51, 0., 0.5],
            [0., 0.51, 0.5],
            [0., 0., 1.]
        ], dtype=np.float32)
        self.intrinsics = intrinsics.astype(np.float32)
        self.n_intrinsics = OF.pixel_intrinsics_to_normalized_intrinsics(
            torch.from_numpy(intrinsics).unsqueeze(0).float(), (input_size,)*2).squeeze(0)
        # self.data_mode = 'train' if data_mode == 'train' else 'val'
        self.data_mode = data_mode
        self.input_size = input_size
        self.resize = transforms.Resize((input_size,) * 2, antialias=True)
        self.to_tensor = transforms.ToTensor()
        if self.data_mode == 'train':
            # self.data_enhance = transforms.Compose([
            #     transforms.RandomEqualize(p=0.9),
            #     transforms.RandomAutocontrast(p=0.9),
            #     # 随机地改变图像的亮度和色调。亮度因子从[0.5, 1.5]之间均匀地选择，色调因子从[-0.3, 0.3]之间均匀地选择。
            #     transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=0.5, hue=0.3)], p=0.6),
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
        # return 200

    def __getitem__(self, idx):
        seq_id, idx = self.idx_ls[idx]
        pic1_raw = Image.open(str(self.image_path_ls[seq_id][idx]))
        depth1_raw = Image.open(str(self.depth_path_ls[seq_id][idx]))
        if self.data_enhance:
            pic1_raw = self.data_enhance(pic1_raw)
        if self.input_size != pic1_raw.size[-1]:
            pic1 = self.to_tensor(self.resize(pic1_raw))
            depth1 = torch.tensor(
                np.array(depth1_raw.resize((self.input_size,)*2, 0), dtype=np.float32) * 0.00001525)
        else:
            pic1 = self.to_tensor(pic1_raw)
            depth1 = torch.tensor(np.array(depth1_raw, dtype=np.float32) * 0.00001525)

        pos = torch.zeros(3)  # light and camera co-located (b,3)
        direction = torch.tensor([0, 0, 1])  # light direction straight towards +z (b,3)
        mu = 0  # approximate attenuation in air as 0 (b,)
        light_data = (pos, direction, mu)

        data_dic = {
            # 'pic1': pic1, 'pic2': pic2_list,
            # 'depth1': depth1, 'depth2': depth2,
            # 'tgt_pose': tgt_pose, 'neigh_pose': neigh_pose,
            # 'tgt_pose_inv': tgt_pose_inv, 'neigh_pose_inv': neigh_pose_inv,
            'rgb': pic1,
            'depth': depth1.unsqueeze(0).clip(0., 1.),
            'intrinsics': self.intrinsics,
            'light_data': light_data,
            'n_intrinsics': self.n_intrinsics,
            'depth1_path': '_'.join(str(self.depth_path_ls[seq_id][idx]).split('/')[-2:])
        }
        return data_dic
