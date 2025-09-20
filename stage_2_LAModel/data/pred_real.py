from pathlib import Path
import logging

import PIL.Image
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

import model.pps_net.utils.optical_flow_funs as OF


class RealColon(Dataset):
    def __init__(self, input_size=352, **kwargs):
        super().__init__()
        logging.info(f'RealColon Loading...')
        self.root_dir = Path('/home/star/datasets/wxh/exColon/')
        self.input_size = input_size
        self.resize_transform = transforms.Resize((input_size,)*2, antialias=True)
        self.to_tensor_transform = transforms.ToTensor()
        # self.transform = transforms.Compose([
        #     transforms.Resize((input_size,)*2, antialias=True),
        #     transforms.ToTensor(),
        # ])
        self.file = list(self.root_dir.glob('*.png'))

        with np.load('/data/2t/jupyter/wxh/PPSNetRe/calibration.npz') as data:
            intrinsics = data['K']
        intrinsics[0, :] = intrinsics[0, :] * (input_size / 1158)
        intrinsics[1, :] = intrinsics[1, :] * (input_size / 1158)
        self.intrinsics = intrinsics
        self.n_intrinsics = OF.pixel_intrinsics_to_normalized_intrinsics(
            torch.from_numpy(intrinsics).unsqueeze(0).float(), (input_size,)*2).squeeze(0)

    def __len__(self):
        # return len(self.file)
        return 200

    def __getitem__(self, idx):
        raw_image = PIL.Image.open(self.file[idx])
        if raw_image.size[0] != self.input_size:
            raw_image = self.resize_transform(raw_image)
        rgb = self.to_tensor_transform(raw_image)
        # Constants for Per-Pixel Lighting (PPL)
        pos = torch.zeros(3)  # light and camera co-located (b,3)
        direction = torch.tensor([0, 0, 1])  # light direction straight towards +z (b,3)
        mu = 0  # approximate attenuation in air as 0 (b,)
        light_data = (pos, direction, mu)

        return {
            'rgb': rgb,
            'intrinsics': self.intrinsics,
            'light_data': light_data,
            'n_intrinsics': self.n_intrinsics,
            'file_dir': str(self.file[idx].name),
        }
