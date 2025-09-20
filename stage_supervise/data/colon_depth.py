from pathlib import Path
import logging

import PIL.Image
import torch
import h5py
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

import model.pps_net.utils.optical_flow_funs as OF


class ColonDepth(Dataset):
    def __init__(self, data_mode='train', input_size=352):
        super().__init__()
        logging.info(f'ColonDepth Loading...')
        self.root_dir = '/data/2t/jupyter/wxh/datasets/SimCol/'
        self.input_size = input_size
        if data_mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((input_size,)*2, antialias=True),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((input_size,)*2, antialias=True),
                transforms.ToTensor()
            ])
        if data_mode == 'train':
            self.file = h5py.File(self.root_dir+'train.h5', 'r')
        elif data_mode == 'val':
            self.file = h5py.File(self.root_dir+'val.h5', 'r')
        elif data_mode == 'test':
            self.file = h5py.File(self.root_dir+'test.h5', 'r')

        cam_dir = Path('/data/2t/jupyter/wxh/datasets/SimCol/SyntheticColon_III/cam.txt')
        intrinsics = np.genfromtxt(cam_dir).astype(np.float32).reshape((3, 3))
        intrinsics[0, :] = intrinsics[0, :] * (input_size / 475)
        intrinsics[1, :] = intrinsics[1, :] * (input_size / 475)
        self.intrinsics = intrinsics
        self.n_intrinsics = OF.pixel_intrinsics_to_normalized_intrinsics(
            torch.from_numpy(intrinsics).unsqueeze(0).float(), (input_size,)*2).squeeze(0)

    def __len__(self):
        return len(self.file['neigh'])

    def __getitem__(self, idx):
        raw_image = PIL.Image.fromarray(self.file['image'][idx], mode="RGB")
        rgb = self.transform(raw_image)
        depth = self.file['depth'][idx].astype(np.float32)[None]
        depth = transforms.Resize(
            (self.input_size,)*2, antialias=True,
            interpolation=transforms.InterpolationMode.BILINEAR)(torch.from_numpy(depth))
        depth = depth.clamp(0.0, 1.0)
        # Constants for Per-Pixel Lighting (PPL)
        pos = torch.zeros(3)  # light and camera co-located (b,3)
        direction = torch.tensor([0, 0, 1])  # light direction straight towards +z (b,3)
        mu = 0  # approximate attenuation in air as 0 (b,)
        light_data = (pos, direction, mu)

        return {
            'rgb': rgb,
            'depth': depth,
            'intrinsics': self.intrinsics,
            'light_data': light_data,
            'n_intrinsics': self.n_intrinsics
        }
