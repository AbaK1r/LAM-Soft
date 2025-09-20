from pathlib import Path
import importlib
import inspect

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim.lr_scheduler as lrs
from PIL import Image


class ModuleInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.instancialize()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def log_lr(self, batch_size):
        opt = self.optimizers()
        if isinstance(opt, list):
            opt = opt[0]  # 如果有多个优化器，选择第一个

        for i, param_group in enumerate(opt.param_groups):
            self.log(f'lr_group_{i}', param_group['lr'],
                     on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

    def training_step(self, batch, batch_idx):
        log_dict = self.model.train_step(batch)
        batch_size = batch['rgb'].size(0)
        loss_total = log_dict['loss']
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log_lr(batch_size)
        return loss_total

    def validation_step(self, batch, batch_idx):
        log_dict = self.model.train_step(batch)
        batch_size = batch['rgb'].size(0)
        loss_total = log_dict['loss']
        log_dict = {f'o_{k}': v for k, v in log_dict.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        return loss_total

    def test_step(self, batch, batch_idx):
        return self.depth_test_step(batch, batch_idx)

    def depth_test_step(self, batch, batch_idx):
        log_dict = self.model.test_step(batch)
        batch_size = batch['rgb'].size(0)
        loss_total = log_dict['loss']
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        return loss_total

    def predict_step(self, batch, batch_idx):
        batch_size = batch['rgb'].size(0)
        assert batch_size == 1
        depth_est = self.model.predict_step(batch)[0, 0]
        depth_est = np.clip(depth_est.detach().cpu().numpy() * 65535, 0, 65536).astype(np.uint16)
        filedir: list[str] = batch['filedir']
        output_dir = Path(self.hparams.predict_output_dir) / (filedir[0].split('/')[-1].replace('jpg', 'tiff'))
        im = Image.fromarray(depth_est)
        im.save(output_dir)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 1e-4
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.hparams.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.hparams.lr, weight_decay=weight_decay)
        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma)
                # scheduler = lrs.ReduceLROnPlateau(
                #     optimizer, mode='min', factor=0.33, patience=4, threshold=1e-6, min_lr=9e-7, cooldown=1)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    # 'monitor': 'loss_epoch',
                }
            }

    def instancialize(self, **other_args):
        """
        Instancialize a model using the corresponding parameters
        from self.hparams dictionary. You can also input any args
        to overwrite the corresponding value in self.kwargs.
        """
        camel_name = ''.join([i.capitalize() for i in self.hparams.model_name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + self.hparams.model_name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {self.hparams.model_name}.{camel_name}!')
        class_args = inspect.getfullargspec(Model.__init__).args[1:]

        args = {arg: self.hparams[arg] for arg in class_args if arg in self.hparams.keys()}
        args.update(other_args)
        return Model(**args)


def logq_to_quaternion(q):
    # return: quaternion with w, x, y, z
    # from geomap paper
    n = torch.norm(q, p=2, dim=1, keepdim=True)
    n = torch.clamp(n, min=1e-8)
    q = q * torch.sin(n)
    q = q / n
    q = torch.cat((torch.cos(n), q), dim=1)
    return q


def quat2mat(q):
    import nibabel.quaternions as nq
    return nq.quat2mat(q)


def get_traj(first, rots, trans, direction='forward'):
    traj = []
    traj_4x4 = []
    next = first
    traj.append(next[:3, -1])
    traj_4x4.append(first)
    Ps = []

    if direction == 'forward':
        for i in range(0, rots.shape[0]):
            ri = rots[i, :, :]

            Pi = np.concatenate((ri, trans[i].reshape((3, 1))), 1)
            Pi = np.concatenate((Pi, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)

            next = np.matmul(next, Pi)
            traj.append(next[:3, -1])
            traj_4x4.append(next)
            Ps.append(Pi)
    elif direction == 'backward':
        for i in range(rots.shape[0] - 1, -1, -1):
            ri = rots[i, :, :]

            Pi = np.concatenate((ri, trans[i].reshape((3, 1))), 1)
            Pi = np.concatenate((Pi, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)

            next = np.matmul(next, Pi)
            traj.append(next[:3, -1])
            traj_4x4.append(next)
            Ps.append(Pi)

    traj = np.array(traj)
    traj_4x4 = np.array(traj_4x4)
    Ps = np.array(Ps)
    return traj, traj_4x4, Ps


def compute_ate_rte(gt, pred, delta=1, plot=True):
    errs = []
    rot_err = []
    rot_gt = []
    trans_gt = []
    for i in range(pred.shape[0] - delta):
        Q = np.linalg.inv(gt[i, :, :]) @ gt[i + delta, :, :]
        P = np.linalg.inv(pred[i, :, :]) @ pred[i + delta, :, :]
        E = np.linalg.inv(Q) @ P
        t = E[:3, -1]
        t_gt = Q[:3, -1]
        trans = np.linalg.norm(t, ord=2)
        errs.append(trans)
        tr = np.arccos((np.trace(E[:3, :3]) - 1) / 2)
        gt_tr = np.arccos((np.trace(Q[:3, :3]) - 1) / 2)
        rot_err.append(tr)
        rot_gt.append(gt_tr)
        trans_gt.append(np.linalg.norm(t_gt, ord=2))

    errs = np.array(errs)

    scale_factor = np.sum(gt[:, :, -1] * pred[:, :, -1]) / np.sum(pred[:, :, -1] ** 2)
    ATE_endo = np.median(np.linalg.norm((gt[:, :, -1] - scale_factor * pred[:, :, -1]), ord=2, axis=1))
    ATE = np.median(np.linalg.norm((gt[:, :, -1] - pred[:, :, -1]), ord=2, axis=1))
    RTE = np.median(errs)
    ROT = np.median(rot_err)

    return ATE, RTE, errs, ROT, np.mean(rot_gt) * 180 / np.pi
