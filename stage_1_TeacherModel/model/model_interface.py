import importlib
import inspect
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim.lr_scheduler as lrs
import torchvision
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
        if (self.trainer.current_epoch + 1) % 10 == 0 and self.trainer.is_last_batch:
            log_dict = self.model.train_step(batch, output_pred=True)
            output_pred = log_dict.pop('pred')
            rgb_grid = torchvision.utils.make_grid(batch['rgb'][:4])
            self.logger.experiment.add_image("train/rgb", rgb_grid, self.current_epoch)
            mask_pred_grid = torchvision.utils.make_grid(output_pred['mask'][:4])
            mask_gt_grid = torchvision.utils.make_grid(output_pred['mask_gt'][:4])
            mask_grid = torch.cat([mask_gt_grid, mask_pred_grid], dim=1)
            self.logger.experiment.add_image("train/mask", mask_grid, self.current_epoch)
            pred_grid = torchvision.utils.make_grid(output_pred['pred'][:4].clip(0., 1.))
            self.logger.experiment.add_image("train/pred", pred_grid, self.current_epoch)
        else:
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

    def on_predict_epoch_start(self):
        self.write_f = open('pred_mask_mean_val_full.txt', 'w')

    def predict_step(self, batch, batch_idx):
        img_tensor = batch['rgb']
        batch_size = img_tensor.size(0)
        assert batch_size == 1
        depth_est, mask_est = self.forward(batch)
        pred_mask_mean_val = mask_est.mean()
        self.write_f.write(f"{batch['file_dir'][0].split('.')[0]} {pred_mask_mean_val.item()}\n")
        depth_est = np.clip(depth_est.detach().cpu().numpy() * 65535, 0, 65536).astype(np.uint16)
        filedir: list[Path] = batch['file_dir']
        output_dir = Path('/home/star/datasets/wxh/real_colon_clip_pics_est_full') / filedir[0].replace('jpg', 'tiff')
        im = Image.fromarray(depth_est[0, 0])
        im.save(output_dir)

    def on_predict_epoch_end(self):
        self.write_f.close()

    def test_step(self, batch, batch_idx):
        return self.depth_test_step(batch, batch_idx)

    def depth_test_step(self, batch, batch_idx):
        depth = batch['depth'] * 10
        img_tensor = batch['rgb']
        batch_size = img_tensor.size(0)

        depth_est = self.forward(batch) * 10
        #depth_est = scale_predictions(depth, depth_est)
        epsilon = 1e-8
        l1 = torch.abs(depth - depth_est).mean()
        abs_rel = torch.abs(depth - depth_est) / (depth + epsilon)
        sq_rel = torch.mean(((depth - depth_est) / (depth + epsilon)) ** 2)
        # sq_rel = torch.mean(((depth - depth_est) ** 2) / (depth + epsilon))
        rmse = torch.sqrt(torch.mean((depth - depth_est) ** 2))
        si = torch.max(depth / (depth_est + epsilon), depth_est / (depth + epsilon))
        s1 = (si < 1.05).float().mean()
        s2 = (si < (1.05**2)).float().mean()
        s3 = (si < (1.05**3)).float().mean()
        # log10error = (torch.log10(depth) - torch.log10(depth_est)).abs().mean()
        a1 = (abs_rel < 0.1).float().mean()

        # if depth.min() < 1e-5:
        #     ling = torch.tensor([0.]).to(depth)
        #     print(0)
        #     self.log('abs_rel', ling, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size,
        #              sync_dist=True)
        #     self.log('sq_rel', ling, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size,
        #              sync_dist=True)
        #     self.log('rmse', rmse.mean(), on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        #     self.log('a1', a1, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        #     return ling
        self.log('l1', l1, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size,
                 sync_dist=True)
        self.log('abs_rel', abs_rel.mean(), on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('sq_rel', sq_rel, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('rmse', rmse.mean(), on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('s1', s1, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('s2', s2, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('s3', s3, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        # self.log('log10error', log10error, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)

        self.log('a1', a1, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)

        return rmse

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 1e-4
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "pretrained" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "pretrained" in n and p.requires_grad],
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

def scale_predictions(gt, est):
    from scipy.optimize import leastsq
    # Flatten the ground truth and estimated depth arrays
    gt_flat = gt.detach().cpu().numpy().flatten()
    est_flat = est.detach().cpu().numpy().flatten()

    # Calculate the scaling factor using least median squares
    def error_func(scale, gt, est):
        return np.median((gt - scale * est) ** 2)

    # Initial guess for the scaling factor
    initial_scale = 1.0

    # Use the least median squares to find the optimal scaling factor
    result = leastsq(error_func, initial_scale, args=(gt_flat, est_flat))
    optimal_scale = result[0][0]

    # Scale the estimated depth array
    scaled_est = est * optimal_scale

    return scaled_est
