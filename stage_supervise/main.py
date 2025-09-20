import logging
import os
import shutil
from pathlib import Path

# os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger

from data import DataInterface
from model import ModuleInterface

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.set_float32_matmul_precision('high')


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(172)
    logger = TensorBoardLogger('')

    if pl.utilities.rank_zero_only.rank == 0:
        src_dirs = [Path('model'), Path('conf'), Path('data')]
        dst_dir = Path(logger.log_dir) / 'code'
        dst_dir.mkdir(parents=True, exist_ok=True)
        copy_code_to_log_dir(src_dirs, dst_dir)
        os.system(f'cp main.py {str(dst_dir / "main.py")}')
    args = cfg.train
    model = ModuleInterface(**args)
    ddir = 'lightning_logs/version_47/checkpoints/last.ckpt'
    # model = ModuleInterface.load_from_checkpoint(checkpoint_path=ddir, **args)
    # model.load_state_dict(torch.load(ddir)['state_dict'], strict=True)
    # model.model.load_pretrained()
    dl = DataInterface(**args)
    checkpoint_callback = ModelCheckpoint(
        monitor='o_rsme',
        filename=f'{args["model_name"]}' + '-{epoch:03d}-{o_rsme:.5f}-{o_l1_loss:.5f}',
        save_top_k=2,
        save_weights_only=False,
        mode='min',
        save_last=True
    )
    early_stop_callback = EarlyStopping(
        monitor="rsme", min_delta=0.00, patience=100, verbose=False, mode="min", check_finite=True)
    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        # StochasticWeightAveraging(swa_lrs=1e-2)
    ]
    trainer = pl.Trainer(
        **args.trainer,
        callbacks=callbacks,
        logger=logger
    )
    trainer.fit(model, datamodule=dl, ckpt_path=None)


# 忽略特定子目录的函数
def ignore_pycache(dir, files):
    """
    Filters out the '__pycache__' directory and its contents.

    :param dir: Directory path (not used).
    :param files: List of files/directories in the given directory.
    :return: List of files/directories to ignore.
    """
    return [f for f in files if f.startswith("__pycache__")]


def copy_code_to_log_dir(src_dirs, dst_dir):
    """
    Copies the content of source directories to destination directory.

    :param src_dirs: List of source directories to copy from.
    :param dst_dir: Destination directory to copy to.
    """
    for src_dir in src_dirs:
        if src_dir.exists() and src_dir.is_dir():
            logging.info(f'copy {src_dir} to {dst_dir / src_dir.name}')
            shutil.copytree(src_dir, dst_dir / src_dir.name, dirs_exist_ok=True, ignore=ignore_pycache)
        else:
            logging.info(f'缺失文件夹{src_dir.name}，已跳过！')


if __name__ == '__main__':
    main()
