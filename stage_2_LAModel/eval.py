import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from data import DataInterface
from model import ModuleInterface

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.set_float32_matmul_precision('high')

ddir = Path('lightning_logs/version_61/checkpoints/depth_anything_v2-epoch=075-o_l1_loss=0.02286.ckpt')


@hydra.main(config_path=str(ddir.parent.parent / "code/conf"), config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    args = cfg.train
    args.val_batch_size = 1
    args.predict_output_dir = f'/home/star/datasets/wxh/real_colon_ref_pps_{args.input_size}_sample'
    Path(args.predict_output_dir).mkdir(parents=True, exist_ok=True)
    print(f'output_dir: {args.predict_output_dir}')
    args.trainer.devices = 1
    args.trainer.strategy = "auto"

    # ddir = 'lightning_logs/version_28/checkpoints/depth_anything_v2-epoch=100-o_l1_loss=0.02365.ckpt'

    model = ModuleInterface.load_from_checkpoint(checkpoint_path=ddir, **args)
    dl = DataInterface(**args)

    trainer = pl.Trainer(
        **args.trainer,
    )
    trainer.predict(model, datamodule=dl)


if __name__ == '__main__':
    main()
