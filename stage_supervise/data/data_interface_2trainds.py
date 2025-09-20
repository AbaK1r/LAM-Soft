import importlib
import inspect
import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class CombinedLoader:
    def __init__(self, loaders):
        self.loaders = loaders
        self.loader_iters = [iter(loader) for loader in loaders]
        self.current_loader_idx = 0
        # 假设每个 DataLoader 的长度相同
        self.length = min(len(loader) for loader in loaders) * len(loaders)

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        # 从当前 DataLoader 获取 batch
        try:
            batch = next(self.loader_iters[self.current_loader_idx])
        except StopIteration:
            # 如果当前 DataLoader 耗尽数据，重置迭代器
            self.loader_iters[self.current_loader_idx] = iter(self.loaders[self.current_loader_idx])
            batch = next(self.loader_iters[self.current_loader_idx])

        # 切换到下一个 DataLoader
        self.current_loader_idx = (self.current_loader_idx + 1) % len(self.loaders)

        return batch


class DataInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.trainset_sup = None
        self.trainset_real = None
        self.valset = None
        self.testset = None
        self.collate_fn = None

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders
        logging.info(f'stage is {stage}, set dataset')
        if stage == 'fit':
            self.trainset_sup, self.collate_fn = self.instancialize(data_mode='train', dataset='colon_depth')
            self.trainset_real, self.collate_fn = self.instancialize(data_mode='train', dataset='real_colon')
            self.valset, _ = self.instancialize(data_mode='val', dataset=self.hparams.dataset)
        elif stage == 'validate' or stage == 'predict':
            self.valset, self.collate_fn = self.instancialize(data_mode='val', dataset=self.hparams.dataset)
        elif stage == 'test':
            self.testset, self.collate_fn = self.instancialize(data_mode='test', dataset=self.hparams.dataset)

    def train_dataloader(self):
        return CombinedLoader([
            DataLoader(
                self.trainset_sup, batch_size=self.hparams.train_batch_size,
                num_workers=self.hparams.num_workers, shuffle=True,
                pin_memory=True, collate_fn=self.collate_fn, drop_last=True
            ),
            DataLoader(
                self.trainset_real, batch_size=self.hparams.train_real_batch_size,
                num_workers=self.hparams.num_workers, shuffle=True,
                pin_memory=True, collate_fn=self.collate_fn, drop_last=True
            )
        ])

    def val_dataloader(self):
        return DataLoader(
            self.valset, batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers, shuffle=False,
            pin_memory=True, collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset, batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers, shuffle=False,
            pin_memory=True, collate_fn=self.collate_fn
        )

    def instancialize(self, dataset, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        camel_name = ''.join([i.capitalize() for i in dataset.split('_')])
        # print(camel_name)
        try:
            data_module = getattr(importlib.import_module(
                '.' + dataset, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{dataset}.{camel_name}')

        class_args = inspect.getfullargspec(data_module.__init__).args[1:]

        args = {arg: self.hparams[arg] for arg in class_args if arg in self.hparams.keys()}
        args.update(other_args)

        try:
            collate_fn = getattr(importlib.import_module(
                '.' + dataset, package=__package__), 'collate_fn')
            logging.info('collate_fn was successfully loaded.')
        except:
            collate_fn = None
            logging.info('collate_fn not found! Use default collate_fn.')

        return data_module(**args), collate_fn
