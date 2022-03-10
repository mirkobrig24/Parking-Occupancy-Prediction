import argparse
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
#from torch_geometric.loader import DataLoader
import utils.data.functions


class SpatioTemporalCSVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        feat_path: str,
        adj_path: str,
        batch_size: int,
        seq_len: int,
        pre_len: int = 1,
        split_ratio: float = 0.8,
        normalize: bool = True,
        **kwargs
    ):
        super(SpatioTemporalCSVDataModule, self).__init__()
        self._feat_path = feat_path
        self._adj_path = adj_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        self._feat = utils.data.functions.load_features(self._feat_path)
        self._feat_max_val = np.max(self._feat)
        self._adj = utils.data.functions.load_adjacency_matrix(self._adj_path)

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--seq_len", type=int, default=12)
        parser.add_argument("--pre_len", type=int, default=3)
        parser.add_argument("--split_ratio", type=float, default=0.8)
        parser.add_argument("--normalize", type=bool, default=True)
        return parser

    def setup(self, stage: str = None):
        (
            self.train_dataset,
            self.val_dataset,
        ) = utils.data.functions.generate_torch_datasets(
            self._feat,
            self._adj,
            self.seq_len,
            self.pre_len,
            split_ratio=self.split_ratio,
            normalize=self.normalize,
        )

    def train_dataloader(self):
        #print('---TRAIN---', self.train_dataset.size())
        print('---TRAIN---', len(self.train_dataset))
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        #print('---LEN1---', self.val_dataset[0].size())
        #print('---LEN1---', self.val_dataset[1].size())
        print('---LEN2---', len(self.val_dataset))
        #print('---LEN3---', len(self.val_dataset[0]))
        #exit()
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def adj(self):
        return self._adj

    @property
    def feat(self):
        return self._feat
