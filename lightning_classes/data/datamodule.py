from lightning import LightningDataModule
from typing import Optional, Dict, Any
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch

class DataModule(LightningDataModule):
    def __init__(self, 
    train_val_test_split: None, 
    data_dir: None,
    batch_size: 32,
    num_worker: 1,
    pin_memory: True
    ): # include some all parameters that needs to accessed by 'self.hparams'; linked to data config file.
        super().__init__()
        # allows access to 'self.hparams', which configured by the data config file.
        self.save_hyperparameters(logger=False)
        # initialising train/val/test data
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """ 
        to load/download data and prep data with data processing (if needed).
        """
        self.data = pd.read(self.hparams.data_dir)
        # data transformation here

    def setup(self):
        # load train, val and test data if it has not been setup
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset = self.data, 
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42)
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

if __name__ == "__main__":
    _ = DataModule()
