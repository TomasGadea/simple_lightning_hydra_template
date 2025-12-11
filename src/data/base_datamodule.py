from lightning import LightningDataModule
import hydra
from torch.utils.data import DataLoader
from typing import Optional


class BaseDatamodule(LightningDataModule):
    def __init__(
        self,
        dataset,
        batch_size: int,
        num_workers: int,
        train_val_test_split: list,
        pin_memory: bool = False,
        overfit_batch: int = 0,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.pin_memory = pin_memory
        self.overfit_batch = overfit_batch

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = hydra.utils.instantiate(self.dataset)
        self.val_dataset = hydra.utils.instantiate(self.dataset)
        self.test_dataset = hydra.utils.instantiate(self.dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                            persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)