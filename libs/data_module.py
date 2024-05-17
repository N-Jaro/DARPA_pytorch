import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from .Dataset_val import PatchDataGenerator

class PatchDataModule(LightningDataModule):
    def __init__(self, train_data_dir, val_data_dir, patch_size, overlap, norm_type, hue_factor, augment, batch_size, num_workers, valid_patch_rate):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.patch_size = patch_size
        self.overlap = overlap
        self.norm_type = norm_type
        self.hue_factor = hue_factor
        self.augment = augment
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_patch_rate = valid_patch_rate

    def setup(self, stage=None):
        self.train_dataset = PatchDataGenerator(
            data_dir=self.train_data_dir,
            patch_size=self.patch_size,
            overlap=self.overlap,
            norm_type=self.norm_type,
            hue_factor=self.hue_factor,
            augment=self.augment,
            valid_patch_rate=self.valid_patch_rate
        )

        self.val_dataset = PatchDataGenerator(
            data_dir=self.val_data_dir,
            patch_size=self.patch_size,
            overlap=self.overlap,
            norm_type=self.norm_type,
            hue_factor=self.hue_factor,
            augment=self.augment,
            valid_patch_rate=self.valid_patch_rate
        )

    def update_datasets(self, valid_patch_rate):
        self.train_dataset = PatchDataGenerator(
            data_dir=self.train_data_dir,
            patch_size=self.patch_size,
            overlap=self.overlap,
            norm_type=self.norm_type,
            hue_factor=self.hue_factor,
            augment=self.augment,
            valid_patch_rate=valid_patch_rate
        )

        self.val_dataset = PatchDataGenerator(
            data_dir=self.val_data_dir,
            patch_size=self.patch_size,
            overlap=self.overlap,
            norm_type=self.norm_type,
            hue_factor=self.hue_factor,
            augment=self.augment,
            valid_patch_rate=valid_patch_rate
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
