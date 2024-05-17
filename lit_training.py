import os
import argparse

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import BinaryAccuracy

import matplotlib.pyplot as plt
import numpy as np

import wandb
from models.unet import UNet
from models.unetTransformer import U_Transformer

from libs.Normalization import MinMaxNormalizer, BasicNormalizer, ImageNetNormalizer
from libs.Dataset_val import PatchDataGenerator

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

class LitUTransformer(LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super(LitUTransformer, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = self.dice_loss
        self.accuracy = BinaryAccuracy()

    def dice_loss(self, pred, target):
        smooth = 1.
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2,3))
        loss = 1 - ((2. * intersection + smooth) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth))
        return loss.mean()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy((y_hat > 0.5).float(), y)
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy((y_hat > 0.5).float(), y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy((y_hat > 0.5).float(), y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class ValidPatchRateCallback(Callback):
    def __init__(self, train_data_dir, validation_data_dir, patch_size=256, overlap=15, norm_type='-1', hue_factor=0.2, augment=True, initial_patch_rate=1.0, rate_decay=0.005, dynamic=False):
        super().__init__()
        self.initial_patch_rate = initial_patch_rate
        self.rate_decay = rate_decay
        self.dynamic = dynamic
        self.train_data_dir= train_data_dir, 
        self.validation_data_dir= validation_data_dir, 
        self.patch_size=patch_size, 
        self.overlap=overlap, 
        self.norm_type=norm_type, 
        self.hue_factor=hue_factor, 
        self.augment=augment, 

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if self.dynamic or epoch == 0:
            new_patch_rate = max(0, self.initial_patch_rate - epoch * self.rate_decay)
            
            # Reinitialize the datasets and dataloaders with the new valid_patch_rate
            train_generator = PatchDataGenerator(data_dir=self.train_data_dir, patch_size=self.patch_size, overlap=self.overlap, norm_type=self.norm_type, hue_factor=self.hue_factor, augment=self.augment, valid_patch_rate = new_patch_rate)
            trainer.datamodule.train_dataloader = DataLoader(train_generator, batch_size=trainer.datamodule.batch_size, shuffle=True, num_workers=trainer.datamodule.num_workers)

            validation_generator = PatchDataGenerator(data_dir=self.validation_data_dir, patch_size=self.patch_size, overlap=self.overlap, norm_type=self.norm_type, hue_factor=self.hue_factor, augment=self.augment, valid_patch_rate = new_patch_rate)
            trainer.datamodule.val_dataloader = DataLoader(validation_generator, batch_size=trainer.datamodule.batch_size, shuffle=False, num_workers=trainer.datamodule.num_workers)

            # Log the new valid_patch_rate
            pl_module.log('valid_patch_rate', new_patch_rate)

def main(args):
    # Config for wandb
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'patch_size': args.patch_size,
        'overlap': args.overlap,
        'norm_type': args.norm_type,
        'hue_factor': args.hue_factor,
        'valid_patch_rate': args.valid_patch_rate,
        'augment': args.augment,
        'num_workers': args.num_workers,
        'dynamic_valid_patch_rate': args.dynamic_valid_patch_rate
    }

    # Initialize wandb
    wandb.init(
        project=args.project_name,
        sync_tensorboard=False,
        name=args.name_id,
        config=config
    )

    # Define the shape of the tensor
    B = args.batch_size  # Batch size
    C = 6  # Number of channels
    H = args.patch_size  # Height
    W = args.patch_size  # Width
    num_samples = 100  # Number of samples in the dataset

    class DummyDataset(Dataset):
        def __init__(self, num_samples, B, C, H, W):
            self.num_samples = num_samples
            self.B = B
            self.C = C
            self.H = H
            self.W = W

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Generate random input data and label
            input_data = torch.randn(self.C, self.H, self.W).float() 
            label = torch.randint(0, 2, (1, self.H, self.W)).float()  # Binary labels
            return input_data, label


    # Create an instance of the dataset
    dummy_dataset = DummyDataset(num_samples, B, C, H, W)

    # Create a DataLoader
    train_data = DataLoader(dummy_dataset, batch_size=B, shuffle=True)
    validation_data = DataLoader(dummy_dataset, batch_size=B, shuffle=False)

    # # Create initial dataset and dataloader
    # train_generator = PatchDataGenerator(data_dir=args.train_data_dir, patch_size=args.patch_size, overlap=args.overlap, norm_type=args.norm_type, hue_factor=args.hue_factor, valid_patch_rate=args.valid_patch_rate, augment=args.augment)
    # train_data = DataLoader(train_generator, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # validation_generator = PatchDataGenerator(data_dir=args.val_data_dir, patch_size=args.patch_size, overlap=args.overlap, norm_type=args.norm_type, hue_factor=args.hue_factor, valid_patch_rate=args.valid_patch_rate, augment=args.augment)
    # validation_data = DataLoader(validation_generator, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize model
    model = U_Transformer(in_channels=6, classes=1)

    # Initialize Lightning model
    lit_model = LitUTransformer(model, args.learning_rate)

    # Initialize Wandb logger
    wandb_logger = WandbLogger(project=args.project_name)

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath= os.path.join(args.checkpoint_dir, args.name_id),
        filename='u-transformer-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    # Trainer
    valid_patch_callback = ValidPatchRateCallback(
        initial_patch_rate=args.valid_patch_rate, 
        rate_decay=0.005, 
        dynamic=args.dynamic_valid_patch_rate, 
        train_data_dir=args.train_data_dir, 
        validation_data_dir=args.val_data_dir, 
        patch_size=args.patch_size, 
        overlap=args.overlap, 
        norm_type=args.norm_type, 
        hue_factor=args.hue_factor, 
        augment=args.augment
    )
    
    trainer = Trainer(
        max_epochs=args.num_epochs,
        callbacks=[checkpoint_callback, valid_patch_callback],
        logger=wandb_logger
    )

    # Training
    trainer.fit(lit_model, train_data, validation_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train U-Transformer model for segmentation.')

    # Data-related arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--train_data_dir', type=str, default='/projects/bbym/shared/data/commonPatchData/', help='Directory for training data')
    data_group.add_argument('--val_data_dir', type=str, default='/projects/bbym/shared/data/commonPatchData/validation', help='Directory for validation data')
    data_group.add_argument('--patch_size', type=int, default=256, help='Patch size for data generator')
    data_group.add_argument('--overlap', type=int, default=32, help='Overlap for patches')
    data_group.add_argument('--norm_type', type=str, choices=['basic', 'imagenet', 'none'], default='basic', help='Normalization type')
    data_group.add_argument('--hue_factor', type=float, default=0.2, help='Hue factor for data augmentation')
    data_group.add_argument('--valid_patch_rate', type=float, default=1, help='Initial valid patch rate')
    data_group.add_argument('--augment', default=True, type=lambda x: (str(x).lower() == 'true'), help='Whether to use data augmentation')
    data_group.add_argument('--num_workers', type=int, default=12, help='Number of workers for data loader')

    # Training-related arguments
    training_group = parser.add_argument_group('Training')
    training_group.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    training_group.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    training_group.add_argument('--num_epochs', type=int, default=150, help='Number of epochs to train')
    training_group.add_argument('--dynamic_valid_patch_rate', default=True, type=lambda x: (str(x).lower() == 'true'), help='Dynamically update valid_patch_rate each epoch')

    # Checkpoint and logging arguments
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    log_group.add_argument('--project_name', type=str, default='U_Transformer_Segmentation', help='WandB project name')
    log_group.add_argument('--name_id', type=str, default='experiment', help='WandB run name')

    args = parser.parse_args()

    main(args)
