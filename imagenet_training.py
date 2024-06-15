import os
import argparse
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import wandb
import mlflow

from libs.Dataset_val import PatchDataGenerator
from models.lit_training import LitTraining

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

tracking_uri = "https://criticalmaas.software-dev.ncsa.illinois.edu/"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'ncsa'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'hydrohpc'
mlflow.set_tracking_uri(uri=tracking_uri)
mlflow.set_experiment("nathan")


def initialize_data_loaders(args):
    train_dataset = PatchDataGenerator(
        data_dir=args.train_data_dir, 
        patch_size=args.patch_size, 
        overlap=args.overlap, 
        norm_type=args.norm_type, 
        hue_factor=args.hue_factor, 
        augment=args.augment,
        valid_patch_rate=args.valid_patch_rate
    )

    val_dataset = PatchDataGenerator(
        data_dir=args.val_data_dir, 
        patch_size=args.patch_size, 
        overlap=args.overlap, 
        norm_type=args.norm_type, 
        hue_factor=args.hue_factor, 
        augment=args.augment,
        valid_patch_rate=args.valid_patch_rate
    )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Please check the data directory or preprocessing steps.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty. Please check the data directory or preprocessing steps.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


def main(args):
    with mlflow.start_run():
        # Append date and time to name_id
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_id = f"{args.name_id}_{args.backbone}_{current_time}"

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
            'dynamic_valid_patch_rate': args.dynamic_valid_patch_rate,
            'name_id': name_id,
            'backbone': args.backbone
        }

        mlflow.log_params(config)

        model = smp.Unet(
            encoder_name=args.backbone,
            encoder_weights="imagenet",
            in_channels=6,
            classes=1,
        )

        lit_model = LitTraining(model, args.learning_rate)

        wandb_logger = WandbLogger(
            project=args.project_name,
            sync_tensorboard=False,
            name=name_id,
            config=config
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val/loss',
            dirpath=os.path.join(args.checkpoint_dir, name_id),
            filename= args.backbone+'-{val/loss:.2f}',
            save_top_k=3,
            mode='min',
        )
        early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.05, patience=50, verbose=True, mode="min")

        train_loader, val_loader = initialize_data_loaders(args)

        trainer = Trainer(
            max_epochs=args.num_epochs,
            accelerator="gpu",
            devices=2, 
            precision=32,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stop_callback],
        )

        trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train U-Net model for segmentation.')

    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--train_data_dir', type=str, default='/projects/bbym/shared/data/commonPatchData/', help='Directory for training data')
    data_group.add_argument('--val_data_dir', type=str, default='/projects/bbym/shared/data/commonPatchData/validation/', help='Directory for validation data')
    data_group.add_argument('--patch_size', type=int, default=256, help='Patch size for data generator')
    data_group.add_argument('--overlap', type=int, default=15, help='Overlap for patches')
    data_group.add_argument('--norm_type', type=str, choices=['basic', 'imagenet', 'none'], default='basic', help='Normalization type')
    data_group.add_argument('--hue_factor', type=float, default=0.2, help='Hue factor for data augmentation')
    data_group.add_argument('--valid_patch_rate', type=float, default=0.75, help='Initial valid patch rate')
    data_group.add_argument('--augment', default=True, type=lambda x: (str(x).lower() == 'true'), help='Whether to use data augmentation')
    data_group.add_argument('--num_workers', type=int, default=12, help='Number of workers for data loader')

    training_group = parser.add_argument_group('Training')
    training_group.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    training_group.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    training_group.add_argument('--num_epochs', type=int, default=15, help='Number of epochs to train')
    training_group.add_argument('--dynamic_valid_patch_rate', default=False, type=lambda x: (str(x).lower() == 'true'), help='Dynamically update valid_patch_rate each epoch')

    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    log_group.add_argument('--project_name', type=str, default='ImageNet_Segmentation', help='WandB project name')
    log_group.add_argument('--name_id', type=str, default='experiment', help='WandB run name')
    log_group.add_argument('--backbone', type=str, default='resnet50', help='Backbone model for U-Net')

    args = parser.parse_args()

    main(args)
