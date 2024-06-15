import os
import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
import mlflow

from models.unet import UNet
from models.unetTransformer import U_Transformer

# from libs.data_module import PatchDataModule
from libs.data_scale_loader import ScaleDataGenerator
from libs.callbacks import ValidPatchRateCallback
from models.lit_training import LitTraining

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

tracking_uri = "https://criticalmaas.software-dev.ncsa.illinois.edu/"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'ncsa'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'hydrohpc'
mlflow.set_tracking_uri(uri=tracking_uri)
mlflow.set_experiment("nathan")


def initialize_data_loaders(args):
    train_loader = ScaleDataGenerator.create_train_dataset(
        batch_size=args.batch_size, 
        patch_size=args.patch_size, 
        overlap=args.overlap, 
        norm_type=args.norm_type, 
        hue_factor=args.hue_factor, 
        augment=args.augment,
        valid_patch_rate=args.valid_patch_rate
    )

    val_loader = ScaleDataGenerator.create_validation_dataset(
        batch_size=args.batch_size, 
        patch_size=args.patch_size, 
        overlap=args.overlap, 
        norm_type=args.norm_type, 
        hue_factor=args.hue_factor, 
        augment=args.augment,
        valid_patch_rate=args.valid_patch_rate
    )

    # Check if the datasets are empty
    if len(train_loader.dataset) == 0:
        raise ValueError("Training dataset is empty. Please check the data directory or preprocessing steps.")
    if len(val_loader.dataset) == 0:
        raise ValueError("Validation dataset is empty. Please check the data directory or preprocessing steps.")

    return train_loader, val_loader


def main(args):

    # Start an MLflow run
    with mlflow.start_run():

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

        # Log parameters
        mlflow.log_params(config)

        # Initialize model
        model = U_Transformer(in_channels=6, classes=1)

        # Initialize Lightning model
        lit_model = LitTraining(model, args.learning_rate)
        
        # Initialize Wandb logger
        wandb_logger = WandbLogger( project=args.project_name,
                                    sync_tensorboard=False,
                                    name=args.name_id,
                                    config=config)

        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor='val/loss',
            dirpath=os.path.join(args.checkpoint_dir, args.name_id),
            filename='u-transformer-best',
            save_top_k=1,
            mode='min',
        )
        early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=50, verbose=True, mode="min")

        # Initialize data loaders
        train_loader, val_loader = initialize_data_loaders(args)

        # Trainer
        trainer = Trainer(
                    max_epochs=args.num_epochs,
                    accelerator="gpu",
                    devices=2, 
                    precision=32,
                    logger=wandb_logger,
                    callbacks=[checkpoint_callback, early_stop_callback],
                    resume_from_checkpoint=args.checkpoint_file if args.checkpoint_file else None,
                )

        # Training
        trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train U-Transformer model for segmentation.')

    # Data-related arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--train_data_dir', type=str, default='/projects/bcxi/nathanj/commonPatchData/', help='Directory for training data')
    data_group.add_argument('--val_data_dir', type=str, default='/projects/bcxi/nathanj/commonPatchData/validation/', help='Directory for validation data')
    data_group.add_argument('--patch_size', type=int, default=256, help='Patch size for data generator')
    data_group.add_argument('--overlap', type=int, default=15, help='Overlap for patches')
    data_group.add_argument('--norm_type', type=str, choices=['basic', 'imagenet', 'none'], default='basic', help='Normalization type')
    data_group.add_argument('--hue_factor', type=float, default=0.2, help='Hue factor for data augmentation')
    data_group.add_argument('--valid_patch_rate', type=float, default=0.75, help='Initial valid patch rate')
    data_group.add_argument('--augment', default=True, type=lambda x: (str(x).lower() == 'true'), help='Whether to use data augmentation')
    data_group.add_argument('--num_workers', type=int, default=12, help='Number of workers for data loader')

    # Training-related arguments
    training_group = parser.add_argument_group('Training')
    training_group.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    training_group.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    training_group.add_argument('--num_epochs', type=int, default=3, help='Number of epochs to train')
    training_group.add_argument('--dynamic_valid_patch_rate', default=True, type=lambda x: (str(x).lower() == 'true'), help='Dynamically update valid_patch_rate each epoch')

    # Checkpoint and logging arguments
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    log_group.add_argument('--checkpoint_file', type=str, default=None, help='Path to the checkpoint file to resume training')
    log_group.add_argument('--project_name', type=str, default='U_Transformer_Segmentation_scales', help='WandB project name')
    log_group.add_argument('--name_id', type=str, default='experiment', help='WandB run name')

    args = parser.parse_args()

    main(args)


# python training_scale.py --name_id "scale_experiment" --checkpoint_file "/u/nathanj/DARPA_pytorch/checkpoints/Utransformer_256_32_w_val_constant_rate/u-transformer-val/loss=0.29.ckpt"