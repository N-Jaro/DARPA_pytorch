import os
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
import mlflow
from mlflow.models.signature import infer_signature

from models.unet import UNet
from models.unetTransformer import U_Transformer, U_Transformer_Lightning

from libs.Dataset_val import PatchDataGenerator
from libs.callbacks import ValidPatchRateCallback
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
        valid_patch_rate=args.valid_patch_rate,
        test=False
    )

    val_dataset = PatchDataGenerator(
        data_dir=args.val_data_dir, 
        patch_size=args.patch_size, 
        overlap=args.overlap, 
        norm_type=args.norm_type, 
        hue_factor=args.hue_factor, 
        augment=args.augment,
        valid_patch_rate=args.valid_patch_rate,
        test=False
    )

    # Check if the datasets are empty
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Please check the data directory or preprocessing steps.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty. Please check the data directory or preprocessing steps.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


def main(args):
    
    pl.seed_everything(4444)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_id = f"{args.name_id}_{current_time}"

    mlflow.set_experiment(args.project_name)

    # Start an MLflow run
    with mlflow.start_run(run_name = name_id):

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

        # Initialize Wandb logger
        wandb_logger = WandbLogger( project=args.project_name,
                                    sync_tensorboard=False,
                                    name=name_id,
                                    config=config)

        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath= os.path.join(args.checkpoint_dir, name_id),
            filename='Utransformer-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min',
        )
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=50, verbose=True, mode="min")

        # Initialize model
        model = U_Transformer_Lightning(in_channels=6, classes=1)

        #load data
        train_loader, val_loader = initialize_data_loaders(args)

        # Trainer
        trainer = Trainer(
                    max_epochs=args.num_epochs,
                    accelerator="gpu",
                    devices=2, 
                    precision= 32,
                    log_every_n_steps=1,
                    logger=wandb_logger,
                    callbacks=[checkpoint_callback, early_stop_callback],
                )

        # Training
        trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)

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
    training_group.add_argument('--num_epochs', type=int, default=150, help='Number of epochs to train')
    training_group.add_argument('--dynamic_valid_patch_rate', default=True, type=lambda x: (str(x).lower() == 'true'), help='Dynamically update valid_patch_rate each epoch')

    # Checkpoint and logging arguments
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    log_group.add_argument('--project_name', type=str, default='U_Transformer_Segmentation', help='WandB project name')
    log_group.add_argument('--name_id', type=str, default='utransformer_experiment_pl', help='WandB run name')

    args = parser.parse_args()

    main(args)