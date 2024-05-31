import os
import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
import mlflow

from models.unet import UNet
from models.unetTransformer import U_Transformer

from libs.data_module import PatchDataModule
from libs.callbacks import ValidPatchRateCallback
from models.lit_training import LitTraining

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

tracking_uri = "https://criticalmaas.software-dev.ncsa.illinois.edu/"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'ncsa'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'hydrohpc'
mlflow.set_tracking_uri(uri=tracking_uri)
mlflow.set_experiment("nathan")

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

        # Initialize wandb
        wandb.init(
            project=args.project_name,
            sync_tensorboard=False,
            name=args.name_id,
            config=config
        )

        # Initialize model
        model = U_Transformer(in_channels=6, classes=1)

        # Initialize Lightning model
        lit_model = LitTraining(model, args.learning_rate)

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

        # Initialize data module
        data_module = PatchDataModule(
            train_data_dir=args.train_data_dir,
            val_data_dir=args.val_data_dir,
            patch_size=args.patch_size,
            overlap=args.overlap,
            norm_type=args.norm_type,
            hue_factor=args.hue_factor,
            augment=args.augment,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            valid_patch_rate=args.valid_patch_rate
        )
        
        valid_patch_callback = ValidPatchRateCallback(
            initial_patch_rate=args.valid_patch_rate, 
            rate_decay=0.005, 
            dynamic=args.dynamic_valid_patch_rate
        )

        # Trainer
        trainer = Trainer(
            max_epochs=args.num_epochs,
            callbacks=[checkpoint_callback, valid_patch_callback],
            accelerator="gpu",
            devices=2,  # <-- NEW
            # strategy="ddp",  # <-- NEW
            precision="16",
            logger=wandb_logger
        )

        # Training
        trainer.fit(lit_model, data_module)

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
