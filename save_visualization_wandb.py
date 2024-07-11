import os
import torch
import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

from models.unetTransformer import U_Transformer_Lightning
from libs.Dataset_val import PatchDataGenerator

# Function to initialize data loaders
def initialize_data_loaders(args):
    print("Initializing data loaders...")
    val_dataset = PatchDataGenerator(
        data_dir=args.val_data_dir, 
        patch_size=args.patch_size, 
        overlap=args.overlap, 
        norm_type=args.norm_type, 
        hue_factor=args.hue_factor, 
        augment=args.augment,
        valid_patch_rate=1,
        test=False
    )
    
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty. Please check the data directory or preprocessing steps.")
    
    # Limit the dataset to the first 25 samples
    val_subset = Subset(val_dataset, range(min(len(val_dataset), 25)))
    
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Loaded {len(val_subset)} samples for validation.")
    return val_loader

# Function to load the model from a checkpoint
def load_model_from_checkpoint(checkpoint_path, device):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = U_Transformer_Lightning.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    print("Model loaded successfully.")
    return model

# Function to perform predictions
def predict(model, data_loader, device):
    print("Starting predictions...")
    model.eval()
    all_raw_predictions = []
    all_predictions = []
    val_imgs = []
    val_masks = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs, masks = data
            imgs, masks = imgs.to(device), masks.to(device)
            raw_preds = model(imgs)
            all_raw_predictions.extend(raw_preds.cpu())
            val_imgs.extend(imgs.cpu())
            val_masks.extend(masks.cpu())

            if raw_preds.shape == (2, raw_preds.shape[1], raw_preds.shape[2]):
                preds = torch.sigmoid(raw_preds)
                all_predictions.extend(preds.cpu())

            print(f"Processed batch {i+1}/{len(data_loader)}")
    print("Predictions completed.")
    return val_imgs, val_masks, all_raw_predictions, all_predictions

# Function to log predictions to WandB in the specified columns format
def log_predictions_to_wandb(project, run_id, data, labels, raw_predictions, predictions):
    print(f"Logging predictions to WandB run ID: {run_id}")
    wandb.init(entity='9bombs', project=project, id=run_id, resume="must")

    num_columns = 4 if not predictions else 5

    # Create a figure with 25 rows and the required number of columns
    fig, axes = plt.subplots(25, num_columns, figsize=(num_columns * 5, 100))

    for i, (input_data, label, raw_prediction) in enumerate(zip(data, labels, raw_predictions)):
        row = i

        # Get first 3 channels (RGB) and last 3 channels
        input_first3 = (input_data[:3].numpy().transpose(1, 2, 0) + 1) / 2  # Rescale from [-1, 1] to [0, 1]
        input_last3 = (input_data[-3:].numpy().transpose(1, 2, 0) + 1) / 2  # Rescale from [-1, 1] to [0, 1]
        label = label.numpy().squeeze()  # Squeeze the singleton dimension

        # Check if raw_prediction has shape (2, H, W) and use argmax if true
        if raw_prediction.shape[0] == 2:
            raw_prediction = torch.argmax(raw_prediction, dim=0, keepdim=True)

        raw_prediction = raw_prediction.numpy().squeeze()  # Squeeze the singleton dimension

        # Plot the first 3 channels
        axes[row, 0].imshow(input_first3)
        axes[row, 0].set_title("First 3 Channels (RGB)")
        axes[row, 0].axis("off")

        # Plot the last 3 channels
        axes[row, 1].imshow(input_last3)
        axes[row, 1].set_title("Last 3 Channels")
        axes[row, 1].axis("off")

        # Plot the label
        axes[row, 2].imshow(label, cmap='gray')
        axes[row, 2].set_title("Label")
        axes[row, 2].axis("off")

        # Plot the raw prediction
        axes[row, 3].imshow(raw_prediction, cmap='gray')
        axes[row, 3].set_title("Raw Prediction")
        axes[row, 3].axis("off")

        if predictions:
            prediction = predictions[i].numpy().squeeze()  # Squeeze the singleton dimension
            # Plot the sigmoid prediction
            axes[row, 4].imshow(prediction, cmap='gray')
            axes[row, 4].set_title("Sigmoid Prediction")
            axes[row, 4].axis("off")

        print(f"Processed image {i+1}/{len(data)}")

    # Adjust layout
    plt.tight_layout()

    # Save the plot as an image file
    plot_filename = "combined_plot.png"
    plt.savefig(plot_filename)
    plt.close(fig)

    # Log the plot to WandB
    wandb.log({"combined_plot": wandb.Image(plot_filename)})
    print("Logging to WandB completed.")


def main(args):
    print("Starting main process...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model from checkpoint
    model = load_model_from_checkpoint(args.checkpoint_file, device)

    # Initialize data loaders
    val_loader = initialize_data_loaders(args)

    # Perform predictions
    val_imgs, val_masks, raw_predictions, predictions = predict(model, val_loader, device)

    # Log predictions to WandB
    log_predictions_to_wandb(args.project, args.run_id, val_imgs, val_masks, raw_predictions, predictions)
    print("Main process completed.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict and log results for U-Transformer model.')

    # Data-related arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--val_data_dir', type=str, default='/projects/bcxi/nathanj/commonPatchData/validation/', help='Directory for validation data')
    data_group.add_argument('--patch_size', type=int, default=256, help='Patch size for data generator')
    data_group.add_argument('--overlap', type=int, default=15, help='Overlap for patches')
    data_group.add_argument('--norm_type', type=str, choices=['basic', 'imagenet', 'none'], default='basic', help='Normalization type')
    data_group.add_argument('--hue_factor', type=float, default=0.2, help='Hue factor for data augmentation')
    data_group.add_argument('--augment', default=True, type=lambda x: (str(x).lower() == 'true'), help='Whether to use data augmentation')
    data_group.add_argument('--num_workers', type=int, default=12, help='Number of workers for data loader')

    # Prediction-related arguments
    prediction_group = parser.add_argument_group('Prediction')
    prediction_group.add_argument('--checkpoint_file', type=str, default='/projects/bcxi/nathanj/DARPA_pytorch/checkpoints/utransformer_experiment_pl_20240618_140955/best_model.ckpt', help='Path to the checkpoint file for the model')
    prediction_group.add_argument('--batch_size', type=int, default=16, help='Batch size for validation')

    # WandB arguments
    wandb_group = parser.add_argument_group('WandB')
    wandb_group.add_argument('--project', type=str, required=True, help='WandB project name')
    wandb_group.add_argument('--run_id', type=str, default='rzbkqsf7', help='WandB run ID to log predictions')

    args = parser.parse_args()

    main(args)






############################# Old sigmoid ouput #############################
# original Utransformer experiment
# checkpoint Path: '/projects/bcxi/nathanj/DARPA_pytorch/checkpoints/utransformer_experiment_pl_20240705_113001/best_model.ckpt'
# project: U_Transformer_Segmentation
# run_id: u3wvt6vj


# Utransformer with scale experiment: 
# checkpoint Path: '/projects/bcxi/nathanj/DARPA_pytorch/checkpoints/scale_experiment_20240708_112512/best_model.ckpt'
# project: 'U_Transformer_Segmentation_scales'
# run_id: 'inajyz6q'


############################# Old non-sigmoid ouput #############################
# original Utransformer experiment
# checkpoint Path: '/projects/bcxi/nathanj/DARPA_pytorch/checkpoints/utransformer_experiment_pl_20240618_140955/best_model.ckpt'
# project: U_Transformer_Segmentation
# run_id: 32robwvu


# Utransformer with scale experiment: 
# checkpoint Path: '/projects/bcxi/nathanj/DARPA_pytorch/checkpoints/scale_experiment_20240701_131149/best_model.ckpt'
# project: 'U_Transformer_Segmentation_scales'
# run_id: '14wjecwv'

#python save_visualization_wandb.py --checkpoint_file '/projects/bcxi/nathanj/DARPA_pytorch/checkpoints/scale_experiment_further_20240710_115856/best_model.ckpt' --project 'U_Transformer_Segmentation_scales' --run_id '3qwawze1'


# Utransformer with scale Trained further from the previous scale experiment: 
# checkpoint Path: '/projects/bcxi/nathanj/DARPA_pytorch/checkpoints/scale_experiment_further_20240710_115856/best_model.ckpt'
# project: 'U_Transformer_Segmentation_scales'
# run_id: '3qwawze1'