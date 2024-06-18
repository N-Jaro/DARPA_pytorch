#!/bin/bash
# Set the account name
#SBATCH -A bcxi-tgirails
# Set the job name
#SBATCH --job-name=scale_trsnaformer
# Set the partition
#SBATCH --partition=gpu
# Set the number of nodes
#SBATCH --nodes=1
# Set the number of tasks per node
#SBATCH --ntasks=1
# Set the number of CPUs per task
#SBATCH --cpus-per-task=16
# Set the number of GPUs
#SBATCH --gpus=2
# Set the amount of memory
#SBATCH --mem=50GB
# Set the time limit (hh:mm:ss)
#SBATCH --time=48:00:00
# Set the output file
#SBATCH --output=transformer_scale_%j.out   


# Activate the Conda environment
source /u/nathanj/.bashrc
conda activate darpa_pytorch

cd /projects/bcxi/nathanj/DARPA_pytorch

# Run the training script
python training_scale.py --project_name "U_Transformer_Segmentation_scales" --name_id "scale_experiment_3" --checkpoint_file "/projects/bcxi/nathanj/DARPA_pytorch/checkpoints/Utransformer_256_32_w_val_constant_rate/u-transformer-val/loss=0.29.ckpt"
